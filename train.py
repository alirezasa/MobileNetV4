"""Train a ConvNeXt-UIB (MobileNetV4 UIB blocks) network on CIFAR-10.

Uses timm for: the UIB block, LayerNorm2d, the AdamW optimizer factory (no weight decay
on norm/bias params), the cosine LR scheduler with warmup, label-smoothing loss and
accuracy/AverageMeter utilities.

Example:
    python train.py --norm bn --epochs 30 --out results/main_bn.json
"""
import argparse
import json
import os
import time

import torch
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.utils import AverageMeter, accuracy

from src.data import TensorLoader, load_cifar10
from src.models import count_params, create_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.environ.get('CIFAR_ROOT', './data'))
    p.add_argument('--model', default='tiny')
    p.add_argument('--block', default='convnext', help='UIB variant: convnext | extra_dw | ib | ffn')
    p.add_argument('--norm', default='bn', help='bn | ln | gn1 | none')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-epochs', type=float, default=2)
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--drop-path', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=None)
    p.add_argument('--train-subset', type=int, default=0, help='use only the first N training images (0 = all)')
    p.add_argument('--eval-train-mode', action='store_true',
                   help='also evaluate with BN using test-batch statistics (batch size = --batch-size)')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--no-channels-last', action='store_true',
                   help='keep NCHW memory format (channels_last makes BatchNorm ~2x faster on CPU)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/run.json')
    p.add_argument('--save-ckpt', default='')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, bn_batch_stats=False):
    """Evaluate in eval mode. With bn_batch_stats=True the BatchNorm layers normalise with the
    statistics of the current test batch (as in training) instead of their running averages;
    everything else (drop-path, dropout) stays in eval mode and running stats are not updated."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if bn_batch_stats:
        for m in bns:
            m.train()
            m.momentum_backup, m.momentum = m.momentum, 0.0  # momentum 0 -> running stats frozen
    loss_m, acc_m = AverageMeter(), AverageMeter()
    p0 = next(model.parameters())
    memory_format = torch.channels_last if p0.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    for x, y in loader:
        x, y = x.to(p0.device).contiguous(memory_format=memory_format), y.to(p0.device)
        out = model(x)
        loss_m.update(criterion(out, y).item(), x.size(0))
        acc_m.update(accuracy(out, y, topk=(1,))[0].item(), x.size(0))
    if bn_batch_stats:
        for m in bns:
            m.momentum = m.momentum_backup
            m.eval()
    model.train()
    return loss_m.avg, acc_m.avg


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    (xtr, ytr), (xte, yte) = load_cifar10(args.data)
    if args.train_subset:
        xtr, ytr = xtr[:args.train_subset], ytr[:args.train_subset]
    train_loader = TensorLoader(xtr, ytr, args.batch_size, train=True, seed=args.seed)
    test_loader = TensorLoader(xte, yte, 500, train=False)
    # small-batch BN is evaluated in train mode with the same small batch size
    test_loader_bs = TensorLoader(xte, yte, args.batch_size, train=False)

    model = create_model(args.model, block=args.block, norm=args.norm, drop_path_rate=args.drop_path).to(device)
    memory_format = torch.contiguous_format if args.no_channels_last else torch.channels_last
    model = model.to(memory_format=memory_format)
    n_params = count_params(model)
    print(f'model={args.model} block={args.block} norm={args.norm} params={n_params / 1e6:.3f}M')

    optimizer = create_optimizer_v2(model, opt='adamw', lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs * steps_per_epoch,
        lr_min=1e-6,
        warmup_t=int(args.warmup_epochs * steps_per_epoch),
        warmup_lr_init=1e-6,
        t_in_epochs=False,
    )
    train_criterion = LabelSmoothingCrossEntropy(args.smoothing)
    eval_criterion = torch.nn.CrossEntropyLoss()

    history = dict(args=vars(args), params=n_params, epochs=[])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device).contiguous(memory_format=memory_format), y.to(device)
            out = model(x)
            loss = train_criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            scheduler.step_update(step)
            loss_m.update(loss.item(), x.size(0))
            acc_m.update(accuracy(out.detach(), y, topk=(1,))[0].item(), x.size(0))
            if not torch.isfinite(loss):
                print('non-finite loss, stopping')
                break
        train_time = time.time() - t0
        test_loss, test_acc = evaluate(model, test_loader, eval_criterion)
        rec = dict(epoch=epoch + 1, lr=optimizer.param_groups[0]['lr'],
                   train_loss=loss_m.avg, train_acc=acc_m.avg,
                   test_loss=test_loss, test_acc=test_acc, epoch_time=train_time)
        if args.eval_train_mode:
            rec['test_loss_trainmode'], rec['test_acc_trainmode'] = evaluate(
                model, test_loader_bs, eval_criterion, bn_batch_stats=True)
        history['epochs'].append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, 'w') as f:
            json.dump(history, f, indent=1)
        if not torch.isfinite(torch.tensor(loss_m.avg)):
            break

    history['total_time'] = time.time() - t_start
    history['best_test_acc'] = max(e['test_acc'] for e in history['epochs'])
    history['final_test_acc'] = history['epochs'][-1]['test_acc']
    with open(args.out, 'w') as f:
        json.dump(history, f, indent=1)
    if args.save_ckpt:
        torch.save(model.state_dict(), args.save_ckpt)
    print(f"done: final test acc {history['final_test_acc']:.2f}  best {history['best_test_acc']:.2f}  "
          f"time {history['total_time'] / 60:.1f} min")


if __name__ == '__main__':
    main()
