"""BatchNorm vs LayerNorm: controlled experiments that need no training.

1. Which axes each layer normalises over (checked numerically against manual formulas).
2. Batch dependence: the output for one fixed image under BN (train mode) changes with
   the other images in the batch; under LN it never does.
3. Batch size 1: BN cannot compute a variance from a single value per channel.
4. Train/eval asymmetry: BN uses batch stats in training and running averages at test time.
5. Inference cost: BN is a fixed affine map at test time and can be folded into the
   preceding conv; LN's statistics depend on the input, so it cannot.

Writes results/bn_vs_ln_analysis.json and figures in results/figures/.
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

from src.data import TensorLoader, load_cifar10
from src.models import create_model
from src.plotstyle import COLORS, INK2, LABELS, plt

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.environ.get('CIFAR_ROOT', './data'))
ap.add_argument('--ckpt-dir', default='checkpoints')
ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
torch.manual_seed(0)
torch.set_num_threads(args.threads)
out = {}


def load_model(norm):
    """Trained checkpoint from train.py if present, else a fresh model whose layer-scale
    gammas are set to 1 so that the residual branches are not silenced at init."""
    m = create_model('tiny', norm=norm, drop_path_rate=0.0)
    path = os.path.join(args.ckpt_dir, f'main_{norm}.pt')
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location='cpu'))
        out.setdefault('models', {})[norm] = path
    else:
        for mod in m.modules():
            if hasattr(mod, 'gamma'):
                mod.gamma.data.fill_(1.0)
        out.setdefault('models', {})[norm] = 'random init (layer scale = 1)'
    return m.eval()


def freeze_bn_stats(m):
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.momentum = 0.0


(_, _), (xte, yte) = load_cifar10(args.data)
test_images = torch.cat([x for x, _ in TensorLoader(xte, yte, 1000, train=False)])

# --------------------------------------------------------------------------- 1. axes
N, C, H, W = 8, 16, 6, 6
x = torch.randn(N, C, H, W) * 3 + 1
bn, ln, gn1 = nn.BatchNorm2d(C, affine=False), LayerNorm2d(C, eps=1e-5, affine=False), nn.GroupNorm(1, C, affine=False)
bn.train()
manual_bn = (x - x.mean((0, 2, 3), keepdim=True)) / torch.sqrt(x.var((0, 2, 3), unbiased=False, keepdim=True) + 1e-5)
manual_ln = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-5)
manual_gn = (x - x.mean((1, 2, 3), keepdim=True)) / torch.sqrt(x.var((1, 2, 3), unbiased=False, keepdim=True) + 1e-5)
out['axes_check'] = {
    'BatchNorm2d == normalise over (N,H,W) per channel': (bn(x) - manual_bn).abs().max().item(),
    'LayerNorm2d (ConvNeXt) == normalise over C per (n,h,w)': (ln(x) - manual_ln).abs().max().item(),
    'GroupNorm(1) == normalise over (C,H,W) per sample': (gn1(x) - manual_gn).abs().max().item(),
    'number of statistics (mean,var pairs) BN / LN / GN1': [C, N * H * W, N],
}

# --------------------------------------------------------------------------- 2. batch dependence
pool = test_images[:2048]
target = pool[:1]
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
dep = {}
models = {n: load_model(n) for n in ['bn', 'ln']}
for name, m in models.items():
    m = copy.deepcopy(m).train()
    freeze_bn_stats(m)  # probe batches must not move the running stats
    dep[name] = {}
    with torch.no_grad():
        for bs in batch_sizes:
            logits = []
            for t in range(10):  # same target image, 10 different random batch partners
                idx = torch.randint(1, len(pool), (bs - 1,))
                batch = torch.cat([target, pool[idx]])
                logits.append(m(batch)[0])
            logits = torch.stack(logits)
            # relative spread of the target's logits across batch compositions
            dep[name][bs] = (logits.std(0).mean() / logits.abs().mean()).item()
            # how often does the predicted class of the same image flip?
            dep.setdefault(name + '_pred_changes', {})[bs] = int((logits.argmax(1) != logits.argmax(1).mode().values).sum())
out['batch_dependence_rel_std'] = dep

fig, ax = plt.subplots(figsize=(5.6, 3.4))
for name in ['bn', 'ln']:
    ys = [dep[name][b] for b in batch_sizes]
    ax.plot(batch_sizes, ys, marker='o', ms=5, color=COLORS[name], label=LABELS[name])
ax.set_xscale('log', base=2)
ax.set_xticks(batch_sizes, [str(b) for b in batch_sizes])
ax.set_xlabel('batch size')
ax.set_ylabel('relative std of logits\nfor one fixed image')
ax.set_title('Same image, different batch partners (train mode)')
ax.legend()
ax.set_ylim(bottom=0)
ax.annotate('LN: exactly 0 for every batch size', (batch_sizes[3], 0), xytext=(0, 14),
            textcoords='offset points', color=INK2, fontsize=8)
fig.savefig('results/figures/batch_dependence.png')
plt.close(fig)

# --------------------------------------------------------------------------- 3. batch size 1
res = {}
for name, m in models.items():
    m = copy.deepcopy(m).train()
    try:
        with torch.no_grad():
            m(test_images[:1])
        res[name] = 'ok'
    except Exception as e:  # BatchNorm1d in the head sees one value per channel
        res[name] = f'{type(e).__name__}: {str(e).splitlines()[0]}'
out['train_mode_batch_size_1'] = res

# --------------------------------------------------------------------------- 4. train/eval asymmetry
# BN at test time normally uses running averages (eval mode). If it instead uses the
# statistics of the test batch (train-mode BN), the prediction for an image depends on the
# test batch size and on which other images are in the batch. LN behaves identically.
def test_acc(m, bs, batch_stats):
    m = copy.deepcopy(m).eval()
    if batch_stats:
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.train(); mod.momentum = 0.0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_images), bs):
            correct += (m(test_images[i:i + bs]).argmax(1) == yte[i:i + bs]).sum().item()
    return 100 * correct / len(test_images)


eval_bs = [2, 4, 16, 128, 1000]
acc = {'bn_running_stats': test_acc(models['bn'], 500, False), 'ln': test_acc(models['ln'], 500, False)}
acc['bn_batch_stats'] = {b: test_acc(models['bn'], b, True) for b in eval_bs}
acc['ln_train_mode'] = {b: test_acc(models['ln'], b, True) for b in [2, 1000]}
out['test_acc_running_vs_batch_stats'] = acc

fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(eval_bs, [acc['bn_batch_stats'][b] for b in eval_bs], marker='o', ms=5, color=COLORS['bn'],
        label='BN, test-batch statistics')
ax.axhline(acc['bn_running_stats'], color=COLORS['bn'], lw=1.2, ls='--', label='BN, running statistics (eval mode)')
ax.axhline(acc['ln'], color=COLORS['ln'], lw=2, label='LN (any batch size)')
ax.set_xscale('log', base=2)
ax.set_xticks(eval_bs, [str(b) for b in eval_bs])
ax.set_xlabel('test batch size')
ax.set_ylabel('test accuracy (%)')
ax.set_title('Inference with batch statistics')
ax.legend(fontsize=8, loc='lower right')
fig.savefig('results/figures/bn_test_batch_stats.png')
plt.close(fig)


# --------------------------------------------------------------------------- 5. BN folding & latency
def fold_bn(model):
    """Fold every Conv2d->BatchNorm2d pair (timm ConvNormAct: .conv + .bn) into the conv."""
    model = copy.deepcopy(model).eval()
    for mod in model.modules():
        if hasattr(mod, 'conv') and hasattr(mod, 'bn') and isinstance(mod.bn, nn.BatchNorm2d):
            conv, b = mod.conv, mod.bn
            scale = b.weight / torch.sqrt(b.running_var + b.eps)
            fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                              conv.padding, conv.dilation, conv.groups, bias=True)
            fused.weight.data = conv.weight * scale.view(-1, 1, 1, 1)
            cb = conv.bias if conv.bias is not None else torch.zeros_like(b.running_mean)
            fused.bias.data = (cb - b.running_mean) * scale + b.bias
            mod.conv = fused
            # keep the activation (timm BatchNormAct2d applies act after norm)
            act = mod.bn.act if hasattr(mod.bn, 'act') else nn.Identity()
            drop = mod.bn.drop if hasattr(mod.bn, 'drop') else nn.Identity()
            mod.bn = nn.Sequential(drop, act)
    return model


def latency(model, x, reps=30):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t) / reps * 1000


with torch.no_grad():
    mbn = models['bn']
    mfold = fold_bn(mbn)
    mln = models['ln']
    xb = test_images[:32]
    n_bn_left = sum(isinstance(mm, nn.BatchNorm2d) for mm in mfold.modules())
    out['bn_folding'] = {
        'max_abs_diff_after_folding': (mbn(xb) - mfold(xb)).abs().max().item(),
        'max_abs_logit': mbn(xb).abs().max().item(),
        'test_acc_folded': test_acc(mfold, 500, False),
        'bn2d_layers_before': sum(isinstance(mm, nn.BatchNorm2d) for mm in mbn.modules()),
        'bn2d_layers_after': n_bn_left,
        'latency_ms_batch32': {},
    }
    for fmt_name, fmt in [('nchw', torch.contiguous_format), ('channels_last', torch.channels_last)]:
        xf = xb.contiguous(memory_format=fmt)
        for name, mm in [('bn', mbn), ('bn_folded', mfold), ('ln', mln)]:
            mm = copy.deepcopy(mm).to(memory_format=fmt)
            out['bn_folding']['latency_ms_batch32'][f'{name}_{fmt_name}'] = latency(mm, xf)

with open('results/bn_vs_ln_analysis.json', 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))

# --------------------------------------------------------------------------- figure: normalisation axes
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
Nn, Cc, S = 5, 6, 4  # batch, channels, spatial positions drawn per cell
titles = ['BatchNorm2d\nper channel, over (N, H, W)', 'LayerNorm2d (ConvNeXt)\nper pixel, over C',
          'GroupNorm(1)\nper sample, over (C, H, W)']
cols = [COLORS['bn'], COLORS['ln'], COLORS['gn1']]
for k, ax in enumerate(axes):
    ax.set_xlim(0, Cc); ax.set_ylim(0, Nn); ax.set_aspect('equal'); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(Nn):
        for j in range(Cc):
            for q in range(S):  # each (n, c) cell = one feature map, drawn as S spatial stripes
                hit = (k == 0 and j == 2) or (k == 1 and i == 1 and q == 1) or (k == 2 and i == 1)
                ax.add_patch(plt.Rectangle((j + .08 + q * .84 / S, Nn - 1 - i + .08), .84 / S - .03, .84,
                                           color=cols[k] if hit else '#e4e3df', lw=0))
    ax.set_title(titles[k], fontsize=9.5)
    ax.set_xlabel('channels C →', fontsize=9)
    ax.set_ylabel('batch N →', fontsize=9)
fig.text(0.5, -0.02, 'Each cell is one feature map (sample n, channel c); its stripes are the H×W spatial positions. '
         'Coloured elements share one mean and variance. Only BatchNorm mixes different samples.',
         ha='center', fontsize=8.5, color=INK2)
fig.savefig('results/figures/norm_axes.png')
plt.close(fig)
