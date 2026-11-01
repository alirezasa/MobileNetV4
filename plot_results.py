"""Make the report figures and a markdown summary table from results/*.json."""
import glob
import json
import os

from src.plotstyle import COLORS, INK2, LABELS, plt

R = 'results'
F = os.path.join(R, 'figures')
os.makedirs(F, exist_ok=True)


def load(path):
    with open(path) as f:
        return json.load(f)


def col(h, k):
    return [e[k] for e in h['epochs']]


lines = []

# ------------------------------------------------------------------ experiment 1: main curves
main = {n: load(f'{R}/main_{n}.json') for n in ['bn', 'ln', 'gn1'] if os.path.exists(f'{R}/main_{n}.json')}
if main:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for n, h in main.items():
        ep = col(h, 'epoch')
        for ax, key in zip(axes, ['loss', 'acc']):
            ax.plot(ep, col(h, f'train_{key}'), color=COLORS[n], lw=1.3, ls='--', label=f'{LABELS[n]} train')
            ax.plot(ep, col(h, f'test_{key}'), color=COLORS[n], lw=2, label=f'{LABELS[n]} test')
        # direct label on the final test accuracy
        axes[1].annotate(f"{h['epochs'][-1]['test_acc']:.2f}%", (ep[-1], h['epochs'][-1]['test_acc']),
                         xytext=(4, {'bn': -3, 'ln': 6, 'gn1': -9}[n]), textcoords='offset points', fontsize=8, color=INK2)
    axes[0].set_title('Loss'); axes[0].set_ylabel('cross-entropy')
    axes[1].set_title('Accuracy'); axes[1].set_ylabel('top-1 accuracy (%)')
    for ax in axes:
        ax.set_xlabel('epoch')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.text(0.5, -0.07, 'Train loss includes label smoothing (0.1), drop-path and augmentation; test loss is plain cross-entropy.',
             ha='center', fontsize=8, color=INK2)
    fig.savefig(f'{F}/main_curves.png')
    plt.close(fig)

    # learning-rate schedule
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    h = next(iter(main.values()))
    ax.plot(col(h, 'epoch'), col(h, 'lr'), color=COLORS['bn'])
    ax.set_xlabel('epoch'); ax.set_ylabel('learning rate (end of epoch)'); ax.set_title('Warmup + cosine schedule')
    fig.savefig(f'{F}/lr_schedule.png')
    plt.close(fig)

    lines += ['## Experiment 1: full CIFAR-10, batch 128, 30 epochs', '',
              '| norm | params | final test acc | best test acc | final train acc | final test loss | sec / epoch |',
              '|---|---|---|---|---|---|---|']
    for n, h in main.items():
        e = h['epochs'][-1]
        avg_t = sum(col(h, 'epoch_time')) / len(h['epochs'])
        lines.append(f"| {LABELS[n]} | {h['params'] / 1e6:.2f}M | {e['test_acc']:.2f} | {max(col(h, 'test_acc')):.2f} | "
                     f"{e['train_acc']:.2f} | {e['test_loss']:.3f} | {avg_t:.0f} |")
    lines.append('')

# ------------------------------------------------------------------ experiment 2: batch size
bs_runs = {}
for p in glob.glob(f'{R}/bs_*.json'):
    _, norm, bs = os.path.basename(p)[:-5].split('_')
    bs_runs[(norm, int(bs))] = load(p)
if bs_runs:
    sizes = sorted({b for (n, b) in bs_runs if n in ('bn', 'ln')})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for n in ['bn', 'ln']:
        pts = [(b, bs_runs[(n, b)]['epochs'][-1]['test_acc']) for b in sizes if (n, b) in bs_runs]
        if pts:
            ax.plot(*zip(*pts), marker='o', ms=6, color=COLORS[n], label=f'{LABELS[n]} (eval mode)')
    pts = [(b, bs_runs[('bn', b)]['epochs'][-1].get('test_acc_trainmode')) for b in sizes if ('bn', b) in bs_runs]
    pts = [p for p in pts if p[1] is not None]
    if pts:
        ax.plot(*zip(*pts), marker='s', ms=5, lw=1.3, ls='--', color=COLORS['bn'],
                label='BatchNorm, test-batch statistics')
    if ('none', 128) in bs_runs:
        ax.plot([128], [bs_runs[('none', 128)]['epochs'][-1]['test_acc']], marker='D', ms=7, ls='',
                color=COLORS['none'], label='No normalisation')
    ax.set_xscale('log', base=2); ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel('training batch size'); ax.set_ylabel('test accuracy after 10 epochs (%)')
    ax.set_title('Batch-size sensitivity (10k training images)')
    ax.legend(fontsize=8)

    ax = axes[1]
    small = min(sizes)
    for n in ['bn', 'ln']:
        if (n, small) in bs_runs:
            h = bs_runs[(n, small)]
            ax.plot(col(h, 'epoch'), col(h, 'test_acc'), color=COLORS[n], label=f'{LABELS[n]} test (eval mode)')
            ax.plot(col(h, 'epoch'), col(h, 'train_acc'), color=COLORS[n], lw=1.3, ls='--', label=f'{LABELS[n]} train')
    ax.set_xlabel('epoch'); ax.set_ylabel('accuracy (%)'); ax.set_title(f'Batch size {small}: accuracy per epoch')
    ax.legend(fontsize=8)
    fig.savefig(f'{F}/batch_size.png')
    plt.close(fig)

    lines += ['## Experiment 2: batch-size sensitivity (10k images, 10 epochs, lr = 2e-3·sqrt(B/128))', '',
              '| norm | batch | test acc (eval mode) | test acc (BN batch stats) | train acc | test loss | sec / epoch |',
              '|---|---|---|---|---|---|---|']
    for (n, b), h in sorted(bs_runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        e = h['epochs'][-1]
        tm = e.get('test_acc_trainmode')
        tm = f'{tm:.2f}' if tm is not None and n == 'bn' else '—'
        avg_t = sum(col(h, 'epoch_time')) / len(h['epochs'])
        lines.append(f"| {LABELS[n]} | {b} | {e['test_acc']:.2f} | {tm} | {e['train_acc']:.2f} | {e['test_loss']:.3f} | {avg_t:.0f} |")
    lines.append('')

with open(f'{R}/summary.md', 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
