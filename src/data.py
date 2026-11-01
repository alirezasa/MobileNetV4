"""CIFAR-10 loaded fully into memory as tensors, with vectorised augmentation.

Keeping the whole dataset (50k x 3 x 32 x 32 uint8 = 150 MB) in a tensor and doing
random-crop / flip on whole batches avoids DataLoader worker processes, which matters
when training on a CPU where every core is needed for the model itself.
"""
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_cifar10(root: str, download: bool = True):
    tr = CIFAR10(root, train=True, download=download)
    te = CIFAR10(root, train=False, download=download)
    to_t = lambda ds: (torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets))
    return to_t(tr), to_t(te)


class TensorLoader:
    """Minimal batch iterator over uint8 image tensors.

    train=True: shuffle + random crop (4 px reflect padding) + horizontal flip.
    """

    def __init__(self, images, labels, batch_size, train, drop_last=None, seed=0):
        self.x, self.y = images, labels
        self.bs = batch_size
        self.train = train
        self.drop_last = train if drop_last is None else drop_last
        self.g = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _augment(self, x):
        b = x.shape[0]
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        i = torch.randint(0, 9, (b,), generator=self.g)
        j = torch.randint(0, 9, (b,), generator=self.g)
        # gather the 32x32 crops with advanced indexing (no python loop)
        ar = torch.arange(32)
        rows = (i[:, None] + ar[None]).view(b, 1, 32, 1).expand(b, 3, 32, 40)
        cols = (j[:, None] + ar[None]).view(b, 1, 1, 32).expand(b, 3, 32, 32)
        x = x.gather(2, rows).gather(3, cols)
        flip = torch.rand(b, generator=self.g) < 0.5
        x[flip] = x[flip].flip(3)
        return x

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, generator=self.g) if self.train else torch.arange(n)
        for k in range(len(self)):
            bi = idx[k * self.bs:(k + 1) * self.bs]
            x = self.x[bi].float().div_(255)
            if self.train:
                x = self._augment(x)
            x = (x - self.mean) / self.std
            yield x, self.y[bi]
