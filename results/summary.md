## Experiment 1: full CIFAR-10, batch 128, 30 epochs

| norm | params | final test acc | best test acc | final train acc | final test loss | sec / epoch |
|---|---|---|---|---|---|---|
| BatchNorm | 1.91M | 87.49 | 87.49 | 93.81 | 0.439 | 107 |
| LayerNorm | 1.91M | 81.05 | 81.05 | 86.09 | 0.607 | 102 |

## Experiment 2: batch-size sensitivity (10k images, 10 epochs, lr = 2e-3·sqrt(B/128))

| norm | batch | test acc (eval mode) | test acc (BN batch stats) | train acc | test loss | sec / epoch |
|---|---|---|---|---|---|---|
| BatchNorm | 4 | 39.67 | 33.31 | 29.50 | 1.722 | 101 |
