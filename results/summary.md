## Experiment 1: full CIFAR-10, batch 128, 30 epochs

| norm | params | final test acc | best test acc | final train acc | final test loss | sec / epoch |
|---|---|---|---|---|---|---|
| BatchNorm | 1.91M | 87.49 | 87.49 | 93.81 | 0.439 | 107 |
| LayerNorm | 1.91M | 81.05 | 81.05 | 86.09 | 0.607 | 102 |
| GroupNorm(1) | 1.91M | 81.58 | 81.67 | 87.46 | 0.591 | 101 |

## Experiment 2: batch-size sensitivity (10k images, 10 epochs, lr = 2e-3·sqrt(B/128))

| norm | batch | test acc (eval mode) | test acc (BN batch stats) | train acc | test loss | sec / epoch |
|---|---|---|---|---|---|---|
| BatchNorm | 4 | 58.50 | 52.09 | 50.41 | 1.215 | 102 |
| LayerNorm | 4 | 58.78 | — | 59.18 | 1.183 | 103 |
| BatchNorm | 16 | 63.18 | 60.90 | 61.67 | 1.070 | 41 |
| LayerNorm | 16 | 58.63 | — | 58.46 | 1.194 | 40 |
| BatchNorm | 128 | 63.76 | 63.59 | 64.85 | 1.050 | 24 |
| LayerNorm | 128 | 56.73 | — | 56.23 | 1.237 | 21 |
| No norm | 128 | 38.13 | — | 35.12 | 1.799 | 15 |
