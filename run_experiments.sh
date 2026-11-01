#!/usr/bin/env bash
# Reproduces every run in the report. On a 4-core CPU two lanes run in parallel (2 threads each).
# On a GPU just run the commands one after another (drop --threads).
set -e
mkdir -p results/logs checkpoints
DATA=${CIFAR_ROOT:-./data}
T=${THREADS:-2}

lane_a() {
  # Experiment 1 (main): BatchNorm, full CIFAR-10, 30 epochs
  python train.py --data $DATA --threads $T --norm bn --epochs 30 --save-ckpt checkpoints/main_bn.pt --out results/main_bn.json > results/logs/main_bn.log 2>&1
}

lane_b() {
  # Experiment 1 (main): LayerNorm, full CIFAR-10, 30 epochs
  python train.py --data $DATA --threads $T --norm ln --epochs 30 --save-ckpt checkpoints/main_ln.pt --out results/main_ln.json > results/logs/main_ln.log 2>&1
  # Experiment 2: batch-size sensitivity, 10k-image training subset, 10 epochs.
  # AdamW learning rate scaled with sqrt(batch/128).
  for bs in 4 16 128; do
    lr=$(python -c "print(2e-3*($bs/128)**0.5)")
    for norm in bn ln; do
      python train.py --data $DATA --threads $T --norm $norm --batch-size $bs --lr $lr --epochs 10 \
        --warmup-epochs 1 --train-subset 10000 --eval-train-mode \
        --out results/bs_${norm}_${bs}.json > results/logs/bs_${norm}_${bs}.log 2>&1
    done
  done
  # reference without any normalisation
  python train.py --data $DATA --threads $T --norm none --batch-size 128 --epochs 10 --warmup-epochs 1 \
    --train-subset 10000 --out results/bs_none_128.json > results/logs/bs_none_128.log 2>&1
}

lane_a & lane_b & wait
echo ALL DONE
