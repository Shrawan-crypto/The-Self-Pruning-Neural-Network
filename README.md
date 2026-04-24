# Self-Pruning Neural Network on CIFAR-10

A feed-forward neural network that **learns to prune itself during training** using learnable gate parameters and L1 sparsity regularization — no post-training pruning step required.

---

## Overview

Standard neural network pruning removes unimportant weights *after* training. This project takes a different approach — every weight in the network has a learnable **gate parameter** that is trained alongside the weights. An L1 penalty on these gates encourages most of them to become exactly zero during training, effectively pruning the network on the fly.

The experiment is run on **CIFAR-10** (10-class image classification) across three values of the sparsity hyperparameter λ to demonstrate the sparsity–accuracy trade-off.

---

## How It Works

```
gate        = sigmoid(gate_score)       ∈ (0, 1)
pruned_w    = weight × gate             (element-wise)
output      = x @ pruned_w.T + bias

Total Loss  = CrossEntropyLoss + λ × Σ gate_ij
```

- When a gate → 0, its weight is effectively removed from the network
- The L1 penalty applies a **constant gradient** to every gate regardless of its size, driving many gates all the way to exactly zero
- λ controls the trade-off: higher λ = more pruning, lower accuracy

---

## Project Structure

```
self-pruning-network/
│
├── self_pruning_network.py   # Main script — all code in one file
├── requirements.txt          # Python dependencies
├── gate_distribution.png     # Output: gate histogram (generated on run)
├── report.md                 # Short report with results and analysis
└── README.md                 # This file
```

---

## Key Components

| Component | Description |
|---|---|
| `PrunableLinear` | Custom linear layer with learnable `gate_scores` per weight |
| `SelfPruningNet` | 4-layer feed-forward network (3072→1024→512→256→10) using `PrunableLinear` |
| `sparsity_loss()` | Computes L1 norm of all gate values across every layer |
| `train_one_epoch()` | Runs one epoch with `CE + λ × sparsity` total loss |
| `compute_sparsity()` | Reports % of gates below threshold `0.01` (considered pruned) |
| `run_experiment()` | Full training loop for one value of λ |
| `plot_gate_distribution()` | Saves gate value histogram for the best model |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/self-pruning-network.git
cd self-pruning-network

# Install dependencies
pip install -r requirements.txt
```

**Requirements**
```
torch==2.3.0
torchvision==0.18.0
numpy==2.0.0
matplotlib==3.9.0
```

> CIFAR-10 (~170 MB) is downloaded automatically on first run into `./data/`

---

## Usage

```bash
python self_pruning_network.py
```

The script will:
1. Download CIFAR-10 if not already cached
2. Train the network with three values of λ: `1e-5`, `1e-4`, `5e-4`
3. Print a results table after each experiment
4. Save `gate_distribution.png` for the best model

**Expected console output:**
```
Using device: cpu

============================================================
  Training with lambda = 1e-05   (30 epochs)
============================================================
  Epoch   1/30 | Total=2.3412  CE=2.3011  Sp=1847.3 | Test Acc=21.45%  Sparsity=0.0%
  Epoch   5/30 | Total=1.9823  CE=1.9204  Sp=1821.6 | Test Acc=34.12%  Sparsity=0.0%
  ...
  Epoch  30/30 | Total=1.5231  CE=1.5118  Sp=1134.2 | Test Acc=52.41%  Sparsity=12.3%

  Final — Test Acc: 52.41%  |  Sparsity: 12.3%
```

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) | Notes |
|:---:|:---:|:---:|:---|
| `1e-5` | 52.41% | 12.3% | Light pruning, near-baseline accuracy |
| `1e-4` | 49.73% | 58.6% | Best trade-off — recommended |
| `5e-4` | 44.18% | 94.8% | Aggressive pruning, accuracy drops |

> Replace these values with your actual console output after running the script.

---

## Gate Distribution Plot

After training, `gate_distribution.png` is saved showing the histogram of all gate values for the best model. A successful result shows:

- **Large spike at 0** — majority of weights pruned (gate < 0.01)
- **Secondary cluster near 1** — surviving active connections
- **Empty middle** — bimodal shape confirming hard binary-like pruning decisions


---

## Training Details

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Architecture | Feed-forward: 3072 → 1024 → 512 → 256 → 10 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Epochs | 30 per λ experiment |
| Batch size | 128 |
| Sparsity threshold | 0.01 (gate < 0.01 = pruned) |
| Device | Auto-detects CUDA, falls back to CPU |

---

## Windows Users

If you are on Windows, set `num_workers=0` and `pin_memory=False` in `get_cifar10_loaders()` to avoid DataLoader multiprocessing issues:

```python
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=False)
```

---

## Concepts Demonstrated

- Custom `nn.Module` with multiple learnable parameter tensors
- Gradient flow through element-wise gating operations
- L1 regularization for inducing true sparsity
- Sparsity–accuracy trade-off analysis across hyperparameter values
- PyTorch training loop with custom loss formulation
- Model evaluation and result visualization with matplotlib

---

## License

MIT License — free to use, modify, and distribute.
