"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters that
encourages sparsity via L1 regularization on sigmoid-gated weights.

Author: AI Engineer Case Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
from collections import defaultdict

# ─────────────────────────────────────────────
# PART 1 – PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate_scores
    tensor of the same shape as the weight matrix.

    Forward pass:
        gates        = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w     = weight ⊙ gates                (element-wise)
        output       = x @ pruned_w.T + bias

    Because all operations are differentiable, gradients flow through both
    `weight` and `gate_scores` automatically via autograd.

    A gate near 0 means the corresponding weight is effectively removed;
    a gate near 1 means the weight is fully active.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias – identical initialisation to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Gate scores: same shape as weight, initialised near +2 so that
        # sigmoid(2) ≈ 0.88 — most gates start "open" and must be learned shut.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map gate_scores → (0,1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Element-wise multiply weight by its gate
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Standard affine transformation – gradient flows through both tensors
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def gate_values(self) -> torch.Tensor:
        """Return the current gate values (no gradient)."""
        return torch.sigmoid(self.gate_scores)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────
# Neural Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Simple feed-forward network for CIFAR-10 (32×32×3 → 10 classes).
    All linear layers are PrunableLinear so that every weight has a gate.
    BatchNorm + ReLU are kept standard – only the Linear projections are gated.
    """

    def __init__(self):
        super().__init__()
        # Flatten 32×32×3 = 3072 → hidden layers → 10
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten
        return self.net(x)

    # ------------------------------------------------------------------
    def prunable_layers(self):
        """Iterate over all PrunableLinear layers in the model."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ─────────────────────────────────────────────
# PART 2 – Sparsity Loss
# ─────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of *all* gate values across every PrunableLinear layer.

    Why L1?
    -------
    The L1 norm penalises each gate proportionally to its magnitude but
    applies a *constant* gradient of ±1 regardless of value. This means even
    a gate that is already close to zero still receives the same push towards
    zero, unlike L2 which has a gradient proportional to the value itself and
    causes weights to shrink exponentially but never reach exactly zero.
    The result is that the L1 penalty naturally drives many gates to exactly 0,
    producing true sparsity rather than merely small values.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)   # ∈ (0,1), all positive
        total = total + gates.sum()                # L1 = sum of abs = sum (gates>0)
    return total


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def compute_sparsity(model, threshold=1e-2):
    """
    Fraction of weights whose gate value is below `threshold`.
    These connections are considered effectively pruned.
    """
    total_weights = pruned_weights = 0
    for layer in model.prunable_layers():
        gates = layer.gate_values()
        total_weights   += gates.numel()
        pruned_weights  += (gates < threshold).sum().item()
    return 100.0 * pruned_weights / total_weights


@torch.no_grad()
def collect_all_gates(model):
    """Return a flat numpy array of every gate value in the model."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.gate_values().cpu().numpy().ravel())
    return np.concatenate(all_gates)


# ─────────────────────────────────────────────
# PART 3 – Training & Evaluation
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Download / cache CIFAR-10 and return train + test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, lam, device):
    """
    Run one full epoch.

    Returns
    -------
    avg_total_loss, avg_ce_loss, avg_sp_loss
    """
    model.train()
    total_loss_sum = ce_loss_sum = sp_loss_sum = 0.0
    n_batches = len(loader)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)

        # Classification loss
        ce = F.cross_entropy(logits, labels)

        # Sparsity regularisation loss
        sp = sparsity_loss(model)

        # Total loss (Part 2 formula)
        loss = ce + lam * sp

        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        ce_loss_sum    += ce.item()
        sp_loss_sum    += sp.item()

    return (total_loss_sum / n_batches,
            ce_loss_sum    / n_batches,
            sp_loss_sum    / n_batches)


def run_experiment(lam, num_epochs=30, device="cpu", seed=42):
    """
    Train a SelfPruningNet with a given λ and return results dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_cifar10_loaders()

    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n{'='*60}")
    print(f"  Training with lambda = {lam}   ({num_epochs} epochs)")
    print(f"{'='*60}")

    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        total_l, ce_l, sp_l = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_acc  = compute_accuracy(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Total={total_l:.4f}  CE={ce_l:.4f}  Sp={sp_l:.1f} | "
                  f"Test Acc={val_acc:.2f}%  Sparsity={sparsity:.1f}%")
            history["epoch"].append(epoch)
            history["val_acc"].append(val_acc)
            history["sparsity"].append(sparsity)

    final_acc     = compute_accuracy(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    gate_vals     = collect_all_gates(model)

    print(f"\n  Final — Test Acc: {final_acc:.2f}%  |  Sparsity: {final_sparsity:.1f}%")

    return {
        "lam"          : lam,
        "model"        : model,
        "final_acc"    : final_acc,
        "final_sparsity": final_sparsity,
        "gate_vals"    : gate_vals,
        "history"      : history,
    }


def plot_gate_distribution(results_list, best_idx, save_path="gate_distribution.png"):
    """
    Plot the distribution of final gate values for the best-λ model.
    A successful result shows a large spike at 0 and a secondary cluster near 1.
    """
    best = results_list[best_idx]
    gates = best["gate_vals"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gates, bins=100, color="#2563EB", edgecolor="white", linewidth=0.3, alpha=0.85)

    ax.set_title(
        f"Gate Value Distribution — lambda = {best['lam']}\n"
        f"(Test Acc: {best['final_acc']:.2f}%  |  Sparsity: {best['final_sparsity']:.1f}%)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=11)
    ax.set_ylabel("Number of Weights", fontsize=11)
    ax.set_xlim(-0.02, 1.02)

    # Annotate the two regions
    near_zero = (gates < 0.01).sum()
    near_one  = (gates > 0.9).sum()
    ax.annotate(
        f"Pruned\n({near_zero:,} weights)",
        xy=(0.005, ax.get_ylim()[1] * 0.6),
        fontsize=9, color="#DC2626", ha="left",
    )
    ax.annotate(
        f"Active\n({near_one:,} weights)",
        xy=(0.92, ax.get_ylim()[1] * 0.3),
        fontsize=9, color="#16A34A", ha="center",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Gate distribution plot saved → {save_path}")


def print_results_table(results_list):
    print("\n" + "─"*55)
    print(f"  {'Lambda':>10}  {'Test Accuracy':>14}  {'Sparsity (%)':>13}")
    print("─"*55)
    for r in results_list:
        print(f"  {r['lam']:>10}  {r['final_acc']:>13.2f}%  {r['final_sparsity']:>12.1f}%")
    print("─"*55)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Three λ values: low / medium / high
    lambdas = [1e-5, 1e-4, 5e-4]

    all_results = []
    for lam in lambdas:
        result = run_experiment(lam=lam, num_epochs=30, device=device)
        all_results.append(result)

    # Summary table
    print_results_table(all_results)

    # Find "best" model — highest accuracy (low-λ usually wins here)
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]["final_acc"])

    # Gate distribution plot for best model
    plot_gate_distribution(all_results, best_idx, save_path="gate_distribution.png")

    print("\nDone! Check gate_distribution.png for the gate histogram.")