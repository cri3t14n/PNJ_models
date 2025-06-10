#!/usr/bin/env python3
"""
extract_basis.py

Load a trained KAN model, extract its learned B-spline basis functions,
and visualise both the raw bases and the resulting activation curves.

Usage
-----
$ python extract_basis.py          # assumes the checkpoint path below exists
"""

import os
import torch
import matplotlib.pyplot as plt
from kan import KAN
from kan.spline import B_batch, coef2curve   # ← extend_grid no longer needed

# ────────────────────────────────────────────────────────────────────────────────
# 1) Set up device & load model
# ────────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️  width/grid/k/seed must match the model that produced the checkpoint
model = KAN(width=[2, 5, 20], grid=10, k=5, seed=1, device=device).to(device)

ckpt_path = "KAN_4/model/final_model.pth"
if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

# ────────────────────────────────────────────────────────────────────────────────
# 2) Helper to grab spline parameters from a KAN layer
# ────────────────────────────────────────────────────────────────────────────────
def get_layer_params(layer):
    """
    Returns
    -------
    grid : Tensor, shape [in_dim, G + 2k]          (knot vector incl. end knots)
    coef : Tensor, shape [in_dim, out_dim, G + k]  (spline coefficients)
    """
    grid = layer.grid.detach().cpu()
    coef = layer.coef.detach().cpu()
    return grid, coef

# ────────────────────────────────────────────────────────────────────────────────
# 3) Plot raw B-spline basis functions
# ────────────────────────────────────────────────────────────────────────────────
def plot_raw_bases(grid, k, filename):
    knots   = grid[0]                                           # first input dim
    x_eval  = torch.linspace(knots.min(), knots.max(), 500)     # [500]
    B       = B_batch(x_eval.unsqueeze(1), grid, k=k)           # [500,in_dim,G+k]
    B       = B[:, 0, :].numpy()                                # [500,G+k]

    plt.figure(figsize=(6, 4))
    for i in range(B.shape[1]):
        plt.plot(x_eval.numpy(), B[:, i], label=f"B{i}")
    plt.title("Raw B-spline Basis Functions")
    plt.xlabel("x")
    plt.ylabel("B_i(x)")
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize="small")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# 4) Plot learned activation curves φ(x)
# ────────────────────────────────────────────────────────────────────────────────
def plot_learned_curves(grid, coef, k, filename):
    knots   = grid[0]
    x_eval  = torch.linspace(knots.min(), knots.max(), 500)
    Y       = coef2curve(x_eval.unsqueeze(1), grid, coef, k)    # [500,in_dim,out_dim]
    Y       = Y[:, 0, :].numpy()                                # [500,out_dim]

    plt.figure(figsize=(6, 4))
    for j in range(Y.shape[1]):
        plt.plot(x_eval.numpy(), Y[:, j], label=f"Neuron {j}")
    plt.title("Learned Activation Curves")
    plt.xlabel("x")
    plt.ylabel("φ(x)")
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize="small")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# 5) Generate & save plots for each KAN layer
# ────────────────────────────────────────────────────────────────────────────────
output_dir = "KAN_4/bases"
os.makedirs(output_dir, exist_ok=True)

for idx, layer in enumerate(model.act_fun):
    grid, coef = get_layer_params(layer)
    raw_fn     = os.path.join(output_dir, f"layer{idx}_raw_bases.png")
    curve_fn   = os.path.join(output_dir, f"layer{idx}_learned_curves.png")

    plot_raw_bases(grid, model.k, raw_fn)
    plot_learned_curves(grid, coef, model.k, curve_fn)
    print(f"Layer {idx}: saved {raw_fn}  and  {curve_fn}")

print("Extraction and visualisation complete.")
