#!/usr/bin/env python3
# mmd_model_compare.py

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import os

from data_processing import load_data
from vae_model      import VAE, BetaVAE

# constants
MAX_IMAGES       = 1000   # subsample per replicate
HEURISTIC_SUBSET = 2000   # for median σ
SUBSET_SIZE      = 2000   # for memory-safe subsampling

def load_model_and_cfg(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    cfg.data.train_path = "data/raw/BBBC021_cleaned_preprocessed.h5"
    name = cfg.model.name
    if name == "VAE":
        model = VAE(input_channels=cfg.model.input_channels,
                    latent_dim=   cfg.model.latent_dim)
    elif name == "Beta_VAE":
        model = BetaVAE(input_channels=cfg.model.input_channels,
                        latent_dim=   cfg.model.latent_dim,
                        beta=         cfg.model.beta)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

def pdist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AA = np.sum(A*A, axis=1)[:, None]
    BB = np.sum(B*B, axis=1)[None, :]
    return AA + BB - 2*A.dot(B.T)

def compute_mmd_multi(X: np.ndarray, Y: np.ndarray, sigmas: list) -> float:
    D_xx = pdist2(X, X)
    D_yy = pdist2(Y, Y)
    D_xy = pdist2(X, Y)
    K_xx = np.zeros_like(D_xx)
    K_yy = np.zeros_like(D_yy)
    K_xy = np.zeros_like(D_xy)
    for s in sigmas:
        K_xx += np.exp(-D_xx/(2*s*s))
        K_yy += np.exp(-D_yy/(2*s*s))
        K_xy += np.exp(-D_xy/(2*s*s))
    return K_xx.mean() + K_yy.mean() - 2*K_xy.mean()

def gather_images(model, loader, device):
    orig, recon = [], []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Gathering images"):
            x = xb.to(device)
            mu, _ = model.encode(x)
            x_rec = model.decode(mu)
            m = getattr(x_rec, "mean", None)
            if m is not None and not callable(m):
                x_rec = m
            bs = x.shape[0]
            orig.append( x.cpu().numpy().reshape(bs, -1) )
            recon.append(x_rec.cpu().numpy().reshape(bs, -1))
    return np.concatenate(orig, axis=0), np.concatenate(recon, axis=0)

def main():
    p = argparse.ArgumentParser(
        description="Compare MMD baselines and multiple models"
    )
    p.add_argument(
        "--checkpoints","-c",
        nargs="+", required=True,
        help="List of VAE .pth checkpoints"
    )
    p.add_argument(
        "--split", default="test",
        choices=["train","val","test"],
        help="Which data split"
    )
    p.add_argument(
        "--reps","-k", type=int, default=5,
        help="Number of bootstrap replicates"
    )
    args = p.parse_args()

    # prepare loader once
    model0, cfg = load_model_and_cfg(args.checkpoints[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = load_data(cfg, split=args.split)

    # --------------------------------------------------
    # 1) Gather full images for baseline model
    print("Baseline model:", args.checkpoints[0])
    model0.to(device)
    orig_full, recon_full0 = gather_images(model0, loader, device)
    N_full, D = orig_full.shape
    print(f"Total images: {N_full}")

    # subsample indices for memory safety
    subs = np.random.choice(N_full, SUBSET_SIZE, replace=False)
    orig_imgs    = orig_full[subs]
    recon_imgs0  = recon_full0[subs]
    del orig_full, recon_full0
    N = SUBSET_SIZE

    # median-heuristic σ
    med = np.median(pdist2(orig_imgs, orig_imgs))
    sigma0 = np.sqrt(med)
    sigmas = [0.5*sigma0, sigma0, 1.5*sigma0, 2.0*sigma0]
    print("Using σ =", [f"{s:.4g}" for s in sigmas])

    # baseline: recon–orig, recon–recon, recon–gauss
    base_recs = []
    print("\nσ      Reconstruction-Original     Reconstrcution-Reconstruction       Reconstruction-Gaussian")
    for s in sigmas:
        v_ro, v_rr, v_rg = [], [], []
        for _ in range(args.reps):
            ir1 = np.random.randint(0, N, size=MAX_IMAGES)
            io  = np.random.randint(0, N, size=MAX_IMAGES)
            R1  = recon_imgs0[ir1]
            O   = orig_imgs[io]
            Z   = np.random.randn(MAX_IMAGES, D)

            v_ro.append(compute_mmd_multi(R1, O,  [s]))
            v_rr.append(compute_mmd_multi(R1, R1, [s]))
            v_rg.append(compute_mmd_multi(R1, Z,  [s]))

        m_ro, sd_ro = np.mean(v_ro), np.std(v_ro)
        m_rr, sd_rr = np.mean(v_rr), np.std(v_rr)
        m_rg, sd_rg = np.mean(v_rg), np.std(v_rg)
        print(f"{s:<5.2f}  {m_ro:8.4f}±{sd_ro:.4f}   "
              f"{m_rr:8.4f}±{sd_rr:.4f}   "
              f"{m_rg:8.4f}±{sd_rg:.4f}")
        base_recs.append([s, m_ro, sd_ro, m_rr, sd_rr, m_rg, sd_rg])

    # baseline DataFrame & tables
    base_df = pd.DataFrame(
        base_recs,
        columns=["sigma","ro_mean","ro_std","rr_mean","rr_std","rg_mean","rg_std"]
    ).set_index("sigma")
    base_tab = (
        base_df.assign(
            recon_orig  = base_df["ro_mean"].round(4).astype(str)+" \\pm "+base_df["ro_std"].round(4).astype(str),
            recon_recon = base_df["rr_mean"].round(4).astype(str)+" \\pm "+base_df["rr_std"].round(4).astype(str),
            recon_gauss = base_df["rg_mean"].round(4).astype(str)+" \\pm "+base_df["rg_std"].round(4).astype(str)
        )
        .loc[:, ["recon_orig", "recon_recon", "recon_gauss"]]
    )
    base_tab.index = base_tab.index.map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))
    print("\n### Baseline Table (Markdown)\n")
    print(base_tab.to_markdown())
    print("\n### Baseline Table (LaTeX)\n")
    print(base_tab.to_latex(index=True,escape=False, column_format="c c c c"))

    # baseline plot
    x = base_df.index.to_numpy()
    y_ro, e_ro = base_df["ro_mean"].to_numpy(), 2*base_df["ro_std"].to_numpy()
    y_rr, e_rr = base_df["rr_mean"].to_numpy(), 2*base_df["rr_std"].to_numpy()
    y_rg, e_rg = base_df["rg_mean"].to_numpy(), 2*base_df["rg_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(x, y_ro, yerr=e_ro, fmt='.-', capsize=4, label="Reconstruction–Original")
    ax.errorbar(x, y_rr, yerr=e_rr, fmt='.-', capsize=4, label="Reconstruction–Reconstruction")
    ax.errorbar(x, y_rg, yerr=e_rg, fmt='.-', capsize=4, label="Reconstruction–Gaussian")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("MMD²")
    short_name = os.path.basename(args.checkpoints[0])
    ax.set_title(f"Baseline (model={short_name})")
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("mmd_baseline.png", dpi=300)

    # --------------------------------------------------
    # 2) Multi-model recon–orig comparison
    multi_recs = []
    for ckpt in args.checkpoints:
        print("Processing model:", ckpt)
        model, _ = load_model_and_cfg(ckpt)
        model.to(device)
        _, recon_full = gather_images(model, loader, device)
        recon_sub = recon_full[subs]
        del recon_full

        for s in sigmas:
            vals = []
            for _ in range(args.reps):
                ir = np.random.randint(0, N, size=MAX_IMAGES)
                R1 = recon_sub[ir]
                io = np.random.randint(0, N, size=MAX_IMAGES)
                O  = orig_imgs[io]
                vals.append(compute_mmd_multi(R1, O, [s]))
            m, sd = np.mean(vals), np.std(vals)
            multi_recs.append({"model": ckpt, "sigma": s, "mean": m, "std": sd})

    dfm = pd.DataFrame(multi_recs)
    wide = (
        dfm.assign(
            text=lambda d: d["mean"].round(4).astype(str)+" \\pm "+d["std"].round(4).astype(str)
        )
        .assign(model=lambda d: d["model"].apply(os.path.basename))  # shorten names
        .pivot(index="sigma", columns="model", values="text")
    )
    wide.index = wide.index.map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))
    print("\n### Multi-Model Table (Markdown)\n")
    print(wide.to_markdown())
    print("\n### Multi-Model Table (LaTeX)\n")
    print(wide.to_latex(index = True, escape=False, column_format="c c c c"))

    fig, ax = plt.subplots(figsize=(6,4))
    for name, grp in dfm.groupby("model"):
        short = os.path.basename(name)
        x_m = grp["sigma"].to_numpy()
        y_m = grp["mean"].to_numpy()
        e_m = 2*grp["std"].to_numpy()
        ax.errorbar(x_m, y_m, yerr=e_m, fmt='.-', capsize=4, label=short)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("MMD² (recon vs orig)")
    ax.set_title("Model comparison")
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("mmd_models.png", dpi=300)

if __name__ == "__main__":
    main()
