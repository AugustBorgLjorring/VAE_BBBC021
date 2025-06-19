#!/usr/bin/env python3
# mmd_model_compare_with_tests.py

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import os
import itertools
from scipy.stats import t, wilcoxon

from data_loading_well import load_data_by_well
from vae_model import VAESmall, VAEMedium, VAELarge, BetaVAE, VAEPlus

# Constants
MAX_IMAGES       = 2000   # number of images per MMD replicate
HEURISTIC_SUBSET = 2000   # for median σ
SUBSET_SIZE      = 10000   # for MMD bootstrap pool

def load_model_and_cfg(checkpoint_path: str):
    """
    Load a checkpoint, reconstruct the corresponding VAE model (small, medium, large, or BetaVAE),
    and return both the model (in eval mode) and its configuration.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    cfg.data.train_path = "data/raw/BBBC021_dataset_cleaned_maxnorm_68.h5"

    name = cfg.model.name
    latent_dim     = cfg.model.latent_dim
    input_channels = cfg.model.input_channels

    if name == "VAESmall":
        model = VAESmall(in_channels=input_channels, latent_dim=latent_dim)
    elif name == "VAEMedium":
        model = VAEMedium(in_channels=input_channels, latent_dim=latent_dim)
    elif name == "VAELarge":
        model = VAELarge(in_channels=input_channels, latent_dim=latent_dim)
    elif name == "BetaVAE":
        beta = cfg.model.beta if "beta" in cfg.model else 1.0
        model = BetaVAE(in_channels=input_channels, latent_dim=latent_dim, beta=beta)
    elif name == "VAE+":
        model = VAEPlus(
            in_channels=input_channels,
            latent_dim=latent_dim,
            beta=cfg.model.beta if "beta" in cfg.model else 1.0,
            T=cfg.model.T if "T" in cfg.model else 1.0,
            use_adverserial =cfg.model.use_adverserial if "use_adverserial" in cfg.model else True
        )
    else:
        raise ValueError(f"Unknown model name in config: {name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

def pdist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    return AA + BB - 2 * A.dot(B.T)

def compute_mmd_multi(X: np.ndarray, Y: np.ndarray, sigmas: list) -> float:
    D_xx = pdist2(X, X)
    D_yy = pdist2(Y, Y)
    D_xy = pdist2(X, Y)
    K_xx = np.zeros_like(D_xx)
    K_yy = np.zeros_like(D_yy)
    K_xy = np.zeros_like(D_xy)
    for s in sigmas:
        K_xx += np.exp(-D_xx / (2 * s * s))
        K_yy += np.exp(-D_yy / (2 * s * s))
        K_xy += np.exp(-D_xy / (2 * s * s))
    return np.sqrt(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())

def gather_images(model, loader, device):
    orig_list, recon_list = [], []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Gathering images"):
            x = xb.to(device)
            mu, _ = model.encode(x)
            x_rec = model.decode(mu)
            if hasattr(x_rec, "mean") and not callable(x_rec.mean):
                x_rec = x_rec.mean
            bs = x.shape[0]
            orig_list.append(x.cpu().numpy().reshape(bs, -1))
            recon_list.append(x_rec.cpu().numpy().reshape(bs, -1))
    orig = np.concatenate(orig_list, axis=0)
    recon = np.concatenate(recon_list, axis=0)
    return orig, recon

def main():
    p = argparse.ArgumentParser(
        description="Compute MMD and run statistical tests; output LaTeX tables"
    )
    p.add_argument(
        "--checkpoints", "-c",
        nargs="+", required=True,
        help="List of VAE .pth checkpoint paths (e.g. small.pth medium.pth large.pth)"
    )
    p.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate on"
    )
    p.add_argument(
        "--reps", "-k", type=int, default=30,
        help="Number of bootstrap replicates (recommend ≥10 for reliable tests)"
    )
    args = p.parse_args()

    # ----------------------------------------------------------------------------
    # FIGURE 1: Toy Example (MMD on Two 1-D Gaussians)
    # ----------------------------------------------------------------------------
    np.random.seed(0)
    N_toy = 500
    gaussA = np.random.randn(N_toy, 1) * 1.0 + 0.0  # N(0,1)
    gaussB = np.random.randn(N_toy, 1) * 1.0 + 2.0  # N(2,1)

    toy_sigmas = np.linspace(0.1, 2.0, 30)
    mmd_AA = []
    mmd_BA = []
    for s in toy_sigmas:
        mmd_AA.append(compute_mmd_multi(gaussA, gaussA, [s]))
        mmd_BA.append(compute_mmd_multi(gaussB, gaussA, [s]))

    fig_toy, ax_toy = plt.subplots(figsize=(5, 3))
    ax_toy.plot(toy_sigmas, mmd_AA, label=r"N(0,1) vs N(0,1)")
    ax_toy.plot(toy_sigmas, mmd_BA, label=r"N(2,1) vs N(0,1)")
    ax_toy.set_xlabel(r"Kernel bandwidth  $\sigma$", fontsize=10)
    ax_toy.set_ylabel(r"MMD", fontsize=10)
    ax_toy.set_title("Toy Example: MMD Between Two 1-D Gaussians", fontsize=12)
    ax_toy.legend(frameon=True, fontsize=8)
    ax_toy.grid(linestyle="--", alpha=0.5)
    fig_toy.tight_layout()
    fig_toy.savefig("mmd_toy_gaussian.png", dpi=300)
    print("Saved toy Gaussian MMD plot as mmd_toy_gaussian.png")

    # ----------------------------------------------------------------------------
    # 2) Prepare data loader and baseline model
    # ----------------------------------------------------------------------------
    model0, cfg = load_model_and_cfg(args.checkpoints[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = load_data_by_well(cfg, split=args.split)

    print("Baseline model:", args.checkpoints[0])
    model0.to(device)
    orig_full, recon_full0 = gather_images(model0, loader, device)
    N_full, D = orig_full.shape
    print(f"Total images gathered: {N_full}")

    # Subsample for memory safety
    subs = np.random.choice(N_full, SUBSET_SIZE, replace=False)
    orig_imgs   = orig_full[subs]
    recon_imgs0 = recon_full0[subs]
    del orig_full, recon_full0
    N = SUBSET_SIZE

    # Median‐heuristic σ
    D_hh = pdist2(orig_imgs, orig_imgs)
    median_dist = np.median(D_hh[np.triu_indices(SUBSET_SIZE, k=1)])
    sigma0 = np.sqrt(median_dist)
    sigmas = [0.5 * sigma0, sigma0, 1.5 * sigma0, 2.0 * sigma0]
    print("Using σ values: ", [f"{s:.4g}" for s in sigmas])

    # ----------------------------------------------------------------------------
    # 3) Baseline summary (mean ± std) – recon vs orig / recon vs recon / recon vs Gaussian
    # ----------------------------------------------------------------------------
    base_recs = []
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
        base_recs.append([s, m_ro, sd_ro, m_rr, sd_rr, m_rg, sd_rg])

    base_df = pd.DataFrame(
        base_recs,
        columns=["sigma","ro_mean","ro_std","rr_mean","rr_std","rg_mean","rg_std"]
    ).set_index("sigma")
    base_df.index = base_df.index.map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))

    base_tab = (
        base_df.assign(
            recon_orig  = base_df["ro_mean"].round(4).astype(str) + " \pm " + base_df["ro_std"].round(4).astype(str),
            recon_recon = base_df["rr_mean"].round(4).astype(str) + " \pm " + base_df["rr_std"].round(4).astype(str),
            recon_gauss = base_df["rg_mean"].round(4).astype(str) + " \pm " + base_df["rg_std"].round(4).astype(str)
        )
        .loc[:, ["recon_orig", "recon_recon", "recon_gauss"]]
    )

    print("\n### Baseline Table (LaTeX)\n")
    print(base_tab.to_latex(index=True, escape=False, column_format="c c c c"))

    # ----------------------------------------------------------------------------
    # 4) Multi‐model MMD: collect replicate values
    # ----------------------------------------------------------------------------
    records = []
    for ckpt in args.checkpoints:
        print("Processing model:", ckpt)
        model, _ = load_model_and_cfg(ckpt)
        model.to(device)

        _, recon_full = gather_images(model, loader, device)
        recon_sub     = recon_full[subs]
        del recon_full

        short_name = os.path.splitext(os.path.basename(ckpt))[0]
        for s in sigmas:
            for rep_id in range(args.reps):
                ir = np.random.randint(0, N, size=MAX_IMAGES)
                io = np.random.randint(0, N, size=MAX_IMAGES)
                R1 = recon_sub[ir]
                O  = orig_imgs[io]
                mmd_val = compute_mmd_multi(R1, O, [s])
                records.append({
                    "model": short_name,
                    "sigma": s,
                    "replicate": rep_id,
                    "mmd": mmd_val
                })

    df = pd.DataFrame(records)
    df["sigma_str"] = df["sigma"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))

    # ----------------------------------------------------------------------------
    # 5) Compute 95% CI and plot figures
    # ----------------------------------------------------------------------------
    summary = (
        df.groupby(["model", "sigma_str"])["mmd"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["tcrit"] = summary["count"].map(lambda n: t.ppf(0.975, df=n-1))
    summary["ci_lower"] = summary["mean"] - summary["tcrit"] * summary["sem"]
    summary["ci_upper"] = summary["mean"] + summary["tcrit"] * summary["sem"]

    # --- FIGURE: All Models CI ---
    fig_all, ax_all = plt.subplots(figsize=(6, 4))
    for model_name in summary["model"].unique():
        sub = summary[summary["model"] == model_name].sort_values("sigma_str")
        x_vals = sub["sigma_str"].astype(float).values
        y_means = sub["mean"].values
        ci_half = (sub["ci_upper"].values - sub["ci_lower"].values) / 2

        ax_all.errorbar(
            x_vals,
            y_means,
            yerr=ci_half,
            fmt='o-',
            capsize=5,
            label=model_name.capitalize()
        )
    ax_all.set_xlabel(r"$\sigma$", fontsize=12)
    ax_all.set_ylabel(r"MMD (Reconstruction vs Original)", fontsize=12)
    ax_all.set_title("MMD with 95% Confidence Intervals", fontsize=14)
    ax_all.legend(frameon=True, fontsize=10)
    ax_all.grid(True, linestyle="--", alpha=0.5)
    ax_all.set_xticks(sorted(summary["sigma_str"].astype(float).unique()))
    ax_all.set_xticklabels([f"{val:.2f}" for val in sorted(summary["sigma_str"].astype(float).unique())])
    fig_all.tight_layout()
    fig_all.savefig("mmd_all_models_ci.png", dpi=300)
    print("Saved combined CI plot as mmd_all_models_ci.png")

    # --- FIGURE: Baseline ---
    fig_base, ax_base = plt.subplots(figsize=(6, 4))
    x = base_df.index.astype(float).to_numpy()
    y_ro, e_ro = base_df["ro_mean"].to_numpy(), 2 * base_df["ro_std"].to_numpy()
    y_rr, e_rr = base_df["rr_mean"].to_numpy(), 2 * base_df["rr_std"].to_numpy()
    y_rg, e_rg = base_df["rg_mean"].to_numpy(), 2 * base_df["rg_std"].to_numpy()

    ax_base.errorbar(x, y_ro, yerr=e_ro, fmt='.-', capsize=4, label="Recon vs Orig")
    ax_base.errorbar(x, y_rr, yerr=e_rr, fmt='.-', capsize=4, label="Recon vs Recon")
    ax_base.errorbar(x, y_rg, yerr=e_rg, fmt='.-', capsize=4, label="Recon vs Gaussian")
    ax_base.set_xlabel(r"$\sigma$", fontsize=12)
    ax_base.set_ylabel(r"MMD", fontsize=12)
    short_basename = os.path.basename(args.checkpoints[0])
    ax_base.set_title(f"Baseline (model={short_basename.replace('.pth','').capitalize()})", fontsize=14)
    ax_base.set_ylim(0, None)
    ax_base.legend(frameon=True, fontsize=10)
    ax_base.grid(True, linestyle="--", alpha=0.5)
    fig_base.tight_layout()
    fig_base.savefig("mmd_baseline.png", dpi=300)
    print("Saved baseline plot as mmd_baseline.png")

    # --- FIGURE: Four‐panel boxplots ---
    unique_sigmas = sorted(df["sigma"].unique())
    n_sig = len(unique_sigmas)
    fig_boxes, axes_boxes = plt.subplots(
        nrows=1,
        ncols=n_sig,
        figsize=(4 * n_sig, 4),
        sharey=True
    )
    if n_sig == 1:
        axes_boxes = [axes_boxes]

    for ax, s in zip(axes_boxes, unique_sigmas):
        subdf = df[df["sigma"] == s]
        models_in_order = sorted(subdf["model"].unique())
        data_for_box = [subdf[subdf["model"] == m]["mmd"].values for m in models_in_order]

        ax.boxplot(
            data_for_box,
            labels=[m.capitalize() for m in models_in_order],
            patch_artist=True,
            boxprops=dict(facecolor="lightgray", edgecolor="black"),
            medianprops=dict(color="firebrick"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker='o', markerfacecolor='black', markersize=4, alpha=0.6),
            widths=0.6,
            whis=1.5,
            showfliers=True
        )
        ax.set_title(r"$\sigma$ = " + f"{s:.2f}", fontsize=12)
        ax.set_xlabel("Model", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", labelrotation=30)
        if ax == axes_boxes[0]:
            ax.set_ylabel(r"MMD (Reconstruction vs Original)", fontsize=12)

    fig_boxes.suptitle("Multi‐Model MMD Comparison (Boxplots over Replicates)", fontsize=14)
    fig_boxes.tight_layout(rect=[0, 0, 1, 0.93])
    fig_boxes.savefig("mmd_models_boxplot_4panel.png", dpi=300)
    print("Saved 4-panel boxplot figure as mmd_models_boxplot_4panel.png")

    # ----------------------------------------------------------------------------
    # 6) Hypothesis Tests and LaTeX Tables (no Shapiro; clear decision)
    # ----------------------------------------------------------------------------

    # 6a) One‐sample Wilcoxon tests: Is MMD significantly > 0?
    one_sample_results = []
    for m_name in sorted(df["model"].unique()):
        for sigma_str in sorted(df["sigma_str"].unique(), key=lambda x: float(x)):
            vals = df[(df["model"] == m_name) & (df["sigma_str"] == sigma_str)]["mmd"].values
            if len(vals) < 2:
                continue

            # Perform one‐sided Wilcoxon signed‐rank test vs 0
            try:
                w_stat, p_wilcox_twosided = wilcoxon(vals - 0.0, zero_method="wilcox", alternative="two-sided")
                if np.median(vals) > 0:
                    p_one_sided = p_wilcox_twosided / 2
                else:
                    p_one_sided = 1.0
            except ValueError:
                # If all values are identical, wilcoxon may fail; set p=1.0
                p_one_sided = 1.0

            decision = "Reject H₀" if p_one_sided < 0.05 else "Fail to Reject H₀"
            one_sample_results.append({
                "Model": m_name.capitalize(),
                "Sigma": sigma_str,
                "Test": "Wilcoxon",
                "p-value": p_one_sided,
                "Decision": decision
            })

    one_sample_df = pd.DataFrame(one_sample_results)
    one_sample_df = one_sample_df.sort_values(
        ["Model", "Sigma"],
        key=lambda col: col.map(lambda x: float(x) if col.name == "Sigma" else x)
    )

    # Format p-values (bold if significant)
    one_sample_df["p-value"] = one_sample_df["p-value"].map(
        lambda p: ("\textbf{%.4f}" % p) if p < 0.05 else ("%.4f" % p)
    )

    one_sample_table = one_sample_df.loc[:, ["Model", "Sigma", "Test", "p-value", "Decision"]]
    one_sample_table = one_sample_table.rename(
        columns={
            "Model": "Model",
            "Sigma": "$\sigma$",
            "Test": "Test",
            "p-value": "$p$",
            "Decision": "Decision"
        }
    )

    print("\n### One‐Sample Wilcoxon Tests (LaTeX)\n")
    print(one_sample_table.to_latex(index=False, escape=False, column_format="l c l c l"))

    # 6b) Two‐sample paired Wilcoxon tests: Compare model pairs at each σ
    two_sample_results = []
    for sigma_str in sorted(df["sigma_str"].unique(), key=lambda x: float(x)):
        df_s = df[df["sigma_str"] == sigma_str]
        for (m1, m2) in itertools.combinations(sorted(df["model"].unique()), 2):
            # Extract replicates for each model, align by replicate index
            vals1 = df_s[df_s["model"] == m1].sort_values("replicate")["mmd"].values
            vals2 = df_s[df_s["model"] == m2].sort_values("replicate")["mmd"].values
            if len(vals1) < 2 or len(vals2) < 2:
                continue

            diffs = vals1 - vals2
            try:
                w_stat, p_wilcox_twosided = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                if np.median(diffs) > 0:
                    p_one_sided = p_wilcox_twosided / 2
                else:
                    p_one_sided = 1.0
            except ValueError:
                p_one_sided = 1.0

            decision = "Reject H₀" if p_one_sided < 0.05 else "Fail to Reject H₀"
            two_sample_results.append({
                "Sigma": sigma_str,
                "Model A": m1.capitalize(),
                "Model B": m2.capitalize(),
                "Test": "Wilcoxon",
                "p-value": p_one_sided,
                "Decision": decision
            })

    two_sample_df = pd.DataFrame(two_sample_results)
    two_sample_df = two_sample_df.sort_values(
        ["Sigma", "Model A", "Model B"],
        key=lambda col: col.map(lambda x: float(x) if col.name == "Sigma" else x)
    )

    two_sample_df["p-value"] = two_sample_df["p-value"].map(
        lambda p: ("\textbf{%.4f}" % p) if p < 0.05 else ("%.4f" % p)
    )

    two_sample_table = two_sample_df.loc[:, [
        "Sigma", "Model A", "Model B", "Test", "p-value", "Decision"
    ]]
    two_sample_table = two_sample_table.rename(
        columns={
            "Sigma": "$\sigma$",
            "Model A": "Model A",
            "Model B": "Model B",
            "Test": "Test",
            "p-value": "$p$",
            "Decision": "Decision"
        }
    )

    print("\n### Two‐Sample Paired Wilcoxon Tests (LaTeX)\n")
    print(two_sample_table.to_latex(index=False, escape=False, column_format="c l l l c l"))

    # ----------------------------------------------------------------------------
    # 7) Confidence Intervals at σ = 15.35
    # ----------------------------------------------------------------------------
    target_sigma = "15.64"
    ci_subset = summary[summary["sigma_str"] == target_sigma].copy()
    m_models = ci_subset.shape[0]
    alpha_bonf = 0.05 / 1  # Bonferroni correction for multiple comparisons
     # Recompute tcrit with Bonferroni correction per model
    tcrit_bonf = []
    for _, row in ci_subset.iterrows():
        dfree = int(row["count"] - 1)
        tcrit_bonf.append(t.ppf(1 - alpha_bonf/2, df=dfree))
    ci_subset["tcrit_bonf"] = tcrit_bonf

    # Compute Bonferroni‐corrected intervals
    ci_subset["CI Lower"] = ci_subset["mean"] - ci_subset["tcrit_bonf"] * ci_subset["sem"]
    ci_subset["CI Upper"] = ci_subset["mean"] + ci_subset["tcrit_bonf"] * ci_subset["sem"]

    ci_subset = ci_subset.rename(columns={
        "model": "Model",
        "mean": "Mean",
        "std": "StdDev"
    })
    ci_subset["Model"] = ci_subset["Model"].str.capitalize()
    ci_table = ci_subset.loc[:, ["Model", "Mean", "StdDev", "CI Lower", "CI Upper"]]

    print(f"\n### Bonferroni‐Corrected 95\% Confidence Intervals at $\sigma$ = {target_sigma} (LaTeX)\n")
    print(ci_table.to_latex(index=False, float_format="%.4f", escape=False,
        column_format="l c c c c"))

if __name__ == "__main__":
    main()
