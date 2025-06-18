import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def run_gradient_sensitivity(model, loader, viz, args):
    """
    Computes global sensitivity per latent dimension using the mean squared
    Jacobian norm of decoder output w.r.t. each latent variable z_k.

    s_k = E_z[ || ∂hat{x} / ∂z_k ||^2 ]
    """
    print(">> Computing gradient sensitivity (mean squared Jacobian norm)")
    plot_small = True # set to True for a smaller plot size
    n_samples = args.sens_n
    batch_size = 100
    device = next(model.parameters()).device
    D = model.fc_mu.out_features  # number of latent dims

    sens = torch.zeros(D, device=device)
    model.decoder.eval()

    n_iter = n_samples // batch_size

    for _ in range(n_iter):
        z = torch.randn(batch_size, D, device=device, requires_grad=True)  # [B, D]
        recon = model.decode(z)                     # [B, C, H, W]
        recon_flat = recon.view(batch_size, -1)     # [B, C*H*W]
        
        # Compute ∇_z ||x̂||^2, shape [B, D]
        grads = torch.autograd.grad(recon_flat.sum(), z, create_graph=False, retain_graph=False)[0]

        # Mean squared gradient per latent dim across batch
        sens += grads.pow(2).mean(dim=0)

    # Average over all sampled z
    sens /= n_samples
    sens_root = torch.sqrt(sens)  # take square root for sensitivity
    sens_cpu = sens_root.cpu().numpy()  # shape: [D]

    # Sort descending
    sorted_idx = np.argsort(sens_cpu)[::-1]
    sorted_sens = sens_cpu[sorted_idx]

    D = len(sorted_sens)

    if plot_small:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig, ax = plt.subplots(figsize=(12, 4)) # 12, 4

    # --- Main bar plot ---
    # sizes = 32 = 8, 64 = 6, 128 = 4, 256 = 3
    if plot_small:
        fontsize_scaled = 5 if D <= 32 else (6 if D <= 64 else (3 if D <= 128 else 2))
    else:
        fontsize_scaled = 8 if D <= 32 else (8 if D <= 64 else (5 if D <= 128 else 3))
    ax.bar(range(D), sorted_sens, color='tab:blue', width=0.8)
    ax.set_xticks(range(D))
    ax.set_xticklabels(sorted_idx, rotation=90, fontsize=fontsize_scaled)
    if plot_small:
        ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(-0.5, D - 0.5)
    if plot_small:
        ax.set_ylabel(r"$\sqrt{\mathrm{Avg} \left( \left| \partial \hat{x} / \partial z_k \right|^2 \right)}$", labelpad=-5)
    else:
        ax.set_ylabel(r"$\sqrt{\mathrm{Avg} \left( \left| \partial \hat{x} / \partial z_k \right|^2 \right)}$")
    ax.set_xlabel("Latent dim (sorted by sensitivity)")
    ax.set_title(f"Gradient sensitivity")

    # --- Inset zoom ---
    if plot_small:
        bbox = (0.25, 0.27, 0.75, 0.75) 
    else:
        bbox = (0.20, 0.25, 0.75, 0.75)

    axins = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc='upper right',
        bbox_to_anchor=bbox,
        bbox_transform=ax.transAxes,
        borderpad=1
    )

    # show from 3 to last
    i_start, i_end = 3, D
    axins.bar(range(i_start, i_end), sorted_sens[i_start:i_end], color='tab:blue', width=0.8)
    axins.set_xticks(range(i_start, i_end))
    axins.set_xticklabels(sorted_idx[i_start:i_end], rotation=90)
    axins.set_xlim(i_start - 0.5, i_end - 0.5)
    axins.set_ylim(0, max(sorted_sens[i_start:i_end]) * 1.1)
    axins.tick_params(axis='y', labelsize=8)
    axins.tick_params(axis='x', labelsize=fontsize_scaled-1)

    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="black", linestyle="-")

    viz.save(fig, f"sensitivity_gradient_n{n_samples}")

def run_traversal_sensitivity(model, loader, viz, args):
    """
    Finite-difference (latent-traversal) sensitivity on encoded latents:
    Computes avg ||D(mu(x) + eps * e_k) - D(mu(x) - eps * e_k)||_2 over n_samples real test images.
    """
    print(">> Computing traversal sensitivity on encoded latents")
    n_samples = args.sens_n
    eps = args.sens_eps
    
    device = next(model.parameters()).device
    D = model.fc_mu.out_features
    sens = torch.zeros(D, device=device)
    model.decoder.eval()

    # Collect exactly n_samples encoded mu's from the test loader
    mus_batches = []
    collected = 0

    model.eval()
    with torch.no_grad():
        for x_batch, _ in tqdm(loader, desc="Collecting encoded latents", total=n_samples // loader.batch_size):
            x_batch = x_batch.to(device)
            mu_batch, _ = model.encode(x_batch)      # [B, D]
            mus_batches.append(mu_batch.cpu())       # keep on CPU for now
            collected += mu_batch.size(0)
            if collected >= n_samples:
                break

    all_mus = torch.cat(mus_batches, dim=0)         # [>=n_samples, D]
    mus =  all_mus[:n_samples].to(device)           # trim & move back to GPU

    # For each encoded latent, perturb each dim +-eps and accumulate L2 diffs
    for mu in tqdm(mus, desc="Computing traversal sensitivity", total=n_samples):
        z0 = mu.unsqueeze(0)                      # [1, D]
        for k in range(D):
            z_plus  = z0.clone(); z_plus[0, k]  += eps
            z_minus = z0.clone(); z_minus[0, k] -= eps
            x_p = model.decode(z_plus)
            x_m = model.decode(z_minus)
            diff = (x_p - x_m).view(1, -1).norm(dim=1)  # [1]
            sens[k] += diff.item()

    # Average & move to CPU numpy
    sens = sens / n_samples
    sens_cpu = sens.cpu().numpy()

    # Sort dims by descending sensitivity
    sorted_idx  = np.argsort(sens_cpu)[::-1]
    sorted_sens = sens_cpu[sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(D), sorted_sens, width=0.8)
    ax.set_xticks(range(D))
    ax.set_xticklabels(sorted_idx, rotation=90, fontsize=6)
    ax.set_xlabel(r"Latent dim (sorted by $\delta$-reconstruction)")
    ax.set_ylabel(r"Avg $\| D(\mu(x)+\epsilon e_k) - D(\mu(x) - \epsilon e_k) \|_2$")
    ax.set_title(rf"Traversal sensitivity on encoded latents ($\epsilon$={eps}, n={n_samples})")

    viz.save(fig, f"traversal_sensitivity_encoded_eps{eps}_n{n_samples}")