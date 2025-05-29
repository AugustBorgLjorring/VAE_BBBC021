import matplotlib.pyplot as plt
import torch
import numpy as np

def run_gradient_sensitivity(model, loader, viz, args):
    """
    Local gradient-based sensitivity:
    Computes avg |d ||x||^2 / dz_k | over random z ~ N(0,I).
    """
    print(">> Computing gradient sensitivity")
    n_samples = args.sens_n
    batch_size = 10

    device = next(model.parameters()).device
    D = model.fc_mu.out_features
    sens = torch.zeros(D, device=device)
    model.decoder.eval()

    for _ in range(n_samples // batch_size):
        mu = torch.zeros(batch_size, D, device=device, requires_grad=True)
        logvar = torch.zeros_like(mu)
        z = model.reparameterize(mu, logvar)  # [S, B, D]
        z_flat = z.view(-1, D)
        recon = model.decode(z_flat)               # [S*B, C, H, W]
        scores = recon.view(batch_size, -1).pow(2).sum(dim=1)  # [B]
        grads = torch.autograd.grad(scores.sum(), z)[0]        # [B, D]
        sens += grads.squeeze(0).abs().sum(dim=0)
    sens = sens / n_samples
    sens_cpu = sens.cpu().numpy()

    # Sort descending
    sorted_idx = np.argsort(sens_cpu)[::-1]
    sorted_sens = sens_cpu[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(D), sorted_sens, width=0.8)
    ax.set_xticks(range(D))
    ax.set_xticklabels(sorted_idx, rotation=90, fontsize=6)
    ax.set_xlabel("Latent dim (sorted by sensitivity)")
    ax.set_ylabel(r"$Avg |\delta \| x \|^2 / \delta z_k|$")
    ax.set_title(f"Local gradient sensitivity (n_samples={n_samples})")
    viz.save(fig, f"grad_sensitivity_sorted_{n_samples}")

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
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            mu_batch, _ = model.encode(x_batch)      # [B, D]
            mus_batches.append(mu_batch.cpu())       # keep on CPU for now
            collected += mu_batch.size(0)
            if collected >= n_samples:
                break

    all_mus = torch.cat(mus_batches, dim=0)         # [>=n_samples, D]
    mus =  all_mus[:n_samples].to(device)           # trim & move back to GPU

    # For each encoded latent, perturb each dim +-eps and accumulate L2 diffs
    for mu in mus:
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