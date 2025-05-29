import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec


def run_roundtrip(model, loader, viz, args):
    print(">> Round-trip consistency")

    idx = args.image_index

    x, _ = next(iter(loader))
    img = x[idx:idx+1]

    with torch.no_grad():
        mu0, _ = model.encode(img)
        rec0 = model.decode(mu0)
        mu1, _ = model.encode(rec0)
    delta = (mu1 - mu0).squeeze(0).cpu().numpy()
    D = delta.shape[0]

    fig = plt.figure(figsize=(8, 6), dpi=150)
    gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img.squeeze(0).permute(1, 2, 0).cpu(), vmin=0, vmax=1)
    ax0.set_title(f"Original (#{idx})", fontsize=10)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rec0.squeeze(0).permute(1, 2, 0).cpu(), vmin=0, vmax=1)
    ax1.set_title(r"Reconstructed from $\mu_0$", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.bar(range(D), delta, width=0.8)
    ax2.set_xlabel("Latent dim", fontsize=9)
    ax2.set_ylabel(r"$\Delta z = \mu_1-\mu_0$", fontsize=9)
    ax2.set_title("Latent change after one round-trip", fontsize=10)
    ax2.tick_params(axis="x", labelsize=5, rotation=90)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"VAE Round-Trip idx={idx}", fontsize=12, y=1.05)
    viz.save(fig, f"roundtrip_{idx}")


def run_plot_latent(model, loader, viz, args):
    print(">> Plotting latent stats")

    idx = args.image_index
    x, _ = next(iter(loader))
    img = x[idx:idx+1]

    with torch.no_grad():
        mu, logvar = model.encode(img)
        
    mu_np = mu.squeeze(0).cpu().numpy()
    sigma_np = torch.exp(0.5 * logvar).squeeze(0).cpu().numpy()
    D = mu_np.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    dims = list(range(D))

    ax1.bar(dims, mu_np, width=0.8)
    ax1.set_ylabel(r"$\mu$", fontsize=9)
    ax1.set_title(f"Latent means idx={idx}", fontsize=10)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.bar(dims, sigma_np, width=0.8)
    ax2.set_xlabel("Latent dim", fontsize=9)
    ax2.set_ylabel(r"$\sigma$", fontsize=9)
    ax2.set_title(f"Latent stddevs idx={idx}", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=90, fontsize=6)

    fig.suptitle(f"Latent statistics (idx={idx})", fontsize=12, y=1.05)
    viz.save(fig, f"latent_stats_{idx}")


def run_latent_usage(model, loader, viz, args):
    print(">> Computing overall latent usage")
    device = next(model.parameters()).device

    D = model.fc_mu.out_features
    sum_mu = torch.zeros(D)
    sum_sigma = torch.zeros(D)

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            mu, lv = model.encode(x_batch)
            sum_mu += mu.abs().sum(0).cpu()
            sum_sigma += torch.exp(0.5 * lv).sum(0).cpu()

    mu_np = sum_mu.numpy()
    sigma_np = sum_sigma.numpy()
    dims = list(range(D))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    ax1.bar(dims, mu_np, width=0.8)
    ax1.set_ylabel(r"$\Sigma|\mu|$", fontsize=9)
    ax1.set_title("Sum of |latent means|", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.bar(dims, sigma_np, width=0.8)
    ax2.set_xlabel("Latent dim", fontsize=9)
    ax2.set_ylabel(r"$\Sigma\sigma$", fontsize=9)
    ax2.set_title(r"Sum of latent $\sigma$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=90, fontsize=6)

    fig.suptitle("Overall latent-usage statistics", fontsize=12, y=1.05)
    viz.save(fig, "latent_usage")


def run_top_latent_dims(model, loader, viz, args, top_k=10):
    print(">> Plotting top latent dimensions")
    # Accumulate |mu| and sigma across the test set
    device = next(model.parameters()).device
    with torch.no_grad():
        # figure out latent size
        dummy, _ = next(iter(loader))
        D = model.encode(dummy[:1].to(device))[0].size(1)

        sum_abs_mu    = torch.zeros(D, device='cpu')
        sum_sigma     = torch.zeros(D, device='cpu')

        for x, _ in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            sigma = torch.exp(0.5 * logvar)

            sum_abs_mu += mu.abs().sum(dim=0).cpu()
            sum_sigma  += sigma.sum(dim=0).cpu()

    # Compute a single usage score and pick top_k
    usage = sum_abs_mu
    top_dims = torch.argsort(usage, descending=True)[:top_k]

    fig, ax = plt.subplots(figsize=(top_k*0.5+1, 4))
    ax.bar([f"{d}" for d in top_dims.tolist()], usage[top_dims].numpy(), width=0.6)
    ax.set_xlabel("Latent dimension index")
    ax.set_ylabel(r"Sum $|\mu|$")
    ax.set_title(f"Top {top_k} most-used latent dimensions", fontsize=12, y=1.02)
    viz.save(fig, f"top{top_k}_latent_usage")