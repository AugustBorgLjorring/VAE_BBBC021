import matplotlib.pyplot as plt
import torch
import numpy as np

def run_reconstruct(model, loader, viz, args):
    print(">> Reconstructing")
    N = args.num_images

    x, _ = next(iter(loader))
    org_imgs = x[:N]

    with torch.no_grad():
        mus, logvars = model.encode(org_imgs)
        recon_imgs = model.decode(mus)

    fig, axes = plt.subplots(2, N, figsize=(8, 4))
    for i in range(N):
        ax0 = axes[0, i]
        ax0.imshow(org_imgs[i].permute(1,2,0), vmin=0, vmax=1)
        ax0.set_xticks([]); ax0.set_yticks([])

        # Recon (B=1, C, H, W), Original (B=1, C, H, W) to (C, H, W), and (B=1, D)
        _, recon_loss, kld_loss, _ = model.loss_function(recon_imgs[i:i+1], org_imgs[i:i+1], mus[i:i+1], logvars[i:i+1])
        recon_loss = recon_loss.item()
        kld_loss = kld_loss.item()

        ax1 = axes[1, i]
        ax1.imshow(recon_imgs[i].permute(1,2,0), vmin=0, vmax=1)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.text(0.05, 0.95, f"RECON={recon_loss:.1f}\nKL={kld_loss:.1f}", transform=ax1.transAxes, 
                 fontsize=8, va='top', color='white', bbox=dict(facecolor='black', alpha=0.3, pad=2))
        ax1.set_xlabel(f"Image {i+1}", fontsize=10, labelpad=6)
        
    axes[0, 0].set_ylabel("Original", rotation=90, fontsize=13, labelpad=6)
    axes[1, 0].set_ylabel("Reconstructed", rotation=90, fontsize=13, labelpad=6)
    fig.suptitle(f"Reconstruction of {N} images", fontsize=20, y=0.93)
    plt.tight_layout(pad=0.5)  # tighter layout
    viz.save(fig, f"reconstructions_{N}")

def run_reconstruct_split_channels(model, loader, viz, args):
    print(">> Channel-split reconstruction")
    idx = args.image_index

    # get one image
    x, _ = next(iter(loader))
    org_img = x[idx:idx+1] # (1, C, H, W)

    # reconstruct
    with torch.no_grad():
        mu, _ = model.encode(org_img)
        recon_img = model.decode(mu)

    # convert from (1, C, H, W) to (H, W, C) numpy
    orig_np  = org_img[0].permute(1,2,0).numpy()
    recon_np = recon_img[0].permute(1,2,0).numpy()

    # 2 rows x 4 columns: RGB, R, G, B
    fig, axes = plt.subplots(2, 4, figsize=(7, 4), constrained_layout=True)

    # row 0: original
    axes[0,0].imshow(orig_np)
    axes[0,0].set_title("Original")
    axes[0,0].axis("off")

    for c, name, color in zip([0,1,2], ["Red","Green","Blue"], ["red","green","blue"]):
        ch = orig_np[:,:,c]
        canvas = np.zeros_like(orig_np)
        canvas[:,:,c] = ch
        ax = axes[0, c+1]
        ax.imshow(canvas)
        ax.set_title(name, color=color, fontsize=12)
        ax.axis("off")

    # row 1: reconstruction
    axes[1,0].imshow(recon_np)
    axes[1,0].set_title("Reconstructed")
    axes[1,0].axis("off")

    for c, name, color in zip([0,1,2], ["Red","Green","Blue"], ["red","green","blue"]):
        ch = recon_np[:,:,c]
        canvas = np.zeros_like(recon_np)
        canvas[:,:,c] = ch
        ax = axes[1, c+1]
        ax.imshow(canvas)
        ax.set_title(name, color=color, fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    fig.suptitle(f"Channel-split reconstruction (Image {idx+1})", fontsize=16, y=1.05)
    viz.save(fig, f"reconstruct_split_channels_{idx}")