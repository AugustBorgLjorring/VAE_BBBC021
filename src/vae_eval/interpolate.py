import os
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

# Visualize sequential interpolation in latent space of a VAE model
# This function generates a GIF showing how the model interpolates 
# between two images by varying each latent dimension sequentially.
def run_interpolate_seq(model, loader, viz, i1, i2, steps):
    print(">> Sequential interpolation")
    x, _ = next(iter(loader))
    img1, img2 = x[i1:i1+1], x[i2:i2+1]

    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        base = model.decode(mu1)
        diffs = []

        # Calculate the difference in reconstruction for each latent dimension
        for d in range(mu1.size(1)):
            tmp = mu1.clone()
            tmp[0,d] = mu2[0,d]
            diffs.append((d, ((model.decode(tmp)-base)**2).mean().item()))
        order = [d for d,_ in sorted(diffs, key=lambda x: x[1], reverse=True)]

        frames = []
        for d in order:
            for a in torch.linspace(0,1,steps):
                z = mu1.clone()
                z[0,d] = (1 - a) * mu1[0,d] + a * mu2[0,d] # linear interpolation from mu1 to mu2 in dim d
                frames.append(model.decode(z)[0])
        frames = torch.stack(frames,0)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.axis("off")
    txt = ax.text(0.02,0.98,"",transform=ax.transAxes, ha="left",va="top", fontsize=10, color="white", bbox=dict(facecolor='black',alpha=0.6,pad=3))
    im = ax.imshow(frames[0].permute(1,2,0).cpu(), vmin=0, vmax=1)
    per_dim = len(frames)//len(order)

    # Update function for animation
    def upd(i):
        im.set_array(frames[i].permute(1,2,0).cpu())
        txt.set_text(f"Dim -> {order[i//per_dim]}")
        return im, txt

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=200, blit=True)
    out = os.path.join(viz.out_dir, f"interp_seq_{i1}_to_{i2}.gif")
    anim.save(out, writer="pillow", fps=5)
    plt.close(fig)
    print(f"Saved {out}")

# Visualize linear interpolation between two images in latent space of a VAE model
# This function generates a grid of images showing the linear interpolation
# between two selected images by varying the latent representation.
def run_interpolate_lin(model, loader, viz, args, steps = 20):
    print(">> Linear interpolation")
    i1, i2 = args.interp_idx

    x, _ = next(iter(loader))
    img1, img2 = x[i1:i1+1], x[i2:i2+1]

    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        a = torch.linspace(0, 1, steps, device=mu1.device).unsqueeze(1)
        latent = (1 - a) * mu1 + a * mu2  # Linear interpolation in latent space
        frames = model.decode(latent)
        alphas = a.squeeze(1)
        
    cols = steps // 2 if steps % 2 == 0 else steps
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4), constrained_layout=True)
    flat_axes = axes.flatten()
    
    # Set the first and last images to be the original images
    for idx, frame in enumerate(frames):
        ax = flat_axes[idx]
        ax.imshow(frame.permute(1,2,0).cpu(), vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(1)
        ax.set_xlabel(rf"$\alpha={alphas[idx]:.3f}$", fontsize=10, labelpad=4)
    
    # Set the first and last images with special colors
    for special, color in [(0, 'green'), (steps - 1, 'red')]:
        ax = flat_axes[special]
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)
        img_idx = i1 if special == 0 else i2
        alpha = alphas[special]
        ax.set_xlabel(rf"Image {img_idx} | $\alpha={alpha:.3f}$", fontsize=10, labelpad=4)
        
    fig.suptitle(f"Linear interpolation: Image {i1} -> {i2}", fontsize=12, y=1.05)
    viz.save(fig, f"interp_lin_{i1}_to_{i2}")