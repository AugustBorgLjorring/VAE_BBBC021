import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from vae_model import VAE, BetaVAE, VAEPlus
from src.data_loading import load_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

class Visualizer:
    def __init__(self, checkpoint_path: str):
        self.ckpt = checkpoint_path
        ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        self.out_dir = os.path.join("experiments", "results", ckpt_name)
        os.makedirs(self.out_dir, exist_ok=True)

    def save_img(self, img: torch.Tensor, name: str):
        """Save a (C,H,W) or (B,C,H,W) tensor as PNG (grid if batch)."""
        out = os.path.join(self.out_dir, f"{name}.png")
        if img.ndim == 4:
            grid = make_grid(img, nrow=img.size(0)//2 or 1, pad_value=1)
            save_image(grid, out, normalize=True)
        else:
            save_image(img, out, normalize=True)
        print(f"  • Saved {out}")

    def save_gif(self, frames: torch.Tensor, name: str, interval=200):
        """frames: (T, C, H, W) tensor."""
        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames[0].permute(1,2,0).cpu().numpy())
        def update(i):
            im.set_array(frames[i].permute(1,2,0).cpu().numpy())
            return [im]
        anim = FuncAnimation(fig, update, frames=frames.size(0), interval=interval, blit=True)
        out = os.path.join(self.out_dir, f"{name}.gif")
        anim.save(out, writer="pillow", fps=1000//interval)
        plt.close(fig)
        print(f"  • Saved {out}")

def load_model_and_cfg(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    cfg = OmegaConf.create(checkpoint['config'])  # Load config from checkpoint
    # Override bbbc021 train_path in the config if needed
    cfg.data.train_path = "C:/BBBC021/BBBC021_cleaned_preprocessed.h5"

    # Load the model architecture from the checkpoint
    if cfg['model']['name'] == "VAE":
        model = VAE(
            input_channels=cfg.model.input_channels,
            latent_dim=cfg.model.latent_dim
        )
    elif cfg['model']['name'] == "Beta_VAE":
        model = BetaVAE(
            input_channels=cfg.model.input_channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, cfg

def do_reconstruct(model, data_loader, viz, num_images):
    x, _ = next(iter(data_loader))
    x = x[:num_images]
    with torch.no_grad():
        recon, _, _ = model(x)
    # stack originals on top of reconstructions
    pairs = torch.cat([x, recon], dim=0)
    viz.save_img(pairs, f"reconstructions_{num_images}")

def do_interpolate(model, data_loader, viz, idx1, idx2, steps):
    x, _ = next(iter(data_loader))
    img1, img2 = x[idx1:idx1+1], x[idx2:idx2+1]
    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        base = model.decode(mu1)
        # rank dims by impact
        diffs = []
        for d in range(mu1.size(1)):
            temp = mu1.clone()
            temp[0,d] = mu2[0,d]
            diffs.append((d, ((model.decode(temp)-base)**2).mean().item()))
        order = [d for d,_ in sorted(diffs, key=lambda x: x[1], reverse=True)]
        # generate frames
        latent = mu1.clone()
        frames = []
        for d in order:
            for a in torch.linspace(latent[0,d], mu2[0,d], steps):
                latent[0,d] = a
                frames.append(model.decode(latent)[0])
        frames = torch.stack(frames, dim=0)  # (T,C,H,W)
    viz.save_gif(frames, f"interpolate_{idx1}_to_{idx2}")

def do_perturb(model, data_loader, viz, idx, n_rows, n_cols, sigma):
    x, _ = next(iter(data_loader))
    img = x[idx:idx+1]
    with torch.no_grad():
        mu, _ = model.encode(img)
        rand = torch.randn_like(mu)
        rand /= rand.norm()
        samples = [model.decode(mu + sigma*rand) for _ in range(n_rows*n_cols)]
    grid = torch.cat(samples, dim=0)
    viz.save_img(grid, f"perturb_{idx}")

def do_random(model, viz):
    with torch.no_grad():
        latent_dim = model.fc_mu.out_features
        latent = torch.randn(1, latent_dim)
        img = model.decode(latent)
    viz.save_img(img, "random_sample")

def do_tsne(model, data_loader, viz, num_samples):
    from sklearn.manifold import TSNE

    imgs = []
    for x_batch, _ in data_loader:
        imgs.append(x_batch)  
        if sum(b.size(0) for b in imgs) >= num_samples:
            break
    xs = torch.cat(imgs, dim=0)[:num_samples]
    print(f"Collected {xs.size(0)} images for t-SNE")

    with torch.no_grad():
        latents = torch.cat([model.encode(img.unsqueeze(0))[0] for img in xs], dim=0).cpu().numpy()
    coords = TSNE(n_components=2, random_state=0).fit_transform(latents)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis("off")
    ax.scatter(coords[:,0], coords[:,1], alpha=0)

    for i, (x0,y0) in enumerate(coords):
        im = OffsetImage(xs[i].permute(1,2,0).cpu().numpy(), zoom=0.1)
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
    out = os.path.join(viz.out_dir, f"tsne_{num_samples}.png")
    fig.savefig(out, bbox_inches="tight", dpi=1200)
    plt.close(fig)
    print(f"  • Saved {out}")

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained VAE checkpoint.")
    p.add_argument("--checkpoint", "-c", required=True, help="Path to .pth checkpoint")
    p.add_argument("--tasks", "-t", nargs="+",
                   choices=["recon","interp","perturb","random","tsne","all"],
                   default=["all"], help="Which tasks to run")
    p.add_argument("--num_images",   type=int, default=5,    help="for recon")
    p.add_argument("--interp_idx",   nargs=2,  type=int, default=[0,1], help="pair for interp")
    p.add_argument("--interp_steps", type=int, default=10,   help="steps per dim")
    p.add_argument("--perturb_idx",  type=int, default=0,    help="which image to perturb")
    p.add_argument("--perturb_rows", type=int, default=2,    help="grid rows")
    p.add_argument("--perturb_cols", type=int, default=10,   help="grid cols")
    p.add_argument("--perturb_sigma",type=float,default=0.1, help="noise scale")
    p.add_argument("--tsne_samples", type=int, default=100,   help="how many for t-SNE")
    args = p.parse_args()

    print("Loading model + config…")
    model, cfg = load_model_and_cfg(args.checkpoint)
    data_loader = load_data(cfg, split="test")
    viz = Visualizer(args.checkpoint)

    tasks = args.tasks
    if "all" in tasks or "recon" in tasks:
        print("reconstructing…")
        do_reconstruct(model, data_loader, viz, args.num_images)
    if "all" in tasks or "interp" in tasks:
        print("interpolating…")
        i1, i2 = args.interp_idx
        do_interpolate(model, data_loader, viz, i1, i2, args.interp_steps)
    if "all" in tasks or "perturb" in tasks:
        print("perturbing…")
        do_perturb(model, data_loader, viz,
                   args.perturb_idx,
                   args.perturb_rows,
                   args.perturb_cols,
                   args.perturb_sigma)
    if "all" in tasks or "random" in tasks:
        print("sampling random image…")
        do_random(model, viz)
    if "all" in tasks or "tsne" in tasks:
        print("computing t-SNE…")
        do_tsne(model, data_loader, viz, args.tsne_samples)

if __name__ == "__main__":
    main()
