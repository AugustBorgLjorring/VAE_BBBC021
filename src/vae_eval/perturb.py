import matplotlib.pyplot as plt
import torch


def run_perturb(model, loader, viz, args):
    print(">> Latent perturbations")
    x, _ = next(iter(loader))
    num_images = args.num_images
    idx = args.image_index
    epsilon_max = args.perturb_eps
    imgs = x[idx:idx+num_images]
    num_perturbations = 9
    rows = num_images
    cols = num_perturbations + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2),
                             sharex='col', sharey='row', constrained_layout=True)
    with torch.no_grad():
        mus, _ = model.encode(imgs)
    base_dir = torch.ones_like(mus[0:1])
    base_dir = base_dir / base_dir.norm(dim=1, keepdim=True)
    
    for row in range(rows):
        mu = mus[row:row+1]
        orig_arr = model.decode(mu)[0].permute(1,2,0).cpu().detach().numpy()
        ax = axes[row, 0]
        ax.imshow(orig_arr, vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"Image {row}", rotation=90, fontsize=14, labelpad=6)
        
        for col in range(1, cols):
            epsilon = epsilon_max * col / num_perturbations
            z = mu + epsilon * base_dir
            pert_arr = model.decode(z)[0].permute(1,2,0).cpu().detach().numpy()
            ax = axes[row, col]
            ax.imshow(pert_arr, vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
            mse = ((pert_arr - orig_arr) ** 2).mean()
            ax.text(0.05, 0.95, f"MSE={mse:.4f}", transform=ax.transAxes,
                    fontsize=12, va='top', color='white',
                    bbox=dict(facecolor='black', alpha=0.5, pad=2))
            
    for col in range(cols):
        ax = axes[-1, col]
        if col == 0:
            ax.set_xlabel("Original", fontsize=14, labelpad=4)
        else:
            epsilon = epsilon_max * col / num_perturbations
            ax.set_xlabel(rf"$\epsilon={epsilon:.2f}$", fontsize=14, labelpad=4)
            
    fig.suptitle(rf"Latent perturbations up to $\epsilon={epsilon_max}$", fontsize=18, y=1.04)
    viz.save(fig, f"perturb_{idx}_to_{idx+num_images-1}")