import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import pandas as pd

def run_pca_grid(model, loader, viz, args, grid_size=70):
    """
    PCA-grid of decoder outputs:
    1) Encode all test latents mu(x)
    2) Fit PCA -> 2D coords
    3) Define a uniform grid in PCA space
    4) Invert grid back to latent space via PCA.inverse_transform
    5) Decode and plot the grid of images
    """

    print(">> Computing PCA grid")
    device = next(model.parameters()).device

    # 1) Encode all test latents
    latents = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            mu, _ = model.encode(x_batch)       # [B, D]
            latents.append(mu.cpu())
    latents = torch.cat(latents, dim=0).numpy()  # (M, D)

    # 2) Fit PCA to 2 components
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(latents)           # (M, 2)
    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()

    # 3) Build uniform grid in PCA-space
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    grid_pts = np.array([[x, y] for y in ys for x in xs])  # (G^2, 2)

    # 4) Invert PCA grid -> latent space
    z_grid = pca.inverse_transform(grid_pts)            # (G^2, D)
    z_grid = torch.from_numpy(z_grid).to(device).float()

    # 5) Decode all grid latents
    with torch.no_grad():
        decoded = model.decode(z_grid).cpu().numpy()     # (G^2, C, H, W)

    # 6) Plot the grid of images
    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(grid_size*1.5, grid_size*1.5),
                             constrained_layout=True)
    for idx, img in enumerate(decoded):
        i = idx // grid_size
        j = idx % grid_size
        axes[i, j].imshow(img.transpose(1, 2, 0), vmin=0, vmax=1, rasterized=True)
        axes[i, j].axis('off')

    fig.suptitle(f"PCA reconstruction grid (G={grid_size})", fontsize=12, y=1.02)
    viz.save(fig, f"pca_grid_full_{grid_size}")


def run_pca_nn_grid(model, loader, viz, args, grid_size=100, border=2):
    """
    Fast mosaic version of the PCA NN grid with white borders.
      1) Encode all test latents mu(x)
      2) Fit PCA -> 2D coords (cached to pca_coords.npy)
      3) Define uniform grid in PCA space
      4) NN lookup & threshold -> select real samples
      5) Decode only needed latents
      6) Tile into one big white-bordered mosaic
      7) Render at dpi=25, one imshow, centered title
    """
    from sklearn.neighbors import KDTree

    print(">> Computing PCA mosaic grid")
    device = next(model.parameters()).device

    # 1) Encode all test latents
    latents = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb)
            latents.append(mu.cpu())
    latents = torch.cat(latents, dim=0).numpy()   # (M, D)

    # 2) Fit/load PCA coords
    pca_path = "pca_coords.npy"
    if not os.path.exists(pca_path):
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(latents)        # (M,2)
        np.save(pca_path, coords)
    else:
        coords = np.load(pca_path)                 # (M,2)

    x_coords, y_coords = coords[:,0], coords[:,1]

    # 3) Build uniform grid in PCA space
    xs = np.linspace(x_coords.min(), x_coords.max(), grid_size)
    ys = np.linspace(y_coords.min(), y_coords.max(), grid_size)
    grid_pts = np.array([[x,y] for y in ys for x in xs])  # (G^2,2)

    # 4) Nearest-neighbor lookup
    tree = KDTree(coords)
    dists, idxs = tree.query(grid_pts, k=1)
    dists = dists.ravel(); idxs = idxs.ravel()

    # 5) Threshold by grid spacing
    dx = xs[1]-xs[0] if grid_size>1 else x_coords.ptp()
    dy = ys[1]-ys[0] if grid_size>1 else y_coords.ptp()
    threshold = 0.5*max(dx, dy)
    valid = (dists <= threshold)

    # 6) Decode only the needed latents
    unique = np.unique(idxs[valid])
    with torch.no_grad():
        to_dec = torch.from_numpy(latents[unique]).to(device).float()
        decoded = model.decode(to_dec).cpu().numpy()  # (K, C, H, W)

    # 7) Build dict idx -> H x W x C images
    decoded_dict = {
        int(u): decoded[i].transpose(1,2,0)
        for i,u in enumerate(unique)
    }
    C, H, W = decoded.shape[1:]  # channels, height, width

    # 8) Assemble mosaic with embedded white border
    G = grid_size
    b = border
    M_h = G*H + (G+1)*b
    M_w = G*W + (G+1)*b
    mosaic = np.ones((M_h, M_w, C), dtype=np.float32)  # white background

    for k in range(G*G):
        if not valid[k]:
            continue
        i, j = divmod(k, G)
        y0 = b + i*(H + b)
        x0 = b + j*(W + b)
        mosaic[y0:y0+H, x0:x0+W, :] = decoded_dict[idxs[k]]

    # 9) Render at 25 DPI, one imshow, centered title
    dpi = 25
    plt.figure(figsize=(M_w/(dpi*2), M_h/(dpi*2)), dpi=dpi, facecolor='white')
    plt.imshow(mosaic, vmin=0, vmax=1, interpolation='nearest', rasterized=True)
    plt.axis('off')
    plt.title(f"PCA NN mosaic grid (G={G})", fontsize=150, loc='center', pad=5)
    plt.tight_layout(pad=0)
    viz.save(plt.gcf(), f"pca_nn_mosaic_{G}_bordered", dpi=dpi)


def run_pca_nn_grid_orig(model, loader, viz, args, grid_size=70, border=2):
    """
    Fast mosaic version of the PCA NN grid showing ORIGINAL images:
      1) Encode all test latents mu(x) and stash originals
      2) Fit / load PCA coords
      3) Uniform grid in PCA space
      4) NN lookup + threshold
      5) Build mosaic from ORIGINAL images
      6) Render at dpi=25 with white borders and centered title
    """
    from sklearn.neighbors import KDTree
    print(">> Computing PCA mosaic grid (orig images)")
    device = next(model.parameters()).device

    # 1) Encode and collect originals
    latents = []
    images  = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb)
            latents.append(mu.cpu().numpy())
            images.append(xb.cpu().numpy())
    latents = np.concatenate(latents, axis=0)  # (M, D)
    images  = np.concatenate(images,  axis=0)  # (M, C, H, W)

    # 2) PCA coords (cached)
    pca_path = "pca_coords.npy"
    if not os.path.exists(pca_path):
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(latents)
        np.save(pca_path, coords)
    else:
        coords = np.load(pca_path)
    x_coords, y_coords = coords[:,0], coords[:,1]

    # 3) Uniform grid in PCA space
    xs = np.linspace(x_coords.min(), x_coords.max(), grid_size)
    ys = np.linspace(y_coords.min(), y_coords.max(), grid_size)
    grid_pts = np.array([[x,y] for y in ys for x in xs])  # (G^2, 2)

    # 4) Nearest-neighbor + threshold
    tree = KDTree(coords)
    dists, idxs = tree.query(grid_pts, k=1)
    dists = dists.ravel(); idxs = idxs.ravel()
    dx = xs[1]-xs[0] if grid_size>1 else x_coords.ptp()
    dy = ys[1]-ys[0] if grid_size>1 else y_coords.ptp()
    threshold = 0.5 * max(dx, dy)  # 0.5 * max(dx, dy) to allow for some overlap
    valid = (dists <= threshold)

    # 5) Build orig_dict mapping index -> H x W x C array
    unique = np.unique(idxs[valid])
    orig_dict = {
        int(u): images[u].transpose(1,2,0)
        for u in unique
    }
    C, H, W = images.shape[1:]  # channels, height, width

    # 6) Assemble mosaic with embedded white border
    G = grid_size; b = border
    M_h = G*H + (G+1)*b
    M_w = G*W + (G+1)*b
    mosaic = np.ones((M_h, M_w, C), dtype=np.float32)

    for k in range(G*G):
        if not valid[k]:
            continue
        i, j = divmod(k, G)
        y0 = b + i*(H + b)
        x0 = b + j*(W + b)
        mosaic[y0:y0+H, x0:x0+W, :] = orig_dict[idxs[k]]

    # 7) Render
    dpi = 25
    plt.figure(figsize=(M_w/(dpi*2), M_h/(dpi*2)), dpi=dpi, facecolor='white')
    plt.imshow(mosaic, vmin=0, vmax=1, interpolation='nearest', rasterized=True)
    plt.axis('off')
    plt.title(f"PCA NN mosaic grid (orig, G={G})", fontsize=150, loc='center', pad=5)
    plt.tight_layout(pad=0)
    viz.save(plt.gcf(), f"pca_nn_mosaic_orig_{G}_bordered", dpi=dpi)




def plot_pca(coords, labels, class_names, viz, title_suffix, filename_suffix, exclude_dmso=False, max_points=10000):
    filtered_classes = [name for name in class_names if not (exclude_dmso and name == "DMSO")]
    cmap = plt.colormaps['tab20'].resampled(len(filtered_classes))

    selected_mask = np.full(len(labels), True)
    if exclude_dmso:
        dmso_idx = class_names.index("DMSO")
        selected_mask = (labels != dmso_idx)

    idx_all = np.where(selected_mask)[0]
    if len(idx_all) > max_points:
        idx_all = np.random.choice(idx_all, size=max_points, replace=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, class_name in enumerate(filtered_classes):
        true_idx = class_names.index(class_name)
        mask = (labels == true_idx) & np.isin(np.arange(len(labels)), idx_all)

        if not mask.any():
            continue

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)],
            alpha=1,
            s=5,
            label=class_name,
            marker='o',
            edgecolors='none'
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(
        title="MOA classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
        title_fontsize="small",
        markerscale=2
    )

    viz.save(fig, f"pca_labeled_{filename_suffix}")



def run_pca_labeled(model, loader, viz, args):
    """
    Compute PCA on test-set latents and color by MOA class,
    with and without DMSO shown.
    """

    print(">> Computing PCA on test set")

    # Step 1: Load metadata
    subset_indices = loader.dataset.indices
    cfg = model.cfg

    with h5py.File(cfg.data.metadata_path, 'r') as f:
        wells = f["metadata_well"][:].astype(str)
        compounds = f["metadata_compound"][:].astype(str)
        concentrations = f["metadata_concentration"][:].astype(str)
        moas = f["metadata_moa"][:].astype(str)

    meta_df = pd.DataFrame({
        "well": wells,
        "compound": compounds,
        "concentration": concentrations,
        "moa": moas
    })
    meta_df = meta_df.iloc[subset_indices].reset_index(drop=True)

    # Step 2: Encode
    device = next(model.parameters()).device
    model.eval()
    print(">> Encoding images into latent space")
    embeddings = []

    with torch.no_grad():
        for x_batch, _ in tqdm(loader, desc="Computing embeddings"):
            x_batch = x_batch.to(device)
            mu, _ = model.encode(x_batch)
            embeddings.append(mu.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # Step 3: Label mapping
    class_names = sorted(meta_df["moa"].unique())
    label_map = {moa: i for i, moa in enumerate(class_names)}
    labels = np.array([label_map[moa] for moa in meta_df["moa"]])

    # Step 4: PCA
    print(">> Computing PCA coordinates")
    if not os.path.exists("pca_coords.npy"):
        coords = PCA(n_components=2).fit_transform(embeddings)
        np.save("pca_coords.npy", coords)
    else:
        coords = np.load("pca_coords.npy")

    # Plot with and without DMSO
    plot_pca(coords, labels, class_names, viz, title_suffix="(with DMSO)", filename_suffix="with_dmso")
    plot_pca(coords, labels, class_names, viz, title_suffix="(no DMSO)", filename_suffix="no_dmso", exclude_dmso=True)
