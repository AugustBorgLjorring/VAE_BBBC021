import torch
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree


def run_tsne(model, loader, viz, args):
    print(">> Computing t-SNE")
    N = 500
    batch = []
    count = 0

    for xb, _ in loader:
        batch.append(xb)
        count += xb.size(0)
        if count >= N: break
    xs = torch.cat(batch, 0)[:N]
    with torch.no_grad():
        lat = torch.cat([model.encode(xs[i:i+1])[0] for i in range(N)], 0).cpu().numpy()
    coords = TSNE(n_components=2, random_state=0).fit_transform(lat)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0)
    for i, (x0, y0) in enumerate(coords):
        im = OffsetImage(xs[i].permute(1, 2, 0).cpu(), zoom=0.1)
        ax.add_artist(AnnotationBbox(im, (x0, y0), frameon=False))
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE of {N} samples", fontsize=12, y=1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    viz.save(fig, f"tsne_{N}")


def run_tsne_nn_grid(model, loader, viz, args, grid_size=70):
    """
    Fast "mosaic" version of the t-SNE NN grid with white grid lines.
    """

    print(">> Computing t-SNE mosaic grid")
    device = next(model.parameters()).device

    # 1-7) identical to before: encode latents, TSNE, grid_pts, KDTree, threshold, decode
    latents = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb)
            latents.append(mu.cpu())
    latents = torch.cat(latents, dim=0).numpy()

    if not os.path.exists("tsne_coords.npy"):
        coords = TSNE(n_components=2, random_state=0).fit_transform(latents)
        # Save t-SNE coords for later use
        np.save("tsne_coords.npy", coords)
    else:
        coords = np.load("tsne_coords.npy")
        
    xs = np.linspace(coords[:,0].min(), coords[:,0].max(), grid_size)
    ys = np.linspace(coords[:,1].min(), coords[:,1].max(), grid_size)
    grid_pts = np.array([[x,y] for y in ys for x in xs])

    tree = KDTree(coords)
    dists, idxs = tree.query(grid_pts, k=1)
    dists = dists.ravel(); idxs = idxs.ravel()

    dx = xs[1]-xs[0] if grid_size>1 else coords[:,0].ptp()
    dy = ys[1]-ys[0] if grid_size>1 else coords[:,1].ptp()
    threshold = max(dx, dy)
    valid = (dists <= threshold)

    unique = np.unique(idxs[valid])
    with torch.no_grad():
        to_decode = torch.from_numpy(latents[unique]).to(device).float()
        decoded = model.decode(to_decode).cpu().numpy()

    decoded_dict = {int(idx): decoded[i].transpose(1, 2, 0) for i, idx in enumerate(unique)}
    C, H, W = decoded.shape[1:]

    # build mosaic
    # 8) Tile into one big white-background mosaic
    G = grid_size
    b = 2
    M_h = G*H + (G+1)*b
    M_w = G*W + (G+1)*b
    mosaic = np.ones((M_h, M_w, C), dtype=np.float32)  # start all-white

    for k in range(G*G):
        if not valid[k]:
            continue
        i, j = divmod(k, G)
        y0 = b + i*(H+b)
        x0 = b + j*(W+b)
        mosaic[y0:y0+H, x0:x0+W, :] = decoded_dict[idxs[k]]

    # 9) Plot the single mosaic image
    dpi = 25
    plt.figure(figsize=(M_w / (dpi * 2), M_h / (dpi * 2)),dpi=dpi,facecolor='white')
    plt.imshow(mosaic, vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.title(f"t-SNE NN mosaic grid (G={G})", fontsize=150, loc='center', pad=5)
    plt.tight_layout(pad=0)
    viz.save(plt.gcf(), f"tsne_nn_mosaic_{grid_size}_bordered", dpi=dpi)



def run_tsne_nn_grid_orig(model, loader, viz, args, grid_size=70, border=2):
    """
    Fast mosaic version of the t-SNE NN grid showing ORIGINAL images:
      1) Encode all test latents mu(x) and stash originals
      2) Fit / load t-SNE coords
      3) Uniform grid in t-SNE space
      4) NN lookup + threshold
      5) Build mosaic from ORIGINAL images
      6) Render at dpi=25 with white borders and centered title
    """
    print(">> Computing t-SNE mosaic grid (orig images)")
    device = next(model.parameters()).device

    # 1) Encode and collect originals
    latents = []
    images  = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb)        # [B, D]
            latents.append(mu.cpu().numpy())
            images.append(xb.cpu().numpy())  # [B, C, H, W]
    latents = np.concatenate(latents, axis=0)  # (M, D)
    images  = np.concatenate(images,  axis=0)  # (M, C, H, W)

    # 2) t-SNE coords (cached)
    coords_path = "tsne_coords.npy"
    if not os.path.exists(coords_path):
        coords = TSNE(n_components=2, random_state=0).fit_transform(latents)
        np.save(coords_path, coords)
    else:
        coords = np.load(coords_path)
    x_coords, y_coords = coords[:,0], coords[:,1]

    # 3) Uniform grid in t-SNE space
    xs = np.linspace(x_coords.min(), x_coords.max(), grid_size)
    ys = np.linspace(y_coords.min(), y_coords.max(), grid_size)
    grid_pts = np.array([[x,y] for y in ys for x in xs])  # (G^2, 2)

    # 4) Nearest-neighbor + threshold
    tree = KDTree(coords)
    dists, idxs = tree.query(grid_pts, k=1)
    dists = dists.ravel(); idxs = idxs.ravel()
    dx = xs[1]-xs[0] if grid_size>1 else x_coords.ptp()
    dy = ys[1]-ys[0] if grid_size>1 else y_coords.ptp()
    threshold = 0.5*max(dx, dy)  # 0.5 * max(dx, dy) to allow for some overlap
    valid = (dists <= threshold)

    # 5) Build orig_dict mapping index -> H x W x C array
    unique = np.unique(idxs[valid])
    orig_dict = {
        int(u): images[u].transpose(1,2,0)  # (H, W, C)
        for u in unique
    }
    C, H, W = images.shape[1:]  # channels, height, width

    # 6) Assemble mosaic with embedded white border
    G = grid_size; b = border
    M_h = G*H + (G+1)*b
    M_w = G*W + (G+1)*b
    mosaic = np.ones((M_h, M_w, C), dtype=np.float32)  # white

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
    plt.title(f"t-SNE NN mosaic grid (orig, G={G})", fontsize=150, loc='center', pad=5)
    plt.tight_layout(pad=0)
    viz.save(plt.gcf(), f"tsne_nn_mosaic_orig_{G}_bordered", dpi=dpi)
    plt.close()


def run_tsne_labeled(model, loader, viz, args, index_file="label_index.npy", label_file="moa_labels_y.npy"):
    """
    Compute t-SNE on all test-set latents and color by class,
    plotting each class separately so legend entries are exact.
    """

    print(">> Computing t-SNE on full test set")

    device = next(model.parameters()).device
    class_names = np.load(index_file, allow_pickle=True)
    labels      = np.load(label_file)    # shape (N_test,)

    # gather all test images into one tensor
    xs = torch.cat([xb for xb,_ in loader], dim=0).to(device)
    N_test = xs.size(0)

    # encode all into latents mu(x)
    model.eval()
    with torch.no_grad():
        mus = []
        for b in torch.split(xs, 64):
            mu_batch, _ = model.encode(b)
            mus.append(mu_batch.cpu())
        lat = torch.cat(mus, dim=0).numpy()

    # run t-SNE
    if not os.path.exists("tsne_coords.npy"):
        coords = TSNE(n_components=2, random_state=0).fit_transform(lat)
        # Save t-SNE coords for later use
        np.save("tsne_coords.npy", coords)
    else:
        coords = np.load("tsne_coords.npy")

    # prepare a discrete colormap
    cmap = plt.colormaps['tab20'].resampled(len(class_names))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, name in enumerate(class_names):
        mask = (labels == idx)
        if not mask.any():
            continue

        # choose alpha & size per class (e.g. class 0 more transparent)
        alpha = 0.1 if idx == 0 else 1
        size  = 2    if idx == 0 else 3
        marker_type = 'x' if idx == 0 else 'o'
        edgecolor = None if idx == 0 else 'none'

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(idx)],  # single-color list
            alpha=alpha, s=size,
            label=name,
            marker=marker_type,
            linewidths=0.5,
            edgecolors=edgecolor
        )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE of {N_test} test samples, by class")
    ax.grid(True, linestyle="--", alpha=0.3)

    # automatic legend: entries appear in the same order as you scatter them
    ax.legend(
        title="MOA classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
        title_fontsize="small",
        markerscale=2  # make legend markers a bit larger
    )

    viz.save(fig, f"tsne_labeled_{N_test}")