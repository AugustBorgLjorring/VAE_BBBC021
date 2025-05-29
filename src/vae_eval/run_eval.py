import argparse
from model_utils   import load_model_and_cfg
from src.data_loading    import load_data
from viz_utils     import Visualizer

# Import all tasks
from recon         import run_reconstruct, run_reconstruct_split_channels
from interpolate   import run_interpolate_seq, run_interpolate_lin
from perturb       import run_perturb
from tsne          import run_tsne, run_tsne_labeled, run_tsne_nn_grid, run_tsne_nn_grid_orig
from pca           import run_pca_grid, run_pca_nn_grid, run_pca_nn_grid_orig
from latent        import run_roundtrip, run_plot_latent, run_latent_usage, run_top_latent_dims
from sensitivity   import run_gradient_sensitivity, run_traversal_sensitivity
from nsc           import run_nsc

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

TASKS = {
    "recon": run_reconstruct,
    "recon_split": run_reconstruct_split_channels,
    # "interp_seq": run_interpolate_seq,
    "interp_lin": run_interpolate_lin,
    "perturb": run_perturb,
    "tsne": run_tsne,
    # "tsne_labeled": run_tsne_labeled,
    "tsne_nn": run_tsne_nn_grid,
    "tsne_nn_orig": run_tsne_nn_grid_orig,
    # "pca": run_pca_grid,
    "pca_nn": run_pca_nn_grid,
    "pca_nn_orig": run_pca_nn_grid_orig,
    "roundtrip": run_roundtrip,
    "plot_latent": run_plot_latent,
    "latent_usage": run_latent_usage,
    "top_latent": run_top_latent_dims,
    "sens_grad": run_gradient_sensitivity,
    # "sens_traverse": run_traversal_sensitivity,
    "nsc": run_nsc,
}

def parse_args():
    p = argparse.ArgumentParser("Evaluate VAE")
    p.add_argument("-c","--checkpoint", required=True)
    p.add_argument("-t","--tasks", nargs="+", choices=list(TASKS)+["all"], default=["all"])

    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--image_index", type=int, default=0)

    p.add_argument("--interp_idx", nargs=2, type=int, default=[0,1])

    p.add_argument("--perturb_eps", type=float, default=50.0)

    p.add_argument("--sens_n", type=int, default=100)
    p.add_argument("--sens_eps", type=float, default=0.1)

    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    return p.parse_args()

def main():
    args       = parse_args()
    model      = load_model_and_cfg(args.checkpoint)
    cfg        = model.cfg
    loader     = load_data(cfg, split=args.split)
    viz        = Visualizer(args.checkpoint)

    to_run = TASKS.keys() if "all" in args.tasks else args.tasks
    for name in to_run:
        TASKS[name](model, loader, viz, args)

if __name__=="__main__":
    main()
