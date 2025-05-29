import os
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, checkpoint_path: str):
        # Split path: get directory name and file name
        ckpt_dir, ckpt_file = os.path.split(checkpoint_path)

        # Base directory name (e.g., "vae_checkpoint_29-05-2025_17-45")
        base_name = os.path.basename(ckpt_dir)

        # Epoch name without extension (e.g., "epoch_45")
        epoch_name = os.path.splitext(ckpt_file)[0]

        # Construct full output directory path
        self.out_dir = os.path.join("experiments", "results", base_name, epoch_name)
        os.makedirs(self.out_dir, exist_ok=True)

    def save(self, fig, name, dpi=300):
        out = os.path.join(self.out_dir, f"{name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved {out}")