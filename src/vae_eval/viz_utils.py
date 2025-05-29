import os
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, checkpoint_path: str):
        ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        self.out_dir = os.path.join("experiments", "results", ckpt_name)
        os.makedirs(self.out_dir, exist_ok=True)

    def save(self, fig, name, dpi=300):
        out = os.path.join(self.out_dir, f"{name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved {out}")