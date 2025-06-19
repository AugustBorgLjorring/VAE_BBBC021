# Variational Autoencoder (VAE) for BBBC021 Single-Cell Imaging

This repository contains code for training and evaluating **Variational Autoencoders (VAEs)** on the **BBBC021** single-cell microscopy dataset. The goal is to learn interpretable latent representations of cell morphology through unsupervised generative modeling.

---

## 🧬 Project Overview

We compare two generative models:

* **Standard VAE**: Trained with ELBO loss
* **VAE+**: An enhanced version with adversarial feature-matching loss to improve reconstruction sharpness

The models aim to balance **image reconstruction quality** and **latent space interpretability**, with applications in phenotypic profiling.

---

## 📂 Dataset: BBBC021 (Single-Cell)

The dataset is derived from [BBBC021](https://bbbc.broadinstitute.org/BBBC021), a publicly available fluorescence microscopy collection from the Broad Bioimage Benchmark Collection.

* Fluorescence microscopy images of MCF-7 breast cancer cells
* Each single-cell image is 68×68 pixels with 3 channels:

  * **Red (DAPI)** – nucleus
  * **Blue (Tubulin)** – microtubules
  * **Green (Actin)** – actin filaments
* Annotated by **compound**, **concentration**, and **Mechanism of Action (MoA)**

---

## ⚙️ Pipeline Summary

### Preprocessing

* Per-channel max normalization
* Well-based train/validation/test split (to avoid data leakage)
* Random rotations and flips

### Training

* **Standard VAE** loss:

  $$
  L_{\text{VAE}} = E_{q(z \mid x)}[\log p(x \mid z)] - D_{\text{KL}}(q(z \mid x) \| p(z))
  $$
* **VAE+** adds feature-matching loss from an auxiliary discriminator

### Evaluation

* Metrics: ELBO, reconstruction loss, MMD, NSC accuracy
* Latent analysis: PCA, t-SNE, traversals, heatmaps, gradient sensitivity

---

## 🧪 Usage

```bash
# Preprocess data
python data/preprocess.py

# Train the model
python src/train.py
```

Configuration is handled using [Hydra](https://hydra.cc/), and experiment tracking is integrated with [Weights & Biases](https://wandb.ai/).

---

## 👥 Authors

* August Borg Ljørring – s224178
* August Emil Holm Jørgensen – s224166

📍 *Technical University of Denmark (DTU)*

---

## 📚 References

* Kingma & Welling, *Auto-Encoding Variational Bayes*, 2014
* Lafarge et al., *Capturing Single-Cell Phenotypic Variation via Adversarial Training*, 2019
