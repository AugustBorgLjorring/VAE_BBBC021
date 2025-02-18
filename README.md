# Variational Autoencoder (VAE) on BBBC021 Dataset

This repository contains an implementation of a **Variational Autoencoder (VAE)** applied to the **BBBC021** dataset. The dataset consists of fluorescence microscopy images of human cells treated with various drugs, making it a valuable resource for studying biological variability and deep generative modeling.

---

## 📦 Environment Setup

To create the Conda environment with all necessary dependencies, run:

```bash
conda create -n VAE python=3.12 numpy matplotlib scipy scikit-learn pandas seaborn jupyter tqdm pytorch torchvision torchaudio cudatoolkit=11.8 tensorflow keras umap-learn opencv imageio pyyaml
```

Activate the environment:

```bash
conda activate VAE
```

---

## 📊 BBBC021 Dataset

The **BBBC021 dataset** is a fluorescence microscopy dataset designed for high-content screening of drug treatments. It contains images of human cells stained for various cellular structures, enabling detailed biological analysis.

### 🔬 **Original Dataset Structure**
Each image follows this filename pattern:

```
Week1_150607_{well}_{site}_{channel}{id}.tif
```

- **well:** Identifies the well location (e.g., B10, F12, G01).
- **site:** Each well is imaged at four different sites (s1, s2, s3, s4) to capture spatial variability.
- **channel:**
  - **w4 (Red - Actin):** Stains actin filaments (cytoskeleton).
  - **w2 (Green - Tubulin):** Stains microtubules (cytoskeleton).
  - **w1 (Blue - DAPI):** Stains the nucleus (DNA).
- **id:** Unique identifier for each image, corresponding to a drug-treatment pair.

### 📜 **Metadata**
The dataset includes a metadata file: `BBBC021_v1_image.csv`, which contains information for each image, such as:

- **Plate and well identifiers** to track experimental conditions.
- **Replicate numbers** ensuring reproducibility (three replicates per treatment).
- **Drug and concentration information** for each treatment condition.

### 🧪 **Experimental Design**
- **3 biological replicates** per drug-concentration pair.
- **4 sites per well** capturing spatial variations.
- **3 fluorescence channels per site**, leading to **36 images per treatment condition**.

### 📁 **Single-Cell Dataset**
For single-cell analysis, segmented images are stored as:

```
Week1_150607_{well}_{site}_{channel}{id}_{cell#}.npy
```

Each file represents a single-cell image with an associated **metadata file**, which links:
- The **original multi-cell image**.
- The **drug and concentration** used.
- The **mechanism of action (MoA)** of the treatment.

---

## 🏗️ Preprocessing Pipeline

To prepare the dataset for training, we apply:
1. **Image resizing** for standardized input dimensions.
2. **Normalization** to ensure stable model training.
3. **Data augmentation** to improve generalization.
4. **Dataset splitting** into training, validation, and test sets.

This process transforms high-dimensional, heterogeneous raw images into a standardized format for deep generative modeling.

---

## 🔢 Variational Autoencoder (VAE)

A Variational Autoencoder (VAE) is a type of generative model that learns latent representations of high-dimensional data. The **VAE loss function** consists of two terms:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{White}\mathcal{L}=\mathbb{E}_{q(z|x)}[\log&space;p(x|z)]-D_{KL}(q(z|x)\|p(z))" alt="VAE Loss Function">
</div>

where:
- **$q(z|x)$** is the approximate posterior distribution.
- **$p(x|z)$** is the likelihood of reconstructing input data.
- **$D_{KL}$** is the Kullback-Leibler divergence ensuring a well-structured latent space.

By minimizing this loss, VAEs can generate realistic biological images while learning meaningful representations.

---

## 🛠️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/AugustBorgLjorring/VAE_BBBC021.git
   cd VAE_BBBC021
   ```

2. Install dependencies (if not using Conda):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

4. Train the VAE model:
   ```bash
   python train_vae.py
   ```

---

## 📢 Conclusion

The **BBBC021 dataset** presents a rich yet challenging resource for **Variational Autoencoders (VAE)** in biological imaging. Through a **carefully designed preprocessing pipeline**, we transform raw microscopy images into a structured dataset suitable for generative modeling. This ensures that subsequent analyses uncover meaningful **latent representations** that reflect underlying biological variations while maintaining **scientific rigor and ethical research standards**.

---

## 📜 References
- Ljosa, V., Sokolnicki, K. L., & Carpenter, A. E. (2012). Annotated high-throughput microscopy image sets for validation. *Nature Methods*, **9**(7), 637.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.

---

## 🛠️ Authors

- **August Borg Ljørring** - s224178  
- **August Emil Holm Jørgensen** - s224166  

📍 From **Technical University of Denmark (DTU)**  

🔗 GitHub Repository: [VAE_BBBC021](https://github.com/AugustBorgLjorring/VAE_BBBC021)
