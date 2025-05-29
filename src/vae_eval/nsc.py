# load data and metadata
import numpy as np
from src.data_loading import load_data, load_metadata
from src.vae_eval.model_utils import load_model_and_cfg
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
import h5py

# LOKY_MAX_CPU_COUNT
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def run_nsc(model, loader, viz, args):
    cfg = model.cfg
    cfg.data.train_path = "D:/BBBC021/BBBC021_dataset_cleaned_maxnorm_68.h5"
    cfg.data.metadata_path = "data/raw/metadata_dataset.h5"

    # use a different loader as we want to load all data
    data_loader = load_data(cfg, split="all")
    data_metadata = load_metadata(cfg)

    print(f"Loaded {len(data_loader.dataset)} images and {len(data_metadata.dataset)} metadata entries.")
    print(f"Loaded {len(data_loader)} batches of images and {len(data_metadata)} metadata batches.")


    ###########################################################
    # Compute embeddings for all images using the VAE model
    ###########################################################
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model is on device: {device}")

    # Collect all embeddings
    embeddings = []

    with torch.no_grad():
        for x_batch, _ in tqdm(data_loader, desc="Computing embeddings"):
            x_batch = x_batch.to(device)
            mu, _ = model.encode(x_batch)  # Get mu only if you're using deterministic mean

            embeddings.append(mu.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    print(f"Total embeddings shape: {embeddings.shape} (should be [N, D])")




    ###########################################################
    # Now we have embeddings and metadata, let's combine them
    ###########################################################
    # Convert metadata and embeddings into a DataFrame
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

    print(f"Metadata shape: {meta_df.shape} (should be [N, 4])")

    meta_df["embedding"] = list(embeddings)

    print(f"Metadata embedding shape: {meta_df['embedding'].shape} (should be [N, D])")




    #############################################################
    # Group by well and compute average embeddings
    #############################################################
    # Group by well and average embeddings
    well_profiles = meta_df.groupby("well").agg({
        "embedding": lambda x: np.mean(np.stack(x), axis=0),
        "compound": "first",
        "concentration": "first",
        "moa": "first"
    }).reset_index()

    # print the first well profile embedding
    print(f"First well profile embedding: {well_profiles['embedding'].iloc[0].shape}")

    ##############################################################
    # Now we can compute treatment profiles by averaging 
    # embeddings over compound and concentration pairs
    ##############################################################
    # A treatment is uniquely defined by (compound, concentration)
    treatment_profiles = well_profiles.groupby(["compound", "concentration"]).agg({
        "embedding": lambda x: np.median(np.stack(x), axis=0),
        "moa": "first"
    }).reset_index()

    # print the first treatment profile embedding
    print(f"First treatment profile embedding: {treatment_profiles['embedding'].iloc[0].shape}")

    ###############################################################
    # Perform Leave-One-Compound-Out Cross-Validation (LOCO-CV)
    ###############################################################
    from collections import defaultdict, Counter

    compound_list = treatment_profiles["compound"].unique()
    compound_preds = defaultdict(list)

    for compound in tqdm(compound_list, desc="LOCO-CV"):
        if compound == "DMSO": # Skip DMSO as it is a control and therefore will not be in more than one compound
            continue
        train_set = treatment_profiles[treatment_profiles["compound"] != compound]
        test_set = treatment_profiles[treatment_profiles["compound"] == compound]

        X_train = np.stack(train_set["embedding"])
        y_train = train_set["moa"].values
        X_test = np.stack(test_set["embedding"])
        y_test = test_set["moa"].values

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Store all predictions per compound
        for true, pred in zip(y_test, y_pred):
            compound_preds[compound].append((true, pred))

    # Build final dataframe
    rows = []
    for compound, pairs in compound_preds.items():
        true_moas = [t for t, _ in pairs]
        pred_moas = [p for _, p in pairs]

        true_moa = true_moas[0]  # should be consistent
        acc = np.mean([t == p for t, p in pairs])

        # Count predictions and format as ['3xA', '1xB']
        pred_counts = Counter(pred_moas)
        formatted_preds = [f"{v}x{m}" for m, v in pred_counts.items()]

        rows.append({
            "compound": compound,
            "true_moa": true_moa,
            "pred_moa": formatted_preds,
            "accuracy": acc
        })

    final_df = pd.DataFrame(rows)

    print(final_df)
    print(f"\nNSC LOCO-CV Accuracy (mean over compounds): {final_df['accuracy'].mean():.4f}")

    # Flatten all (true, pred) pairs for confusion matrix
    # Step 1: Flatten all predictions
    flat_rows = []
    for compound, pairs in compound_preds.items():
        for true_moa, pred_moa in pairs:
            flat_rows.append({
                "compound": compound,
                "true_moa": true_moa,
                "pred_moa": pred_moa,
                "compound_label": f"{true_moa} ({compound})"
            })

    df = pd.DataFrame(flat_rows)

    # Step 2: Pivot to get heatmap matrix
    heatmap_data = pd.crosstab(df["compound_label"], df["pred_moa"])

    # Step 3: Move 'DMSO' column to the end if present
    columns = list(heatmap_data.columns)
    if "DMSO" in columns:
        columns = [col for col in columns if col != "DMSO"] + ["DMSO"]
        heatmap_data = heatmap_data[columns]

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Step 4: Plot
    plt.figure(figsize=(14, max(6, 0.25 * len(heatmap_data))))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=True)

    plt.title("NSC: Predictions per Compound (True MoA shown on Y-axis)")
    plt.xlabel("Predicted MoA")
    plt.ylabel("True MoA (Compound)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    viz.save(plt.gcf(), "nsc_heatmap")
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    import matplotlib.lines as mlines

    # PCA
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(np.stack(treatment_profiles["embedding"]))
    pc1, pc2 = pca_embeddings[:, 0], pca_embeddings[:, 1]

    compounds = treatment_profiles["compound"]
    moas = treatment_profiles["moa"]

    # Unique identifiers
    unique_compounds = np.unique(compounds)
    unique_moas = sorted(np.unique(moas))  # sorted MOAs

    # Assign consistent color and shape
    moa_palette = sns.color_palette("tab20", len(unique_moas))
    moa_color_dict = {moa: color for moa, color in zip(unique_moas, moa_palette)}

    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', 'H', '8', '<', '>', 'p', '|', '_']
    compound_marker_dict = {compound: markers[i % len(markers)] for i, compound in enumerate(unique_compounds)}

    # Plot all points
    plt.figure(figsize=(12, 9))
    sns.set_theme(style="whitegrid")

    # For building the legend later
    legend_entries = []

    for i in range(len(treatment_profiles)):
        compound = compounds[i]
        moa = moas[i]
        label = f"{compound} ({moa})"
        
        # Plot point
        plt.scatter(
            pc1[i],
            pc2[i],
            c=[moa_color_dict[moa]],
            marker=compound_marker_dict[compound],
            s=120,
            edgecolor='black',
            linewidth=0.5,
        )

        # Keep track of (label, color, marker) — avoiding duplicates
        if label not in [entry[0] for entry in legend_entries]:
            legend_entries.append((label, moa_color_dict[moa], compound_marker_dict[compound], moa))

    # Sort legend entries by MOA
    legend_entries_sorted = sorted(legend_entries, key=lambda x: x[3])  # sort by moa name

    # Create legend handles
    handles = [
        mlines.Line2D([], [], marker=marker, color=color, label=label,
                    markersize=10, linestyle='None', markeredgecolor='black', markeredgewidth=0.5)
        for label, color, marker, _ in legend_entries_sorted
    ]

    # Add title and axis
    plt.title("PCA of Treatment Profiles", fontsize=16, weight="bold", pad=15)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=13)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=13)

    # Custom legend (sorted by MOA)
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9, title="Compound (MOA)")
    plt.tight_layout()
    sns.despine()

    # Save
    viz.save(plt.gcf(), "nsc_pca")