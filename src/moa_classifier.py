from data_processing import load_data
import argparse
import torch
from omegaconf import OmegaConf
from vae_model import VAE, BetaVAE
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def load_model_and_cfg(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    cfg = OmegaConf.create(checkpoint['config'])  # Load config from checkpoint
    # Override bbbc021 train_path in the config if needed
    cfg.data.train_path = "data/raw/BBBC021_cleaned_preprocessed.h5"

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

def plot_confusion_matrix(cm, class_names, figsize=(10, 8), title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    # Set tick marks and labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color)
    
    fig.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "-c", required=True, help="Path to .pth checkpoint")
    args = p.parse_args()
    print("Loading model + config…")

    model, cfg = load_model_and_cfg(args.checkpoint)
    data_loader = load_data(cfg, split="test")
    metadata = pd.read_csv("data/raw/metadata.csv", index_col="Single_Cell_Image_Name")
    metadata["moa_code"], label_index = pd.factorize(metadata["moa"])
    print(f"Found {len(label_index)} MOA classes:", label_index.tolist())
    np.save("label_index.npy", label_index)
    num_classes = len(label_index)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    # Lists for storing latents and labels
    latents = []
    labels = []


    model.eval()
    model.to(device)
    with torch.no_grad():
        for x_batch, name_batch in tqdm(data_loader):
            
            x = x_batch.to(device)
            mu, _ = model.encode(x)
            latents.append(mu.cpu())
            
            moa_codes = metadata.loc[list(name_batch), "moa_code"].tolist()
            labels.append(torch.tensor(moa_codes, dtype=torch.long))

    Z = torch.cat(latents, dim=0)   # [N, latent_dim]
    y = torch.cat(labels,  dim=0) 
    # Save y as npy
    np.save("moa_labels_y.npy", y.cpu().numpy())
    print(f"Latents shape: {Z.shape}")
    print(f"Labels shape: {y.shape}")

    # Classifier LDA

    Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=42)
    # Print unique Y-test values coubt
    print(f"Unique Y-test values: {len(set(y_test.numpy()))}")
    np.save("Z_test.npy", Z_test.cpu().numpy())
    # Multinomial Softmax Classifier
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced"
    )

    # Random Forest Classifier
    # clf = RandomForestClassifier(
    #     n_estimators=200,
    #     random_state=42,
    #     n_jobs=-1,
    # )

    # XGBoost Classifier
    # clf = XGBClassifier(
    #     objective='multi:softprob',
    #     num_class=num_classes,
    #     eval_metric='mlogloss',
    #     use_label_encoder=False,
    #     n_estimators=200,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     random_state=42,
    #     n_jobs=-1,
    # )
    

    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    np.save("y_pred.npy", y_pred)
    # Classification report
    report_dict = classification_report(
        y_test, y_pred,
        target_names=label_index,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    # Round floats to 2 decimals for neatness
    report_df[['precision','recall','f1-score']] = report_df[['precision','recall','f1-score']].round(2)


    # print("\n% --- Classification Report (LaTeX) ---")
    # print(report_df.to_latex(columns=['precision','recall','f1-score','support'], 
    #                          header=['Prec.','Recall','F1','Support'], 
    #                          index_names=['Class'], 
    #                          float_format="%.2f"))
    # print("% -------------------------------------\n")

    # # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion matrix:\n", cm)

    # Print classification report
    print("\n% --- Classification Report ---")
    print(classification_report(
        y_test, y_pred,
        target_names=label_index
    ))
    print("% -------------------------------------\n")
    # print(report_df)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # print("Confusion matrix:\n", cm)
    # # Plot confusion matrix
    # plot_confusion_matrix(cm, class_names=label_index, title='Confusion Matrix')


    # after computing cm and uniques:
    labels_to_remove = ["DMSO", "Microtubule stabilizers"]
    idxs = [label_index.get_loc(lbl) for lbl in labels_to_remove]


    cm_pruned = cm.copy()
    for idx in sorted(idxs, reverse=True):
        cm_pruned = np.delete(cm_pruned, idx, axis=0)
        cm_pruned = np.delete(cm_pruned, idx, axis=1)


    names_pruned = [n for n in label_index if n not in labels_to_remove]

    plot_confusion_matrix(
    cm_pruned,
    names_pruned,
    figsize=(10, 8),
    title="Confusion Matrix")

if __name__ == "__main__":
    main()




