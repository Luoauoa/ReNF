import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import os

def visualize_tsne_with_labels(data, labels, title="t-SNE Visualization with Labels", save_path=None, perplexity=30, n_components=2):
    """
    Perform t-SNE dimensionality reduction and create a scatter plot visualization with labels.
    
    Args:
        data: Input data array for t-SNE (shape: [n_samples, n_features])
        labels: Labels for each sample (shape: [n_samples])
        title: Title for the plot
        save_path: Path to save the plot (optional)
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of dimensions for t-SNE output (2 or 3)
    """
    print(f"Input data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(perplexity=perplexity, n_components=n_components)
    X_embedded = tsne.fit_transform(data)
    
    print(f"t-SNE output shape: {X_embedded.shape}")
    
    # Create the scatter plot with different colors for each label
    if n_components == 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if n_components == 3:
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], X_embedded[mask, 2],
                      c=[colors[i]], label=f'Rep {label}', alpha=0.7, s=50)
        else:
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                      c=[colors[i]], label=f'Rep {label}', alpha=0.7, s=50)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    if n_components == 3:
        ax.set_zlabel('t-SNE Component 3', fontsize=12)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if n_components == 2:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(True, alpha=0.3)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return X_embedded

def visualize_tsne(data, title="t-SNE Visualization", save_path=None, perplexity=30, n_components=2):
    """
    Perform t-SNE dimensionality reduction and create a scatter plot visualization.
    
    Args:
        data: Input data array for t-SNE (shape: [n_samples, n_features])
        title: Title for the plot
        save_path: Path to save the plot (optional)
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of dimensions for t-SNE output (2 or 3)
    """
    print(f"Input data shape: {data.shape}")
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(perplexity=perplexity, n_components=n_components)
    X_embedded = tsne.fit_transform(data)
    
    print(f"t-SNE output shape: {X_embedded.shape}")
    
    # Create the scatter plot
    if n_components == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], alpha=0.7, s=50)
        ax.set_zlabel('t-SNE Component 3', fontsize=12)
    else:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, s=50)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return X_embedded

def main():
    """
    Main function to run t-SNE visualization with labels for each representation layer.
    """
    print("Loading data...")
    np.random.seed(42)
    filepath = './rep_store.npy'
    data = np.load(filepath, allow_pickle=True)
    print("Available representations:", list(data.item().keys()))

    # Load and process all representations
    all_data = []
    all_labels = []
    
    for i, key in enumerate(sorted(data.item().keys())):
        rep_data = data.item()[key]
        b, v, t = rep_data.shape
        rep_data_flat = rep_data.reshape(b, -1)
        
        all_data.append(rep_data_flat)
        all_labels.extend([i] * b)  # Label each sample with its representation layer index
        
        print(f"{key}: shape {rep_data.shape} -> flattened {rep_data_flat.shape}")
    
    # Concatenate all data
    r_all = np.concatenate(all_data, axis=0)
    labels = np.array(all_labels)
    
    print(f"Combined data shape: {r_all.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Perform t-SNE visualization with labels
    X_embedded = visualize_tsne_with_labels(
        data=r_all,
        labels=labels,
        title="t-SNE Visualization of Different Representation Layers",
        save_path="tsne_plot_with_labels.pdf",
        perplexity=50,  # Adjust perplexity based on data size
        n_components=2,  # Use 2D visualization
    )
    
    print("t-SNE Visualization complete!")

if __name__ == "__main__":
    main()
