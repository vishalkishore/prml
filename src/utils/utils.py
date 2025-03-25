import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batch_size=128):
    """Loads CIFAR-10 dataset and returns a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def feature_extractor(extraction_method):
    """Decorator to extract features using the specified method."""
    def wrapper(data, **kwargs):
        features, labels = [], []
        
        # If data is a DataLoader, iterate over batches
        if isinstance(data, torch.utils.data.DataLoader):
            for images, targets in data:
                extracted_features = extraction_method(images, **kwargs)
                features.append(extracted_features)
                labels.extend(targets.numpy())
        
        # If data is a list of images, process directly
        elif isinstance(data, (list, tuple)):
            extracted_features = extraction_method(data, **kwargs)
            features.append(extracted_features)
        
        return np.concatenate(features, axis=0), np.array(labels)
    
    return wrapper

def apply_umap(features, n_components=2):
    """Applies UMAP for dimensionality reduction."""
    reducer = umap.UMAP(n_components=n_components, random_state=42,n_neighbors=20)
    return reducer.fit_transform(features)

def plot_umap(umap_embeddings, labels,class_names=None):
    """Plots the UMAP clustering results."""
    labels_ = labels
    if class_names is not None:
        labels_ = [class_names[label] for label in labels]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=labels_, palette="tab10", alpha=0.7)
    plt.legend(title="Class")
    plt.title("UMAP Clustering of CIFAR-10 Features")
    plt.show()


