import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batch_size=128,transform=True):
    """Loads CIFAR-10 dataset and returns a DataLoader."""
    t = [
        transforms.Resize(224),
        transforms.ToTensor(),
        
    ]
    if transform:
        t.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(t)
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
import torch
import numpy as np

def sample_class_images(dataset):
    """
    Select one image per class from a dataset.
    
    Args:
        dataset: A PyTorch dataset
    
    Returns:
        A tuple containing:
        - List of images (numpy arrays) with one image per class
        - List of corresponding class labels
    """
    import random
    class_images = {i: None for i in range(10)}
    found_classes = set()

    # Use dataset directly if it supports indexing
    try:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for idx in indices:
            img, label = dataset[idx]
            
            # Convert label to integer if it's a tensor
            label = label.item() if torch.is_tensor(label) else label
            
            # Convert image to numpy if it's a tensor
            if torch.is_tensor(img):
                img = img.numpy().transpose(1, 2, 0)
            
            # Store the first image for each class
            if class_images[label] is None:
                class_images[label] = img
                found_classes.add(label)
            
            # Stop if we've found an image for each class
            if len(found_classes) == 10:
                break
    
    except TypeError:
        # Fallback to DataLoader if direct indexing fails
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        for img, label in dataloader:
            # Convert label and image to appropriate format
            label = label.item()
            img = img.squeeze(0).numpy().transpose(1, 2, 0)
            
            # Store the first image for each class
            if class_images[label] is None:
                class_images[label] = img
                found_classes.add(label)
            
            # Stop if we've found an image for each class
            if len(found_classes) == 10:
                break

    # Verify that we found images for all classes
    if any(image is None for image in class_images.values()):
        raise ValueError(f"Could not find images for all classes. Found images for classes: {found_classes}")

    # Return list of images and corresponding class labels
    return [class_images[i] for i in range(10)], list(class_images.keys())