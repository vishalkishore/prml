import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

def extract_hog(img):
    """
    Extract HOG features and return as an image.
    Specifically tailored for processing CIFAR-10 images.
    """
    import cv2
    import numpy as np
    from skimage.feature import hog
    
    # Ensure the input image is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Scale if needed
    
    # Convert the RGB image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Extract HOG features and visualization
    feature, hog_image = hog(
        gray,
        orientations=9,  # Number of gradient bins
        pixels_per_cell=(8, 8),  # Size of cell in pixels
        cells_per_block=(2, 2),  # Size of block in cells
        block_norm='L2-Hys',  # Normalization method
        visualize=True,       # Return the HOG image
        transform_sqrt=True   # Apply power law compression to normalize image brightness
    )
    
    return hog_image


def extract_edges(img):
    """Apply Sobel edge detection."""
    import cv2
    import numpy as np
    from scipy import ndimage
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) 
    red_edges = ndimage.sobel(img[:,:,0])
    green_edges = ndimage.sobel(img[:,:,1]) 
    blue_edges = ndimage.sobel(img[:,:,2])
    edges = red_edges | green_edges | blue_edges
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray, 100, 200)  # Apply Canny Edge Detection
    return edges

import cv2
import numpy as np

def extract_sift(img):
    """Extracts SIFT keypoints and descriptors and returns an image with keypoints drawn."""
    sift = cv2.SIFT_create() 
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    sift_img = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_img

def extract_gabor(img, ksize=5, sigma=1.0, theta=np.pi/4, lambd=10.0, gamma=0.5):
    """Extracts Gabor filter features."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
    
    return filtered_img

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Load Pretrained ResNet-18 and remove the fully-connected layers
resnet18 = models.efficientnet_b0(pretrained=True)
resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-2])
resnet18.eval()  # Set to evaluation mode

# Define the transformation pipeline for CIFAR-10 images
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),            # Convert NumPy image to PIL image
    transforms.Resize((224, 224)),        # Resize image to 224x224 for ResNet-18
    transforms.ToTensor(),                # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_resnet_features(img):
    """
    Extracts feature maps from ResNet-18.
    Expected output shape is (512, H, W).
    """
    # Ensure image is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Apply the transformation pipeline and add a batch dimension
    img_tensor = transform_pipeline(img).unsqueeze(0)

    with torch.no_grad():  # Inference mode
        features = resnet18(img_tensor)  # Use the pre-loaded model instance

    # Remove the batch dimension: expected shape (512, H, W)
    features = features.squeeze(0)

    if len(features.shape) == 3:
        return features.cpu().numpy()
    else:
        raise ValueError(f"Unexpected feature shape: {features.shape}")

def extract_resnet_feature_map(img):
    """
    Extracts and visualizes feature maps from ResNet-18 using PCA.
    """
    features = extract_resnet_features(img)  # Expected shape: (512, H, W)

    # Reshape features from (512, H, W) to (H*W, 512) for PCA
    H, W = features.shape[1], features.shape[2]
    reshaped_features = features.reshape(512, -1).T

    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    feature_map = pca.fit_transform(reshaped_features)

    # Reshape back to (H, W) and normalize for visualization
    feature_map = feature_map.reshape(H, W)
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

    return feature_map

def extract_resnet_feature_map_norm(img):
    """
    Extracts and visualizes a feature map from ResNet-18 by computing the L2 norm
    across the channel dimension. This produces a single 2D feature map.
    """
    features = extract_resnet_features(img)  # Expected shape: (512, H, W)

    # Compute the L2 norm over the channel dimension (axis 0)
    feature_map = np.linalg.norm(features, axis=0)
    
    # Normalize the feature map to the [0, 1] range for visualization
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    
    return feature_map
