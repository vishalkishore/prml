
import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

# Config
N_CLUSTERS, N_SAMPLES = 150, 15000
np.random.seed(26)

def extract_features(img):
    # Convert to grayscale
    img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # Extract HOG features
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    # Extract SIFT features
    try:
        sift = cv2.SIFT_create(nfeatures=50)
        _, desc = sift.detectAndCompute(gray, None)
        desc_feat = desc.mean(axis=0) if desc is not None and desc.size > 0 else np.zeros(128)
    except:
        desc_feat = np.zeros(128)
        
    if not hasattr(extract_features, "model"):
        extract_features.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        extract_features.model = nn.Sequential(*list(extract_features.model.children())[:-1])
        extract_features.model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_resized = transform(img).unsqueeze(0) 
    with torch.no_grad():
        effnet_feat = extract_features.model(img_resized).flatten().numpy()
    
    return np.hstack([desc_feat, effnet_feat])

# Build vocabulary
print("Building vocabulary...")
indices = np.random.choice(len(x_train), N_SAMPLES)
features = np.array([extract_features(x_train[i]) for i in indices])
print("Extracted features from training set.")
pca = PCA(n_components=256).fit(features)
kmeans = KMeans(n_clusters=N_CLUSTERS).fit(pca.transform(features))

# Create histogram for an image
def get_hist(img):
    feat_pca = pca.transform(extract_features(img).reshape(1, -1))
    pred = kmeans.predict(feat_pca)
    hist = np.bincount(pred, minlength=N_CLUSTERS)
    return hist / max(hist.sum(), 1e-10)

# Process test set
print("Processing test set...")
test_hists = np.array([get_hist(img) for img in x_test])

# Retrieve similar images
def retrieve(q_img, q_idx=None, n=5):
    q_hist = get_hist(q_img)
    # Chi-square distance
    dist = 0.5 * np.sum(((q_hist - test_hists) ** 2) / (q_hist + test_hists + 1e-10), axis=1)
    idx = np.argsort(dist)
    if q_idx is not None:
        idx = idx[idx != q_idx]
    return idx[:n]

# Evaluate retrieval performance
def evaluate(queries=100, top_k=5):
    mAP = []
    for idx in np.random.choice(len(x_test), queries):
        ret_idx = retrieve(x_test[idx], idx, top_k)
        relevant = (y_test == y_test[idx])
        ap = average_precision_score(relevant[ret_idx], np.arange(len(ret_idx), 0, -1)) if np.sum(relevant[ret_idx]) > 0 else 0
        mAP.append(ap)
    return np.mean(mAP)

# Run evaluation
print(f"Mean Average Precision: {evaluate():.4f}")

