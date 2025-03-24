#Importing Required Libraries------------------------------------------------------------------------------------------------------------------------------
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import cv2
from skimage.feature import hog
from tqdm import tqdm
#------------------------------------------------------------------------------------------------------------------------------------------------------------


# Constants Defined------------------------------------------------------------------------------------------------------------------------------------------
IMG_HEIGHT = 28
IMG_WIDTH = 28
#------------------------------------------------------------------------------------------------------------------------------------------------------------


# Function to preprocess image: resize & convert to grayscale------------------------------------------------------------------------------------------------
def preprocess_image(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).numpy() * 255  # Convert from Tensor (C,H,W) to numpy (H,W,C)
    img_np = img_np.astype(np.uint8)
    img_resized = cv2.resize(img_np, (IMG_WIDTH, IMG_HEIGHT))
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    return gray_img
#------------------------------------------------------------------------------------------------------------------------------------------------------------



# Loading Dataset and Extracting HOG Features----------------------------------------------------------------------------------------------------------------
def extract_hog_features():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    hog_features_all = []
    for img_tensor, _ in tqdm(dataset, desc="Extracting HOG features"):
        gray_img = preprocess_image(img_tensor)
        
        # Extract HOG feature vector
        fd = hog(
            gray_img,
            orientations=10,
            pixels_per_cell=(5, 5),
            cells_per_block=(1, 1),
            visualize=False,
            block_norm='L2-Hys'
        )
        
        hog_features_all.append(fd)
    
    hog_features_array = np.array(hog_features_all)
    return hog_features_array
#------------------------------------------------------------------------------------------------------------------------------------------------------------



# Call the function and store the features-------------------------------------------------------------------------------------------------------------------
hog_features = extract_hog_features()
#------------------------------------------------------------------------------------------------------------------------------------------------------------


print(hog_features[0])
print(f"✅ HOG feature extraction completed. Shape of feature array: {hog_features.shape}")
