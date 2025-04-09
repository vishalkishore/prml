import numpy as np
import cv2
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture



tf.config.optimizer.set_jit(True)

device = tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')
print(f"Using device: {device}")
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')


# Feature extraction functions remain the same, with batch processing capability
def extract_sift_batch(images, batch_size=32):
    """Extract SIFT features from a batch of images"""
    all_descriptors = []
    
    for image in images:
        # Convert to uint8 and ensure proper format for SIFT
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Extract SIFT features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None:
            all_descriptors.append(descriptors)
        else:
            all_descriptors.append(np.zeros((1, 128)))
    
    return all_descriptors

def extract_cnn_batch(images, batch_size=32):
    """Extract CNN features from a batch of images using EfficientNetB0"""
    
    # Preprocess all images
    preprocessed_images = []
    with tf.device('/GPU:0'):
        for image in images:
            image_resized = cv2.resize(image, (224, 224))
            image_preprocessed = preprocess_input(image_resized.astype(np.float32))
            preprocessed_images.append(image_preprocessed)
    
    # Process in mini-batches to avoid memory issues
    features = []
    for i in range(0, len(images), batch_size):
        batch = np.array(preprocessed_images[i:i+batch_size])
        batch_features = model.predict(batch, batch_size=batch_size)
        features.append(batch_features)
    
    # Combine all features
    if features:
        features = np.vstack(features)
    
    return features

class VLAD:
    def __init__(self, feature_extractor='sift', clustering_method='kmeans'):
        self.n_clusters = 40  
        self.pca_components = 140
        self.cluster_model = None
        self.pca = None
        self.feature_extractor = feature_extractor
        self.clustering_method = clustering_method
        self.image_ids = []
        self.image_labels = {}
        self.centroids = None
        
        if feature_extractor == 'sift':
            self.extract_features = extract_sift_batch
        elif feature_extractor == 'cnn':
            self.extract_features = extract_cnn_batch
        else:
            raise ValueError("Unsupported feature extractor")

    def build_codebook(self, images, labels, image_ids=None, batch_size=32):
        """Build codebook with automatic hyperparameter tuning"""
        if image_ids is None:
            image_ids = np.arange(len(images))
        
        self.image_ids = list(image_ids)

        # Set default values if not tuned
        if self.n_clusters is None:
            self.n_clusters = 64
            print(f"Using default n_clusters={self.n_clusters}")
        
        if self.pca_components is None:
            self.pca_components = 64
            print(f"Using default pca_components={self.pca_components}")
        
        print(f"Building VLAD codebook with {self.pca_components} PCA components and {self.n_clusters} clusters...")
        print(f"Clustering method: {self.clustering_method}")
        
        # Extract features in batches
        all_descriptors = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_descriptors = self.extract_features(batch_images, batch_size)
            
            if self.feature_extractor == 'cnn':
                all_descriptors.extend(batch_descriptors)
            else:
                for descriptors in batch_descriptors:
                    if descriptors is not None and descriptors.shape[0] > 0:
                        all_descriptors.extend(descriptors)
        
        all_descriptors = np.array(all_descriptors)
        
        # Apply PCA
        self.pca = PCA(n_components=self.pca_components)
        reduced_descriptors = self.pca.fit_transform(all_descriptors)
        
        # Initialize and fit clustering model based on specified method
        if self.clustering_method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=1)
            self.cluster_model.fit(reduced_descriptors)
            self.centroids = self.cluster_model.cluster_centers_
        elif self.clustering_method == 'gmm':
            self.cluster_model = GaussianMixture(n_components=self.n_clusters, random_state=42, n_init=1)
            self.cluster_model.fit(reduced_descriptors)
            self.centroids = self.cluster_model.means_
        elif self.clustering_method == 'minibatch_kmeans':
            self.cluster_model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=256)
            self.cluster_model.fit(reduced_descriptors)
            self.centroids = self.cluster_model.cluster_centers_
        elif self.clustering_method == 'spectral':
            # For large datasets, spectral clustering might be slow, so we sample
            max_samples = min(10000, len(reduced_descriptors))
            sample_indices = np.random.choice(len(reduced_descriptors), max_samples, replace=False)
            sampled_descriptors = reduced_descriptors[sample_indices]
            
            self.cluster_model = SpectralClustering(n_clusters=self.n_clusters, random_state=42, 
                                                   affinity='nearest_neighbors', n_neighbors=10)
            cluster_labels = self.cluster_model.fit_predict(sampled_descriptors)
            
            # Compute centroids for each cluster
            self.centroids = np.array([sampled_descriptors[cluster_labels == i].mean(axis=0) 
                                      for i in range(self.n_clusters) if np.sum(cluster_labels == i) > 0])
            
            # If some clusters are empty, adjust n_clusters
            if len(self.centroids) < self.n_clusters:
                print(f"Warning: Only {len(self.centroids)} non-empty clusters found.")
                self.n_clusters = len(self.centroids)
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        # Store the prediction function based on clustering method
        if self.clustering_method in ['kmeans', 'minibatch_kmeans']:
            self.predict_func = self.cluster_model.predict
        elif self.clustering_method == 'gmm':
            self.predict_func = self.cluster_model.predict
        elif self.clustering_method == 'spectral':
            # For spectral clustering, use nearest centroids for prediction
            def nearest_centroid_predict(X):
                distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
                return np.argmin(distances, axis=1)
            self.predict_func = nearest_centroid_predict
        
        # Compute VLAD features and build index
        features = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_ids = image_ids[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_features = self._compute_vlad_features_batch(batch_images, batch_size)
            features.append(batch_features)
            
            # Store labels for evaluation
            for j, image_id in enumerate(batch_ids):
                self.image_labels[image_id] = batch_labels[j]
        
        # Combine all features
        features = np.vstack(features)
        
        # Build KNN index
        self.nn_index = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
        self.nn_index.fit(features)
        self.features = features  # Store for later use
        
        print(f"VLAD codebook built successfully with {len(features)} images indexed")

    def _compute_vlad_features_batch(self, images, batch_size=32):
        """Compute VLAD features for a batch of images"""
        batch_descriptors = self.extract_features(images, batch_size)
        batch_vlads = []
        
        # Process according to feature extractor type
        if self.feature_extractor == 'cnn':
            for descriptors in batch_descriptors:
                # Apply PCA
                descriptors = self.pca.transform(descriptors.reshape(1, -1))
                
                # Initialize VLAD vector
                vlad = np.zeros((self.n_clusters, descriptors.shape[1]))
                
                # Get cluster assignments
                assignments = self.predict_func(descriptors)
                
                # Calculate VLAD
                for i in range(self.n_clusters):
                    # Get descriptors assigned to cluster i
                    assigned_descriptors = descriptors[assignments == i]
                    
                    if len(assigned_descriptors) > 0:
                        # Sum of differences between descriptors and centroid
                        vlad[i] = np.sum(assigned_descriptors - self.centroids[i], axis=0)
                
                # Flatten and normalize
                vlad = vlad.flatten()
                
                # Power normalization
                vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
                
                # L2 normalization
                vlad_norm = np.linalg.norm(vlad)
                if vlad_norm > 0:
                    vlad = vlad / vlad_norm
                
                batch_vlads.append(vlad)
        else:
            # For local feature extractors (SIFT, etc.)
            for descriptors in batch_descriptors:
                if descriptors is None or descriptors.shape[0] == 0:
                    # Create empty VLAD vector with correct dimensions
                    batch_vlads.append(np.zeros(self.n_clusters * self.pca_components))
                    continue
                
                # Apply PCA
                descriptors = self.pca.transform(descriptors)
                
                # Initialize VLAD vector
                vlad = np.zeros((self.n_clusters, descriptors.shape[1]))
                
                # Get cluster assignments
                assignments = self.predict_func(descriptors)
                
                # Calculate VLAD
                for i in range(self.n_clusters):
                    # Get descriptors assigned to cluster i
                    assigned_descriptors = descriptors[assignments == i]
                    
                    if len(assigned_descriptors) > 0:
                        # Sum of differences between descriptors and centroid
                        vlad[i] = np.sum(assigned_descriptors - self.centroids[i], axis=0)
                
                # Flatten and normalize
                vlad = vlad.flatten()
                
                # Power normalization
                vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
                
                # L2 normalization
                vlad_norm = np.linalg.norm(vlad)
                if vlad_norm > 0:
                    vlad = vlad / vlad_norm
                
                batch_vlads.append(vlad)
        
        return np.array(batch_vlads)


    def query(self, query_images, top_k=5, batch_size=32):
        """Query the system with batch of images and return top_k matches"""
        results = []
        
        # Process query images in batches
        for i in range(0, len(query_images), batch_size):
            batch_images = query_images[i:i+batch_size]
            
            # Compute features for the batch
            batch_features = self._compute_vlad_features_batch(batch_images, batch_size)
            
            # Search using KNN index
            distances, indices = self.nn_index.kneighbors(batch_features, n_neighbors=top_k)
            
            # Map indices to image IDs
            for batch_indices in indices:
                result_ids = [self.image_ids[idx] for idx in batch_indices if idx < len(self.image_ids)]
                results.append(result_ids)
        
        return results
    
    def evaluate(self, query_images, query_labels, top_k=5, batch_size=32):
        """Evaluate the retrieval system using precision and recall"""
        precisions = []
        recalls = []
        
        results = self.query(query_images, top_k, batch_size)
        
        for i, (result_ids, query_label) in enumerate(zip(results, query_labels)):
            retrieved_labels = [self.image_labels[image_id] for image_id in result_ids]
            
            relevant_retrieved = sum(1 for label in retrieved_labels if label == query_label)
            
            precision = relevant_retrieved / len(result_ids)
            precisions.append(precision)
            
            total_relevant = sum(1 for label in self.image_labels.values() if label == query_label)
            
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            recalls.append(recall)
        
        return {
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls)
        }


from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
y_train = y_train.flatten()
y_test = y_test.flatten()

train_ids = np.arange(len(x_train))
test_ids = np.arange(len(x_test))

train_dir = "/home/redix/Desktop/prml/data/imagenette2-320/train"
val_dir = "/home/redix/Desktop/prml/data/imagenette2-320/val"

# Parameters
img_size = (128, 128)
batch_size = 32
subset_size = 30000 
eval_size = 100

# # Load datasets
# from tensorflow.keras.utils import image_dataset_from_directory
# train_ds = image_dataset_from_directory(train_dir, image_size=img_size, batch_size=batch_size, shuffle=False)
# val_ds = image_dataset_from_directory(val_dir, image_size=img_size, batch_size=batch_size, shuffle=False)

# # Convert datasets to numpy arrays
# def dataset_to_numpy(dataset):
#     images = []
#     labels = []
#     for batch in dataset:
#         imgs, lbls = batch
#         images.append(imgs.numpy())
#         labels.append(lbls.numpy())
#     return np.concatenate(images), np.concatenate(labels)

# x_train, y_train = dataset_to_numpy(train_ds)
# x_test, y_test = dataset_to_numpy(val_ds)

# # Normalize pixel values to float32
# x_train = x_train.astype('float32') 
# x_test = x_test.astype('float32')

# # Flatten labels if needed
# y_train = y_train.flatten()
# y_test = y_test.flatten()

# train_ids = np.arange(len(x_train))
# test_ids = np.arange(len(x_test))


vlad = VLAD(feature_extractor='cnn',clustering_method='spectral')
vlad.build_codebook(x_train[:subset_size], y_train[:subset_size], image_ids=train_ids[:subset_size], batch_size=batch_size)
vlad_results = vlad.evaluate(x_test[:eval_size], y_test[:eval_size], top_k=5, batch_size=batch_size)

print("VLAD Evaluation Results:",vlad_results)
