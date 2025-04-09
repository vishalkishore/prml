import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
import time
from sklearn.neighbors import NearestNeighbors

tf.config.optimizer.set_jit(True)

device = tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')
print(f"Using device: {device}")
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')


def extract_sift_batch(images, batch_size=32):
    all_descriptors = []
    
    for image in images:
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None:
            all_descriptors.append(descriptors)
        else:
            all_descriptors.append(np.zeros((1, 128)))
    
    return all_descriptors

def extract_cnn_batch(images, batch_size=32):
    preprocessed_images = []
    with tf.device('/GPU:0'):
        for image in images:
            image_resized = cv2.resize(image, (224, 224))
            image_preprocessed = preprocess_input(image_resized.astype(np.float32))
            preprocessed_images.append(image_preprocessed)
    
    features = []
    for i in range(0, len(images), batch_size):
        batch = np.array(preprocessed_images[i:i+batch_size])
        batch_features = model.predict(batch, batch_size=batch_size)
        features.append(batch_features)
    
    # Combine all features
    if features:
        features = np.vstack(features)
    
    return features

class BoVW:
    def __init__(self, feature_extractor='sift'):
        self.n_clusters = 120  # Will be determined by hyperparameter tuning
        self.pca_components = 140  # Will be determined by hyperparameter tuning
        self.kmeans = None
        self.pca = None
        self.feature_extractor = feature_extractor
        self.image_ids = []
        self.image_labels = {}
        
        # Select feature extractor
        if feature_extractor == 'sift':
            self.extract_features = extract_sift_batch
        elif feature_extractor == 'cnn':
            self.extract_features = extract_cnn_batch
        else:
            raise ValueError("Unsupported feature extractor")
    
    def tune_hyperparameters(self, images, max_pca_components=200, max_clusters=300, sample_size=3000, batch_size=32):
        """Tune both PCA components and number of clusters automatically"""
        print("Starting hyperparameter tuning...")
        start_time = time.time()
        
        # Sample images for faster tuning
        if len(images) > sample_size:
            indices = np.random.choice(len(images), sample_size, replace=False)
            tuning_images = images[indices]
        else:
            tuning_images = images
        
        # Extract features for tuning
        all_descriptors = []
        for i in range(0, len(tuning_images), batch_size):
            batch_images = tuning_images[i:i+batch_size]
            batch_descriptors = self.extract_features(batch_images, batch_size)
            
            if self.feature_extractor == 'cnn':
                all_descriptors.extend(batch_descriptors)
            else:
                for descriptors in batch_descriptors:
                    if descriptors is not None and descriptors.shape[0] > 0:
                        if len(all_descriptors) < 50000:  # Limit number of descriptors for memory
                            all_descriptors.extend(descriptors)
        
        all_descriptors = np.array(all_descriptors)
        print(f"Extracted {len(all_descriptors)} descriptors for tuning")
        
        # Tune PCA components first
        self.pca_components = self._tune_pca(all_descriptors, max_components=min(max_pca_components, all_descriptors.shape[1]-1))
        
        # Apply PCA with optimal components
        pca = PCA(n_components=self.pca_components)
        reduced_descriptors = pca.fit_transform(all_descriptors)
        
        # Tune number of clusters
        self.n_clusters = self._tune_kmeans(reduced_descriptors, max_clusters=max_clusters)
        
        print(f"Optimal parameters - PCA Components: {self.pca_components}, K-means Clusters: {self.n_clusters}")
        print(f"Tuning completed in {time.time() - start_time:.2f} seconds")
        
        return {"pca_components": self.pca_components, "n_clusters": self.n_clusters}
    
    def _tune_pca(self, descriptors, max_components=200, n_steps=10):
        """Find optimal number of PCA components using explained variance"""
        print("Tuning PCA components...")
        
        # Calculate steps based on input dimensions and max_components
        dimension = min(descriptors.shape[1], max_components)
        step_size = max(1, dimension // n_steps)
        components_to_try = list(range(step_size, dimension + 1, step_size))
        
        # Ensure we include max_components in the list
        if max_components not in components_to_try:
            components_to_try.append(max_components)
        components_to_try.sort()
        
        # Calculate explained variance for different numbers of components
        explained_variance = []
        for n_components in components_to_try:
            pca = PCA(n_components=n_components)
            pca.fit(descriptors)
            # Calculate cumulative explained variance
            explained_variance.append(sum(pca.explained_variance_ratio_))
            print(f"  Components: {n_components}, Explained Variance: {explained_variance[-1]:.4f}")
        
        # Find optimal number of components using KneeLocator
        kneedle = KneeLocator(components_to_try, explained_variance, S=1.0, curve="concave", direction="increasing")
        optimal_components = kneedle.knee
        
        # If knee point detection fails, use the point where variance reaches 95%
        if optimal_components is None:
            optimal_components = next((n for i, n in enumerate(components_to_try) 
                                       if explained_variance[i] >= 0.95), components_to_try[-1])
        
        # Visualize the results
        plt.figure(figsize=(10, 6))
        plt.plot(components_to_try, explained_variance, 'b-', linewidth=2)
        plt.axvline(x=optimal_components, color='r', linestyle='--')
        plt.xlabel('Number of PCA Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Components Tuning')
        plt.grid(True)
        plt.savefig('pca_tuning.png')
        plt.close()
        
        print(f"Optimal number of PCA components: {optimal_components}")
        return optimal_components
    
    def _tune_kmeans(self, descriptors, max_clusters=300, n_steps=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("Tuning K-means clusters...")
        
        # Limit number of descriptors for faster tuning
        if len(descriptors) > 10000:
            indices = np.random.choice(len(descriptors), 10000, replace=False)
            tuning_descriptors = descriptors[indices]
        else:
            tuning_descriptors = descriptors
        
        # Calculate steps
        step_size = max(1, max_clusters // n_steps)
        clusters_to_try = list(range(step_size, max_clusters + 1, step_size))
        
        # Ensure we have at least 5 points for knee detection
        if len(clusters_to_try) < 5:
            clusters_to_try = list(range(max(2, max_clusters//5), max_clusters + 1, max_clusters//5))
        
        inertia = []
        silhouette = []
        
        for k in clusters_to_try:
            print(f"  Testing k={k}")
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=1)
            kmeans.fit(tuning_descriptors)
            inertia.append(kmeans.inertia_)
            
            # Calculate silhouette score for smaller datasets only (it's computationally expensive)
            if len(tuning_descriptors) <= 5000:
                labels = kmeans.predict(tuning_descriptors)
                s_score = silhouette_score(tuning_descriptors, labels) if len(np.unique(labels)) > 1 else 0
                silhouette.append(s_score)
        
        # Find optimal k using elbow method
        kneedle = KneeLocator(clusters_to_try, inertia, S=1.0, curve="convex", direction="decreasing")
        optimal_k = kneedle.knee
        
        # If knee detection fails, use default or silhouette score
        if optimal_k is None:
            if len(silhouette) > 0:
                optimal_k = clusters_to_try[np.argmax(silhouette)]
            else:
                optimal_k = min(100, max_clusters)  # Default fallback
        
        # Visualize the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Inertia plot (Elbow method)
        ax1.plot(clusters_to_try, inertia, 'b-', linewidth=2)
        ax1.axvline(x=optimal_k, color='r', linestyle='--')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for K-means')
        ax1.grid(True)
        
        # Silhouette plot (if available)
        if len(silhouette) > 0:
            ax2.plot(clusters_to_try, silhouette, 'g-', linewidth=2)
            ax2.axvline(x=optimal_k, color='r', linestyle='--')
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Silhouette analysis skipped\n(dataset too large)", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig('kmeans_tuning.png')
        plt.close()
        
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def build_codebook(self, images, labels, image_ids=None, batch_size=32, auto_tune=True):
        """Build codebook with automatic hyperparameter tuning"""
        if image_ids is None:
            image_ids = np.arange(len(images))
        
        self.image_ids = list(image_ids)
        
        # Automatically tune hyperparameters if requested
        if auto_tune and (self.n_clusters is None or self.pca_components is None):
            self.tune_hyperparameters(images, batch_size=batch_size)
        
        # Set default values if not tuned
        if self.n_clusters is None:
            self.n_clusters = 100
            print(f"Using default n_clusters={self.n_clusters}")
        
        if self.pca_components is None:
            self.pca_components = 64
            print(f"Using default pca_components={self.pca_components}")
        
        print(f"Building codebook with {self.pca_components} PCA components and {self.n_clusters} clusters...")
        
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
        
        # Apply K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=1)
        self.kmeans.fit(reduced_descriptors)
        
        # Compute histogram features and build index
        features = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_ids = image_ids[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_features = self._compute_histogram_features_batch(batch_images, batch_size)
            features.append(batch_features)
            
            # Store labels for evaluation
            for j, image_id in enumerate(batch_ids):
                self.image_labels[image_id] = batch_labels[j]
        
        # Combine all features
        features = np.vstack(features)
        
        # Build KNN index for nearest neighbor search
        self.nn_index = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
        self.nn_index.fit(features)
        self.features = features  # Store for later use
        
        print(f"Codebook built successfully with {len(features)} images indexed")

    def _compute_histogram_features_batch(self, images, batch_size=32):
        """Compute histogram of visual words for a batch of images"""
        batch_descriptors = self.extract_features(images, batch_size)
        batch_histograms = []
        
        # Process according to feature extractor type
        if self.feature_extractor == 'cnn':
            # For CNN features, directly cluster the features
            for descriptors in batch_descriptors:
                # Apply PCA
                descriptors = self.pca.transform(descriptors.reshape(1, -1))
                
                # Get cluster assignment
                cluster_assignments = self.kmeans.predict(descriptors)
                
                # Create histogram
                histogram = np.zeros(self.n_clusters)
                for cluster_id in cluster_assignments:
                    histogram[cluster_id] += 1
                
                # Normalize
                if np.sum(histogram) > 0:
                    histogram = histogram / np.sum(histogram)
                
                batch_histograms.append(histogram)
        else:
            # For local feature extractors (SIFT, etc.)
            for descriptors in batch_descriptors:
                if descriptors is None or descriptors.shape[0] == 0:
                    batch_histograms.append(np.zeros(self.n_clusters))
                    continue
                
                # Apply PCA
                descriptors = self.pca.transform(descriptors)
                
                # Get cluster assignments
                cluster_assignments = self.kmeans.predict(descriptors)
                
                # Create histogram
                histogram = np.zeros(self.n_clusters)
                for cluster_id in cluster_assignments:
                    histogram[cluster_id] += 1
                
                # Normalize
                if np.sum(histogram) > 0:
                    histogram = histogram / np.sum(histogram)
                
                batch_histograms.append(histogram)
        
        return np.array(batch_histograms)
    
    def query(self, query_images, top_k=5, batch_size=32):
        """Query the system with batch of images and return top_k matches"""
        results = []
        
        # Process query images in batches
        for i in range(0, len(query_images), batch_size):
            batch_images = query_images[i:i+batch_size]
            
            # Compute features for the batch
            batch_features = self._compute_histogram_features_batch(batch_images, batch_size)
            
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



bovw = BoVW(feature_extractor='cnn')
bovw.build_codebook(x_train[:subset_size], y_train[:subset_size], image_ids=train_ids[:subset_size], batch_size=batch_size,auto_tune=False)

bovw_results = bovw.evaluate(x_test[:eval_size], y_test[:eval_size], top_k=5, batch_size=batch_size)

print("BoVW Evaluation Results:",bovw_results)
