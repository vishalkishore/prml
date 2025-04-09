import numpy as np
import cv2
import joblib
import faiss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
batch_size = 64

def extract_cnn_features(images, batch_size=64):
    """Extract CNN features from images using EfficientNetB3"""
    features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        batch_resized = np.array([cv2.resize(img, (224, 224)) for img in batch])
        # Preprocess images
        batch_preprocessed = preprocess_input(batch_resized)
        # Extract features
        batch_features = model.predict(batch_preprocessed, verbose=0)
        features.append(batch_features)
    
    return np.vstack(features)


class ImageRetriever:
    def __init__(self, n_clusters=40, pca_components=256):
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.kmeans = None
        self.pca = None
        self.faiss_index = None
        self.features = None
        self.image_ids = None
        self.labels = None
        self.batch_size = 64
        
    def fit(self, images, labels=None):
        """Build the retrieval model with KMeans clustering on CNN features"""
        # Extract CNN features
        print("Extracting CNN features...")

        features = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_descriptors = extract_cnn_features(batch_images, batch_size)
            features.extend(batch_descriptors)
        features = np.array(features)
        # Apply PCA for dimensionality reduction
        print(f"Applying PCA with {self.pca_components} components...")
        self.pca = PCA(n_components=self.pca_components)
        reduced_features = self.pca.fit_transform(features)
        
        # Apply KMeans clustering
        print(f"Applying KMeans with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=SEED)
        clusters = self.kmeans.fit_predict(reduced_features)
        
        # Create histogram features
        print("Creating histogram features...")
        self.features = np.zeros((len(images), self.n_clusters))
        for i, cluster in enumerate(clusters):
            self.features[i, cluster] += 1
        
        # Normalize histograms
        row_sums = self.features.sum(axis=1)
        self.features = self.features / row_sums[:, np.newaxis]
        
        # Build FAISS index for fast similarity search
        print("Building FAISS index...")
        d = self.features.shape[1]  # Dimension of feature vectors
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(self.features.astype('float32'))
        
        # Store image IDs and labels
        self.image_ids = np.arange(len(images))
        self.labels = labels
        
        return self
    
    def process_query(self, query_image):
        """Process a query image to get its feature histogram"""
        # Handle both single image and batch
        is_batch = len(query_image.shape) == 4
        query_images = query_image if is_batch else np.expand_dims(query_image, axis=0)
        
        # Extract features
        query_features = extract_cnn_features(query_images)
        query_reduced = self.pca.transform(query_features)
        query_clusters = self.kmeans.predict(query_reduced)
        
        # Create histogram
        query_hist = np.zeros((len(query_images), self.n_clusters))
        for i, cluster in enumerate(query_clusters):
            query_hist[i, cluster] += 1
        
        # Normalize
        row_sums = query_hist.sum(axis=1)
        query_hist = query_hist / row_sums[:, np.newaxis]
        
        return query_hist
    
    def query(self, query_image, top_k=5):
        """Query the index with an image and return top_k matches"""
        query_hist = self.process_query(query_image)
        
        # Search using FAISS
        distances, indices = self.faiss_index.search(
            query_hist.astype('float32'), top_k
        )
        
        # Map indices to original image IDs
        result_ids = [[int(self.image_ids[idx]) for idx in row] for row in indices]
        
        return result_ids, distances

    @classmethod
    def load(cls, filepath):
        """Load model from joblib file"""
        data = joblib.load(filepath)
        
        # Create instance
        instance = cls(n_clusters=data['n_clusters'], pca_components=data['pca_components'])
        
        # Load components
        instance.pca = data['pca']
        instance.kmeans = data['kmeans']
        instance.features = data['features']
        instance.image_ids = data['image_ids']
        instance.labels = data['labels']
        
        # Deserialize FAISS index
        instance.faiss_index = faiss.deserialize_index(data['faiss_bytes'])
        
        print(f"Model loaded from {filepath}")
        return instance

def inference_pipeline(query_image, model_path=None, retriever=None, all_images=None, top_k=5):
    if retriever is None and model_path is not None:
        retriever = ImageRetriever.load(model_path)
    
    if retriever is None:
        raise ValueError("Either retriever or model_path must be provided")
    
    result_ids, distances = retriever.query(query_image, top_k=top_k)
    if all_images is not None:
        retriever.plot_results(query_image, result_ids, distances, all_images)
    
    return result_ids, distances

