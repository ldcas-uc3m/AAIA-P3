import os
from pathlib import Path
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def read_articles(base_folder):
    """Read all articles from the folder structure and return them with their true categories"""
    articles = []
    true_labels = []
    file_paths = []
    
    categories = ["business", "entertainment", "politics", "sports", "tech"]
    
    for category in categories:
        category_path = Path(base_folder) / category
        if not category_path.exists():
            print(f"Warning: Category folder {category_path} not found")
            continue
            
        for file_path in category_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                articles.append(content)
                true_labels.append(category)
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return articles, true_labels, file_paths

def preprocess_text(articles):
    return [re.sub(r'\W+', ' ', article.lower()) for article in articles]

def perform_clustering(base_folder, n_clusters=5, max_features=5000, random_state=100451001):
    """Perform clustering on articles using TF-IDF and KMeans"""
    start_time = time.time()
    
    print(f"Reading articles from {base_folder}...")
    articles, true_labels, file_paths = read_articles(base_folder)
    
    if not articles:
        print("No articles found to cluster.")
        return
    
    print(f"Found {len(articles)} articles. Starting clustering...")
    
    # Preprocess the articles
    processed_articles = preprocess_text(articles)
    
    # Convert text to TF-IDF features
    print("Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(processed_articles)
    
    # Apply KMeans clustering
    print(f"Applying KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Create a mapping from true category labels to numeric labels for evaluation
    unique_categories = sorted(list(set(true_labels)))
    category_to_id = {category: i for i, category in enumerate(unique_categories)}
    true_label_ids = [category_to_id[label] for label in true_labels]
    print(f"True categories: {unique_categories}")
    print(f"Cluster labels: {sorted(set(cluster_labels))}")    
    # Visualize clusters using dimensionality reduction
    visualize_clusters(X, cluster_labels, true_label_ids, unique_categories)
    
    # Analyze clusters and extract key terms
    analyze_clusters(cluster_labels, articles, vectorizer, n_clusters)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

def visualize_clusters(X, cluster_labels, true_label_ids, category_names):
    """Visualize clusters using PCA and t-SNE"""
    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    X_dense = X.toarray()  # Convert sparse matrix to dense
    X_pca = pca.fit_transform(X_dense)
    
    # Plot PCA results colored by cluster
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    ax1.set_title('PCA visualization of clusters')
    
    # Plot PCA results colored by true category
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_label_ids, cmap='viridis', alpha=0.5)
    ax2.set_title('PCA visualization of true categories')
    
    # Add a colorbar legend
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Cluster')
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('True Category')
    cbar2.set_ticks(range(len(category_names)))
    cbar2.set_ticklabels(category_names)
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')
    print("Visualization saved as 'cluster_visualization.png'")

def analyze_clusters(cluster_labels, articles, vectorizer, n_clusters, top_n_terms=10):
    """Extract and display the top terms for each cluster"""
    print("\n--- CLUSTER ANALYSIS ---")
    
    # Get feature names
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get cluster centers (average TF-IDF score for each term in each cluster)
    if hasattr(vectorizer, 'transform'):
        X = vectorizer.transform(articles)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        
        # For each cluster, find the top terms
        for i in range(n_clusters):
            print(f"\nCluster {i+1} top terms:")
            # Sort terms by importance in cluster
            sorted_indices = centers[i].argsort()[::-1]
            top_terms = feature_names[sorted_indices[:top_n_terms]]
            print(", ".join(top_terms))
            
            # Count documents in cluster
            cluster_docs = [articles[j] for j, label in enumerate(cluster_labels) if label == i]
            print(f"Number of documents in cluster: {len(cluster_docs)}")

if __name__ == "__main__":
    perform_clustering("./bbc-test")
