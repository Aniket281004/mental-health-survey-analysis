import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
def build_pipeline(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_trans=Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())
    ])
    cat_trans=Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        (
            'onehot',OneHotEncoder(handle_unknown='ignore')
        )
    ])
    preprocessor=ColumnTransformer([
        ('nums',num_trans,num_cols),
        ('cat',cat_trans,cat_cols)
    ])
    return preprocessor
df=pd.read_csv('clean_data.csv')
X=df
X_proc = build_pipeline(X).fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_proc)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, )
X_tsne = tsne.fit_transform(X_pca)
models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
   
}
cluster_assignments = {}
results = {}
silhouette_scores = {}
for name, model in models.items():
    if name == "GMM":
        labels = model.fit_predict(X_pca)
    else:
        labels = model.fit_predict(X_pca)
    results[name] = labels
    cluster_assignments[name] = labels
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(X_pca, labels)
        print(f"{name} Silhouette Score: {score:.5f}")
    else:
        print("Silhouette score not applicable — only one cluster or noise.")
    silhouette_scores[name] = score
    
    plt.figure(figsize=(5, 4))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(f"{name} Clusters (t-SNE view)")
    plt.show()
