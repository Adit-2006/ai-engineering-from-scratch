import cupy as cp
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = cp.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = cp.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrix)

        sorted_idx = cp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = cp.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return (X_reduced @ self.components) + self.mean

cp.random.seed(42)
n_samples = 500

t = cp.random.uniform(0, 2 * cp.pi, n_samples)
x1 = 3 * cp.cos(t) + cp.random.normal(0, 0.2, n_samples)
x2 = 3 * cp.sin(t) + cp.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + cp.random.normal(0, 0.1, n_samples)

X_synthetic = cp.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synthetic)

print(f"Original shape: {X_synthetic.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")
print()

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist_cpu = mnist.data[:5000].astype(float)

X_mnist_gpu = cp.array(X_mnist_cpu)
for i in [10,50,200]:
    princi = PCA(n_components=i)
    x_redu = princi.fit_transform(X_mnist_gpu)
    x_new = princi.inverse_transform(x_redu)
    mse = cp.mean((X_mnist_gpu - x_new) ** 2)
    print("mse ",i," ", mse)
    
print()
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=150)
X_pca50 = pca_mnist.fit_transform(X_mnist_gpu)
print(f"150 components capture {sum(pca_mnist.explained_variance_ratio_):.2%} of variance")

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist_gpu)
X_pca2d_cpu = X_pca2d.get()
print(f"2 components capture {sum(pca_2d.explained_variance_ratio_):.2%} of variance")
print()


from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist_cpu)

print(f"\nOur PCA explained variance:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d_cpu) - np.abs(X_sklearn_pca))
print(f"Max absolute difference: {diff.max():.10f}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist_cpu)
print(f"\nt-SNE output shape: {X_tsne.shape}")
print()

