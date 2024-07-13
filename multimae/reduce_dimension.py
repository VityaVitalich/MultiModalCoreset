import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def reduce_dimensionality(embeddings, method, n_components):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Method should be 'pca', 'tsne', or 'umap'")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def main():
    parser = argparse.ArgumentParser(description='Reduce dimensionality of embeddings.')
    parser.add_argument('embed_path', type=str, help='Path to the embeddings file')
    parser.add_argument('method', type=str, choices=['pca', 'tsne', 'umap'], help='Dimensionality reduction method to use')
    parser.add_argument('n_components', type=int, help='Number of components for reduction')

    args = parser.parse_args()

    embeddings_original = np.load(args.embed_path)
    reduced_embeddings = reduce_dimensionality(embeddings_original, args.method, args.n_components)
    
    output_path = args.embed_path.replace('.npy', f'_{args.method}_{args.n_components}d.npy')
    np.save(output_path, reduced_embeddings)
    print(f'Reduced embeddings saved to {output_path}')

if __name__ == "__main__":
    main()
