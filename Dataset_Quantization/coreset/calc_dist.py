import numpy as np

if __name__ == '__main__':
    embed_path = '/home/cache/data/dq/clevr/train/multimae_embeds.npy'
    embeddings = np.load(embed_path)[:20000, ]
    sim_matrix = embeddings @ embeddings.T
    np.save('/home/cache/data/dq/clevr/train/sim_matrix.npy', sim_matrix)
