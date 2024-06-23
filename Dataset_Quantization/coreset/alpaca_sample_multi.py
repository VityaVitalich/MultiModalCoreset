import argparse

import numpy as np
import util.utils as utils
from dq.methods.methods_utils.submodular_function import GraphCut
from dq.methods.methods_utils.submodular_optimizer import NaiveGreedy
import pickle


def cossim_np_my(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    res = num / denom
    if np.isneginf(res):
        res = 0.0
    return 0.5 + 0.5 * res

def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.0
    return 0.5 + 0.5 * res

def random_sample(data, n=1000):
    indices = np.random.choice(len(data), n, replace=False)
    return indices


def dataset_quantization(embed_path, ratio=0.02, k=50):
    embeddings_original = np.load(embed_path)
    embeddings = embeddings_original.copy()

    total_objects = embeddings_original.shape[0]
    n = int(total_objects * ratio)
    bins_n = int(n / k)
    budget_n = total_objects // k
    print(f"total: {total_objects} n: {n}, k: {k}, budget_n: {budget_n}, bins_n: {bins_n}")

    
    indices_original = np.arange(total_objects)
    indices = indices_original.copy()

    def sim_matrix(a, b):
        return cossim_np(embeddings[a], embeddings[b])

    # bin generation
    bins = []
    for i in range(k):
        print(f"bin {i}/{k}")
        submod_f = GraphCut(index=indices, similarity_kernel=sim_matrix)
        submod_opt = NaiveGreedy(args=None, index=indices, budget=budget_n)
        result_indices = submod_opt.select(
            gain_function=submod_f.calc_gain,
            update_state=submod_f.update_state,
        )

        bins.append(result_indices)
        indices = np.delete(indices_original, np.concatenate(bins))
        embeddings = np.delete(embeddings_original, np.concatenate(bins), axis=0)

    # bin sampling
    index = []
    assert len(bins) == k
    for i in range(k):
        sampled_indices = random_sample(bins[i], n=bins_n)
        index.extend(sampled_indices)
    #data = [data[i] for i in index]
    print(f"sampled: {len(index)} examples")

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--total_len", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    #data = utils.jload("alpaca_data.json")
    #total_len = len(data)

    # random sample
    if args.random:
        n = int(args.total_len * args.ratio)
        index = np.random.choice(args.total_len, n, replace=False) 
        print(f"Random sample: {len(index)} examples")
    
    # DQ
    else:
        k = args.k
        ratio = args.ratio
        index = dataset_quantization(embed_path=args.embed_path, ratio=ratio, k=k)
        print(f"DQ: {len(index)} examples")

    with open(args.save_path, 'wb') as f:
        pickle.dump(index, f)
