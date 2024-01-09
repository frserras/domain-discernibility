import threading
import os

from functools import cache
from pathlib import Path
from time import time

import numpy as np
import polars as pl

from gensim import models
from scipy import sparse as sp
from sklearn import metrics


def load_model(path):
    return models.KeyedVectors.load_word2vec_format(str(path))


def simple_dict(docs):
    d = dict()
    idx = 0
    for doc in docs:
        for word in doc:
            if not word in d:
                d[word] = idx
                idx += 1
    return d


def create_ranges(counts):
    ranges = np.empty(counts.sum(), dtype=int)
    seq = np.arange(counts.max())
    starts = np.zeros(len(counts), dtype=int)
    starts[1:] = np.cumsum(counts[:-1])
    for start, count in zip(starts, counts):
        ranges[start : start + count] = seq[:count]
    return ranges


def populate(series):
    series = series.map_elements(lambda x: x.to_numpy())
    return np.concatenate(series)


def tfidf(data, row, col):
    uniq, freq = np.unique(data, return_counts=True)
    M = np.zeros((row.shape[0], uniq.shape[0]))
    return M, uniq, freq


def transform(df, model, c="sentence", t="domain_label"):
    d = model.key_to_index
    # Split tokens
    df = df.with_columns(pl.col(c).str.split(by=" "))
    # Map tokens do ids
    df = df.with_columns(pl.col(c).list.eval(pl.element().map_dict(d)))

    # Count the total length
    df = df.with_columns(pl.col(c).list.len().alias("plength"))

    # Remove nulls from lists requires polar 0.19
    df = df.with_columns(pl.col(c).list.drop_nulls())

    # Create auxiliar structure
    df = df.with_columns(length=pl.col(c).list.len())
    df = df.with_columns(oovs=pl.col("plength") - pl.col("length"))

    # Remove empty lists
    df = df.filter(pl.col("length") > 0)

    # index=pl.int_range(0, df.shape[0]))
    lengths = df.select("length").to_numpy().flatten()
    oovs = df.select("oovs").to_numpy().flatten()
    # index = df.select('index').to_numpy().flatten()
    # row_idx = np.repeat(index, length)
    # col_idx = create_ranges(length)
    data = df.select(c).explode(c).to_numpy().flatten()
    labels = df.select(t).to_numpy().flatten()

    # This works, takes 6 minutes with a embedding of size 50
    # TODO: Extrair labels
    embeddings = model.get_normed_vectors()
    return embeddings, data, lengths, labels, oovs
    # df = df.with_columns(pl.col(c).map_elements(lambda x: np.sum(embeddings[x], axis=0)))
    return df, data, (row_idx, col_idx)

    normed_vectors = model.get_normed_vectors()
    M = np.empty((df.shape[0], normed_vectors.shape[1]), dtype="float32")
    # Now ndf is a vector of vector of indices
    # compute mean, taking each index
    return ndf


def parallelprocess(embeddings, data, lengths, step=1):
    starts = np.roll(lengths, 1)
    starts[0] = 0
    starts = starts.cumsum()
    output_shape = (1, embeddings.shape[1])
    total_tasks = len(lengths)

    # output = np.empty((len(lengths), embeddings.shape[1]))
    output = []

    def worker(i, step):
        worker_output = np.zeros((step, output_shape[1]), dtype="float32")
        total_iterations = min(total_tasks, i + step)
        c = step
        for c, j in enumerate(range(i, total_iterations, 1)):
            result = embeddings[data[starts[j] : starts[j] + lengths[j]]].sum(axis=0)
            np.add.at(worker_output, j - i, result)
            # worker_output.append(result.reshape(output_shape))
        # output.append((i, np.concatenate(worker_output)))
        output.append((i, worker_output[: c + 1]))

    threads = []
    # for i in range(lengths)//10_000):
    for i in range(0, len(lengths), step):
        thread = threading.Thread(target=worker, args=(i, step))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Sort output
    # worker(total_tasks-3, step)
    output.sort()
    output = np.concatenate([v for i, v in output])
    return output


def check(output):
    for i, (idx, val) in enumerate(output):
        if i != idx:
            return i
    return -1


if __name__ == "__main__":
    input_dir = Path("./experiments/embeddings")
    output_dir = Path("./experiments/vectors2")

    output_dir.mkdir(exist_ok=True, parents=True)

    src_file = Path("./carolina_balanceado_sentenciado_rlx.csv")

    df = pl.read_csv(src_file)
    embedding_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith("txt"):
                embedding_files.append((f, os.path.join(root, f)))
                print(f, os.path.join(root, f))

    for name, embed_file_path in embedding_files:
        output_file_name = output_dir / name.replace("txt", "npz")
        if output_file_name.exists():
            print("skipping", name)
            continue
        model = load_model(embed_file_path)
        embeddings, data, lengths, labels, oovs = transform(
            df, model, c="sentence", t="domain_label"
        )
        vectorized = parallelprocess(embeddings, data, lengths, step=200)
        np.savez_compressed(
            str(output_file_name).replace(".npz", ""),
            x=vectorized,
            y=labels,
            l=lengths,
            o=oovs,
        )
