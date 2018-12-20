import numpy as np
from sklearn.manifold import TSNE


def apply_tsne(x):
    x_embedded = TSNE(n_components=2, verbose=1, n_iter=250).fit_transform(x)
    return x_embedded