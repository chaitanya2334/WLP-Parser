import sys

import pandas
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

import config as cfg
import matplotlib

from visualization.vtsne import VTSNE
from visualization.wrapper import Wrapper

matplotlib.use('Agg')


def preprocess(X, y, perplexity=30, metric='euclidean'):
    pos = X
    y = y
    n_points = pos.shape[0]
    print("Creating Pairwise Distances ...", end="")
    sys.stdout.flush()
    distances2 = pairwise_distances(pos, metric=metric, squared=True)
    print(" Done")
    # This return a n x (n-1) prob array
    print("Generating Joint probabilities ...", end="")
    sys.stdout.flush()
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    print(" Done")
    # Convert to n x n prob array
    pij = squareform(pij)
    return n_points, pij, y

def points_from_window(df, xmin, xmax, ymin, ymax):
    df_win = df.ix[(xmin <= df['x']) & (df['x'] <= xmax) & (ymin <= df['y']) & (df['y'] <= ymax)]
    return df_win

def scatter_points(x, y, labels):
    # We create a scatter plot.
    df = pandas.DataFrame({'x': x[:, 0], 'y': x[:, 1], 'words': y[:], 'labels': labels})
    df.to_csv("tsne_data.csv", sep=" ", columns=["x", "y", "words", "labels"], index=False)

    df_win1 = points_from_window(df, xmin=5, xmax=10, ymin=5, ymax=10)
    df_win2 = points_from_window(df, xmin=-15, xmax=-10, ymin=-20, ymax=-15)

    df_win1.to_csv("win1.csv", sep=" ", columns=["x", "y", "words", "labels"], index=False)
    df_win2.to_csv("win2.csv", sep=" ", columns=["x", "y", "words", "labels"], index=False)

    f = plt.figure(figsize=(8, 8))
    plt.axes().set_aspect('equal')
    oov_i = [i for i in range(len(labels)) if labels[i] == "OOV"]
    not_oov_i = [i for i in range(len(labels)) if labels[i] == "Not OOV"]

    plt.scatter(x[oov_i, 0], x[oov_i, 1], y[oov_i], lw=0, s=4, c='red', label="OOV")
    plt.scatter(x[not_oov_i, 0], x[not_oov_i, 1], y[not_oov_i], lw=0, s=4, c='blue', label="Not OOV")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.axis('on')
    plt.axis('tight')
    plt.legend(labels=["OOV", "Not OOV"])


    # for i, txt in enumerate(y):
    #     ax.annotate(txt[0], (x[i, 0], x[i, 1]))
    from matplotlib2tikz import save as tikz_save
    tikz_save('test.tex', figure=f)
    save_file = os.path.join(cfg.VIS_SAVE_DIR, 'tsne_file.png')
    plt.savefig(save_file, dpi=120)
    return f


def simple_vis_tsne(X, y, labels):
    RS = 123424
    x = TSNE(random_state=RS, n_iter=1000, learning_rate=10, perplexity=23).fit_transform(X)
    print("TSNE done")
    scatter_points(x, y, labels)


def viz_tsne(X, y):
    draw_ellipse = True
    n_points, pij2d, y = preprocess(X, y)
    i, j = np.indices(pij2d.shape)
    i = i.ravel()
    j = j.ravel()
    pij = pij2d.ravel().astype('float32')
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    n_topics = 2
    n_dim = 2
    print(n_points, n_dim, n_topics)

    model = TSNE(n_points, n_topics, n_dim)
    wrap = Wrapper(model, batchsize=32768, epochs=1)
    for itr in tqdm(range(500)):
        wrap.fit(pij, i, j)
        save_file = os.path.join(cfg.VIS_SAVE_DIR, 'scatter_{:03d}.png'.format(itr))
        # Visualize the results
        embed = model.logits.weight.cpu().data.numpy()
        f = plt.figure()
        if not draw_ellipse:
            plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
            plt.axis('off')
            plt.savefig(save_file, bbox_inches='tight')
            plt.close(f)
        else:
            # Visualize with ellipses
            var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
            ax = plt.gca()
            for xy, (w, h), c in zip(embed, var, y):
                e = Ellipse(xy=xy, width=w, height=h, ec=None, lw=0.0)
                e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
                e.set_alpha(0.5)
                ax.add_artist(e)
            ax.set_xlim(-9, 9)
            ax.set_ylim(-9, 9)
            plt.axis('off')
            plt.savefig(save_file, bbox_inches='tight')
            plt.close(f)
