import os
from data.data import normalize
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def render(mat, outpath):
    im = Image.fromarray(mat).convert('L')
    im.save(outpath)


def contrast(mats, outpath, suptitle, titles, cmap='gray'):
    n = len(mats)
    fig, subplots = plt.subplots(1, n)
    fig.suptitle(suptitle)

    for i in range(n):
        subplots[i].imshow(mats[i], cmap=cmap)
        subplots[i].set_title(titles[i])

    plt.savefig(outpath)


def plot_heatmap(mat):
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(1,1,1)


def plot_accuracy():
    return 0
