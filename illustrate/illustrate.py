import os
import numpy as np
from math import sin, cos, radians
import data
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as pat


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
    range_depth = 12.8
    y_lim = +cos(radians(60)) * range_depth
    range_width = +sin(radians(60)) * range_depth

    x, y = data.get_transform_index()
    #x, y, mat = x.ravel(), y.ravel(), mat.ravel()
    zi = data.interpolate(mat)

    # matplotlib
    plt.figure(figsize=(10,6))
    ax = plt.subplot(1,1,1)
    #ax.imshow(mat, cmap='rainbow',
              #extent=[-range_width, +range_width, 0, range_depth],
              #alpha=0.95)
    ax.pcolormesh(x, y, mat, cmap='rainbow')
    #ax.scatter(x, y, mat)
    ax.set_title('Range Azimuth Heatmap (-60, 60)')
    ax.set_xlabel('Azimuth [m]')
    ax.set_ylabel('Range [m]')
    ax.set_xlim([-range_width - 0.5, range_width + 0.5])
    ax.set_ylim([0, range_depth + 0.5])
    ax.plot([0, -range_width], [0, +y_lim], color='black', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, +y_lim], color='black', linewidth=0.5, linestyle=':', zorder=1)
    plt.show()
    plt.savefig('Heatmap.png')
    plt.close()


def plot_accuracy():
    return 0
