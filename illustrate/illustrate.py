import os
from scipy.interpolate import griddata
import data
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
    plt.figure(figsize=(12, 6, ))
    plt.title('Range Azimuth Heatmap')
    plt.xlabel('Azimuth(-60, 60)')
    plt.ylabel('Range(m)')
    plt.show()
    plt.close()
    x, y, xi, yi = data.transform()
    zi = griddata((x.ravel(), y.ravel()), mat.ravel(), (xi, yi))
    plt.imshow(zi.T, cmap='rainbow', extent=(0, 12.8, -60.0, 60.0), origin='lower')


def plot_accuracy():
    return 0
