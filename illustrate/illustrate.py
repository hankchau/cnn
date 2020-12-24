import data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sin, radians


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


def plot_heatmap(mat, fpath, shading='gouraud', cmap='rainbow'):
    if len(mat.shape) == 1:
        print(fpath + ' cannot be generated, since no data of this kind is found.')
        return 0

    range_depth = 12.8
    range_width = +sin(radians(60)) * range_depth

    x, y = data.get_transform_index()
    # matplotlib
    plt.figure(figsize=(12,6))
    plt.pcolormesh(x, y, mat, cmap=cmap, shading=shading, vmin=0.0, vmax=500)
    plt.colorbar()
    plt.title('Range Azimuth Heatmap (-60\N{DEGREE SIGN}, 60\N{DEGREE SIGN})')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    plt.xlim([-range_width - 0.5, range_width + 0.5])
    plt.ylim([0, range_depth + 0.5])
    plt.tight_layout()

    slots = [53, 54, 55, 56, 57, 58]
    # plot ROI boxes
    for i in range(len(slots)):
        id = slots[i]
        reg = data.data_index[str(id)]
        pts = reg.get_corners()
        pts.append(pts[0])

        new_x = []
        new_y = []
        for p in pts:
            new_x.append(x[p[0], p[1]])
            new_y.append(y[p[0], p[1]])
        plt.plot(new_x, new_y, 'black')

    plt.savefig(fpath)
    plt.close()


def plot_accuracy():
    return 0
