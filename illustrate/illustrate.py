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


def plot_heatmap(mat, fpath, shading='gouraud', cmap='rainbow', roi_boxes=True):
    if len(mat.shape) == 1:
        print(fpath + ' cannot be generated, since no data of this kind is found.')
        return 0

    range_depth = 12.8
    range_width = +sin(radians(60)) * range_depth

    x, y = data.get_transform_index()
    # matplotlib
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(x, y, mat, cmap=cmap, shading=shading, vmin=0.0, vmax=500)
    fig.colorbar(im)
    ax.set_title('Range Azimuth Heatmap (-60\N{DEGREE SIGN}, 60\N{DEGREE SIGN})')
    ax.set_xlabel('Azimuth [m]')
    ax.set_ylabel('Range [m]')
    ax.set_xlim([-range_width - 0.5, range_width + 0.5])
    ax.set_ylim([0, range_depth + 0.5])

    # plot ROI boxes
    if roi_boxes:
        slots = [53, 54, 55, 56, 57, 58]
        for i in range(len(slots)):
            id = slots[i]
            reg = data.data_index[str(id)]
            pts = reg.get_corners()
            pts.append(pts[0])

            x_coord = []
            y_coord = []
            for p in pts:
                x_coord.append(p[0])
                y_coord.append(p[1])
            ax.plot(x_coord, y_coord, 'black')
    fig.tight_layout()
    fig.savefig(fpath)
    fig.clf()
    plt.close(fig)


def plot_accuracy():
    return 0
