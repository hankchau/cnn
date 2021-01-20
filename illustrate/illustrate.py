import data
import numpy as np
import matplotlib.pyplot as plt
import gc
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


def plot_heatmap(mat, outpath, shading='gouraud', cmap='rainbow', roi_boxes=True):
    if len(mat.size) < 2:
        print('error: heatmap of csv matrix could not be plotted')
        exit(0)
    range_depth = 12.8
    range_width = +sin(radians(60)) * range_depth

    x, y = data.get_transform_index()
    # matplotlib
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    fig.suptitle('Range Azimuth Heatmap (-60\N{DEGREE SIGN}, 60\N{DEGREE SIGN})')
    fig.xlabel('Azimuth [m]')
    fig.ylabel('Range [m]')
    fig.xlim([-range_width - 0.5, range_width + 0.5])
    fig.ylim([0, range_depth + 0.5])

    im = ax.pcolormesh(x, y, mat, cmap=cmap, shading=shading, vmin=0.0, vmax=500)
    fig.colorbar(im)
    fig.tight_layout()

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

    fig.savefig(outpath)
    plt.cla()
    plt.clf()
    plt.close(fig)
    gc.collect()


def plot_heatmaps(csv_files, outpaths, shading='gouraud', cmap='rainbow', roi_boxes=True):
    range_depth = 12.8
    range_width = +sin(radians(60)) * range_depth

    x, y = data.get_transform_index()
    # matplotlib
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    fig.suptitle('Range Azimuth Heatmap (-60\N{DEGREE SIGN}, 60\N{DEGREE SIGN})')

    for i in range(len(csv_files)):
        ax.set_xlabel('Azimuth [m]')
        ax.set_ylabel('Range [m]')
        ax.set_xlim([-range_width - 0.5, range_width + 0.5])
        ax.set_ylim([0, range_depth + 0.5])
        # plot ROI boxes
        if roi_boxes:
            slots = [53, 54, 55, 56, 57, 58]
            for j in range(len(slots)):
                id = slots[j]
                reg = data.data_index[str(id)]
                pts = reg.get_corners()
                pts.append(pts[0])
                x_coord = []
                y_coord = []
                for p in pts:
                    x_coord.append(p[0])
                    y_coord.append(p[1])
                ax.plot(x_coord, y_coord, 'black')

        mat = data.read_csv(csv_files[i])
        im = ax.pcolormesh(x, y, mat, cmap=cmap, shading=shading, vmin=0.0, vmax=500)
        cb = fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(outpaths[i])
        cb.remove()
        plt.cla()
        gc.collect()
    plt.clf()
    plt.close(fig)
    gc.collect()


def plot_ROIs(hm_files, outpaths):
    for i in range(len(hm_files)):
        with Image.open(hm_files[i]) as im:
            im = im.convert('RGB')
            top, bot = 73, 154
            left, right = 112, 343
            im = im.crop((left, top, right, bot))
            im.save(outpaths[i])
            im = im.resize((256, 64))
            im.close()
        gc.collect()
