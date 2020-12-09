import os
import matplotlib.pyplot as plt


def draw(mat, outpath, title, cmap='grayscale'):
    plt.imshow(mat, cmap=cmap)
    plt.title = title
    plt.show()
    plt.close()
    plt.savefig(outpath)


def contrast_2(mat1, mat2, outpath, titles, cmap='grayscale'):
    fig, (p1, p2) = plt.subplots(1, 2)
    p1.imshow(mat1, cmap=cmap)
    p1.set_title(titles[0])
    p2.imshow(mat2, cmap=cmap)
    p2.set_title(titles[1])
    cbar = plt.colorbar(p2)

    plt.show()
    plt.close()
    plt.savefig(outpath)


def contrast_3(mat1, mat2, mat3, outpath, titles, cmap='grayscale'):
    fig, (p1, p2, p3) = plt.subplots(1, 3)
    p1.imshow(mat1, cmap=cmap)
    p1.set_title(titles[0])
    p2.imshow(mat2, cmap=cmap)
    p2.set_title(titles[1])
    p3.imshow(mat3, cmap=cmap)
    p3.set_title(titles[2])
    cbar = plt.colorbar(p3)

    plt.show()
    plt.close()
    plt.savefig(outpath)


def plot_accuracy():
    return 0
