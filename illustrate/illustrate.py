import matplotlib.pyplot as plt


def draw(mat, outpath, cmap='grayscale'):
    plt.imshow(mat, cmap=cmap)
    plt.show()
    plt.close()
    plt.savefig(outpath)


def contrast_3(mat1, mat2, mat3, outpath, cmap='grayscale'):
    fig, (p1, p2, p3) = plt.subplots(1, 4)
    p1.imshow(mat1, cmap=cmap)
    p2.imshow(mat2, cmap=cmap)
    p3.imshow(mat3, cmap=cmap)
    cbar = plt.colorbar(p3)

    plt.show()
    plt.close()
    plt.savefig(outpath)
