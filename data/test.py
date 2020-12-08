import numpy as np
from data.data_index import *


def check_index():
    print('53, 54, 55, 56, 57, 58 : ')
    print(len(s53), len(s54), len(s54), len(s55), len(s56), len(s57), len(s58))

    height = 48
    width = 64
    mat = np.zeros((height, width), dtype=int)
    indices = [53, 54, 55, 56, 57, 58, 1, 2, 3, 4]
    slots = [s53, s54, s55, s56, s57, s58, A, B, C, D]

    index = 0
    for s in slots:
        for h, w in s:
            mat[h, w] = indices[index]
        index += 1
    # print('ROI and Noise Area Distributions')
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(mat)
    header = '53 = slot 53, etc.\nA = 1, B = 2, C = 3, D = 4\n'
    np.savetxt('ROI_and_Area_Distributions.txt', mat, fmt='%2.d', header=header)


def main():
    check_index()


if __name__ == '__main__':
    main()