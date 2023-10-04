import numpy as np


def img2col(input_array: np.ndarray, FH, FW, padding=0, stride=1) -> np.ndarray:
    N, C, H, W = input_array.shape
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1

    img = np.pad(input_array, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    col = np.zeros((N, C, OH, OW, FH, FW))

    for out_y in range(OH):
        y = out_y * stride
        y_max = y + FH

        for out_x in range(OW):
            x = out_x * stride
            x_max = x + FW
            col[:, :, out_y, out_x, :, :] = img[:, :, y:y_max, x:x_max]

    col = col.transpose((0, 2, 3, 1, 4, 5))
    col = col.reshape((N * OH * OW, -1))

    # out.shape: (N * OH * OW, C * FH * FW)
    return col


def col2img(input_array: np.ndarray, N, C, H, W, FH, FW, padding=0, stride=1) -> np.ndarray:
    # input_array.shape: (N * OH * OW, C * FH * FW)

    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1

    col = input_array.reshape((N, OH, OW, C, FH, FW))
    # col.shape: (N, C, OH, OW, FH, FW)
    col = col.transpose(0, 3, 1, 2, 4, 5)
    img = np.pad(np.zeros((N, C, H, W)), [(0, 0), (0, 0), (padding, padding), (padding, padding)])

    for out_y in range(OH):
        y = out_y * stride
        y_max = y + FH

        for out_x in range(OW):
            x = out_x * stride
            x_max = x + FW
            img[:, :, y:y_max, x:x_max] = col[:, :, out_y, out_x, :, :]

    # target shape: (N, C, H, W)
    return img[:, :, padding:H + padding, padding:W + padding]


def fil2col(fil: np.ndarray) -> np.ndarray:
    # fil.shape: (FN, FC, FH, FW)
    FN = fil.shape[0]

    # returns (FN, FC * FH * FW)
    return fil.reshape(FN, -1)


def col2fil(col: np.ndarray, FC: int, FH: int, FW: int) -> np.ndarray:
    # col.shape: (FN, FC * FH * FW)

    # returns: (FN, FC, FH, FW)
    return col.reshape((col.shape[0], FC, FH, FW))
