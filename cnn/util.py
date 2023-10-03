import numpy as np


def img2col(input_array: np.ndarray, filter_height, filter_width, padding=0, stride=1) -> np.ndarray:
    N, C, H, W = input_array.shape
    out_h = (H + 2 * padding - filter_height) // stride + 1
    out_w = (W + 2 * padding - filter_width) // stride + 1

    img = np.pad(input_array, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    col = np.zeros((N, C, out_h, out_w, filter_height, filter_width))

    for out_y in range(out_h):
        y = out_y * stride
        y_max = y + filter_height

        for out_x in range(out_w):
            x = out_x * stride
            x_max = x + filter_width
            col[:, :, out_y, out_x, :, :] = img[:, :, y:y_max, x:x_max]

    col = col.transpose((0, 2, 3, 1, 4, 5))
    col = col.reshape((N * out_h * out_w, -1))

    # out.shape: (N * out_h * out_w, C * filter_h * filter_w)
    return col


def fil2col(fil: np.ndarray) -> np.ndarray:
    FN = fil.shape[0]

    return fil.reshape(FN, -1)
