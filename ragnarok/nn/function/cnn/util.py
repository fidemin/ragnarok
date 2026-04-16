import numpy as np


def _np_img2col(
    x_data: np.ndarray,
    *,
    FH: int,
    FW: int,
    padding: int,
    stride: int,
) -> np.ndarray:
    N, C, H, W = x_data.shape

    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1

    img = np.pad(x_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    col_data = np.zeros((N, C, OH, OW, FH, FW))

    for out_y in range(OH):
        y = out_y * stride
        y_max = y + FH

        for out_x in range(OW):
            x_data = out_x * stride
            x_max = x_data + FW
            col_data[:, :, out_y, out_x, :, :] = img[:, :, y:y_max, x_data:x_max]

    # col.shape: (N, C, OH, OW, FH, FW) -> (N, OH, OW, C, FH, FW)
    col_data = col_data.transpose((0, 2, 3, 1, 4, 5))
    # col.shape: (N, OH, OW, C, FH, FW) -> (N * OH * OW, C * FH * FW)
    col_data = col_data.reshape((N * OH * OW, -1))
    return col_data


def _np_col2img(
    dcol_x_data: np.ndarray,
    *,
    N: int,
    C: int,
    H: int,
    W: int,
    FH: int,
    FW: int,
    padding: int,
    stride: int,
) -> np.ndarray:
    # dcol_x_data.shape: (N * OH * OW, C * FH * FW)
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1

    # col_data.shape: (N, OH, OW, C, FH, FW)
    col_data = dcol_x_data.reshape((N, OH, OW, C, FH, FW))

    # col_data.shape: (N, C, OH, OW, FH, FW)
    col_data = col_data.transpose(0, 3, 1, 2, 4, 5)

    # img_data.shape: (N, C, H + 2 * padding, W + 2 * padding)
    img_data_w_padding = np.pad(
        np.zeros((N, C, H, W)),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
    )

    for out_y in range(OH):
        y = out_y * stride
        y_max = y + FH

        for out_x in range(OW):
            x = out_x * stride
            x_max = x + FW
            img_data_w_padding[:, :, y:y_max, x:x_max] += col_data[
                :, :, out_y, out_x, :, :
            ]

    # img_data.shape: (N, C, H, W)
    img_data = img_data_w_padding[:, :, padding : H + padding, padding : W + padding]
    return img_data
