from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable, ones_like
from src.main.ragnarok.nn.function.conv import (
    img2col,
    fil2col,
    Conv2D,
    col2img,
    col2fil,
)


def test_img2col():
    # N, C, H, W = 1 X 2 X 2 X 3
    arr = [
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
    ]

    input_var = Variable(arr)

    col = img2col(input_var, FH=2, FW=3, padding=2, stride=1)

    expected_arr = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9],
        [0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.4, 0.0, 0.0, 0.7, 0.0, 0.0, 1.0],
        [0.0, 0.1, 0.2, 0.0, 0.4, 0.5, 0.0, 0.7, 0.8, 0.0, 1.0, 1.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.2, 0.3, 0.0, 0.5, 0.6, 0.0, 0.8, 0.9, 0.0, 1.1, 1.2, 0.0],
        [0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 1.2, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 0.0, 0.0, 0.0],
        [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 1.1, 1.2, 0.0, 0.0, 0.0],
        [0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 1.1, 1.2, 0.0, 0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    expected_var = Variable(expected_arr)
    assert allclose(col, expected_var)


def test_col2img():
    input_arr = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9],
        [0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.4, 0.0, 0.0, 0.7, 0.0, 0.0, 1.0],
        [0.0, 0.1, 0.2, 0.0, 0.4, 0.5, 0.0, 0.7, 0.8, 0.0, 1.0, 1.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.2, 0.3, 0.0, 0.5, 0.6, 0.0, 0.8, 0.9, 0.0, 1.1, 1.2, 0.0],
        [0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 1.2, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 0.0, 0.0, 0.0],
        [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 1.1, 1.2, 0.0, 0.0, 0.0],
        [0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 1.1, 1.2, 0.0, 0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    # img_arr_w_padding = [
    #     [
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.1 * 6, 0.2 * 6, 0.3 * 6, 0.0, 0.0],
    #         [0.0, 0.0, 0.4 * 6, 0.5 * 6, 0.6 * 6, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     ],
    # ]

    expected_arr = [
        [
            [[0.1 * 6, 0.2 * 6, 0.3 * 6], [0.4 * 6, 0.5 * 6, 0.6 * 6]],
            [[0.7 * 6, 0.8 * 6, 0.9 * 6], [1.0 * 6, 1.1 * 6, 1.2 * 6]],
        ],
    ]
    expected_var = Variable(expected_arr)

    input_var = Variable(input_arr)

    actual_var = col2img(input_var, N=1, C=2, H=2, W=3, FH=2, FW=3, padding=2, stride=1)
    allclose(actual_var, expected_var)


def test_fil2col():
    # FN, FC, FH, FW = 2 X 2 X 2 X 3
    arr = [
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
        [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]],
    ]

    input_var = Variable(arr)

    col = fil2col(input_var)

    expected_arr = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
    ]

    expected_var = Variable(expected_arr)
    assert allclose(col, expected_var)


def test_col2fil():
    input_arr = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
    ]

    input_var = Variable(input_arr)

    actual_var = col2fil(input_var, FC=2, FH=2, FW=3)

    expected_arr = [
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
        [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]],
    ]
    expected_var = Variable(expected_arr)

    assert allclose(actual_var, expected_var)


class TestConv2D:
    def test_forward(self):
        # N, C, H, W = 1 X 2 X 2 X 3
        x_arr = [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
        ]

        # FN, FC, FH, FW = 2 X 2 X 2 X 3
        w_arr = [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
            [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]],
        ]

        # FN X 1 X 1 = 2 X 1 X 1
        b_arr = [[[0.1]], [[0.2]]]

        x_var = Variable(x_arr)
        w_var = Variable(w_arr)
        b_var = Variable(b_arr)

        f = Conv2D()

        actual = f(x_var, w_var, b_var, padding=2, stride=1)
        # N, FN, OH, OW = 1 X 2 X 2 X 3
        # expected_arr =

        # assert allclose(actual, Variable(expected_arr))
        expected_arr = [
            [
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [1.0, 2.0, 3.08, 2.12, 1.12],
                    [2.2, 4.38, 6.6, 4.38, 2.2],
                    [1.12, 2.12, 3.08, 2.0, 1.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                ],
                [
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [1.9, 3.9, 6.18, 4.42, 2.42],
                    [4.5, 9.28, 14.5, 10.08, 5.3],
                    [2.62, 5.22, 7.98, 5.5, 2.9],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                ],
            ]
        ]

        expected_var = Variable(expected_arr)
        assert (1, 2, 5, 5) == actual.shape
        assert allclose(actual, expected_var)

    def test_backward(self):
        x_arr = [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
        ]

        # FN, FC, FH, FW = 2 X 2 X 2 X 3
        w_arr = [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
            [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]],
        ]

        # FN X 1 X 1 = 2 X 1 X 1
        b_arr = [[[0.1]], [[0.2]]]

        x_var = Variable(x_arr)
        w_var = Variable(w_arr)
        b_var = Variable(b_arr)

        f = Conv2D()

        for_weak_ref = f(x_var, w_var, b_var, padding=2, stride=1)
        dout_arr = [
            [
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                ],
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                ],
            ]
        ]
        dout_var = Variable(dout_arr)

        dx, dW, db = f.backward(dout_var)

    def test_gradient_check(self):
        x_arr = [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
        ]
        w_arr = [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
            [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]],
        ]

        # FN X 1 X 1 = 2 X 1 X 1
        b_arr = [[[0.1]], [[0.2]]]

        x_var = Variable(x_arr)
        w_var = Variable(w_arr)
        b_var = Variable(b_arr)

        f1 = Conv2D()
        f2 = Conv2D()

        for_weak_ref = f1(x_var, w_var, b_var, padding=2, stride=1)
        actual_dx, actual_dw, actual_db = f1.backward(ones_like(for_weak_ref))
        expected_dx, expected_dw, expected_db = numerical_diff(
            f2, x_var, w_var, b_var, padding=2, stride=1
        )

        allclose(actual_dx, expected_dx)
        allclose(actual_dw, expected_dw)
        allclose(actual_db, expected_db)
