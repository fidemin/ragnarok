import numpy as np

from src.main.cnn.layer import Convolution, Pooling
from src.main.core.optimizer import SGD


class TestConvolution:
    def test_forward(self):
        test_input = np.random.rand(10, 3, 4, 4)
        test_W = np.random.rand(5, 3, 3, 3)
        test_b = np.random.rand(5)

        layer = Convolution(test_W, test_b, SGD())
        out = layer.forward(test_input)

        assert out.shape == (10, 5, 2, 2)

    def test_backward(self):
        N = 10
        C = 3
        H = 4
        W = 4
        FN = 5
        FH = 3
        FW = 3
        padding = 0
        stride = 1

        OH = (H + 2 * padding - FH) // stride + 1
        OW = (W + 2 * padding - FW) // stride + 1

        test_x = np.random.rand(N, C, H, W)
        test_dout = np.random.rand(N, FN, OH, OW)
        test_W = np.random.rand(FN, C, FH, FW)
        test_b = np.random.rand(FN)

        layer = Convolution(test_W, test_b, SGD(), stride=stride, padding=padding)
        layer.forward(test_x)
        dx = layer.backward(test_dout)

        assert dx.shape == (N, C, H, W)
        assert layer._dF.shape == (FN, C, FH, FW)
        assert layer._db.shape == (FN,)


class TestPooling:

    def test_forward(self):
        test_input = np.array([
            [
                [
                    [1.0, 2.0, 1.0],
                    [-1.0, 3.0, 2.2],
                    [5.0, -4.0, 2.5]
                ],
                [
                    [5.0, 4.0, 1.1],
                    [-2.0, 1.0, 2.0],
                    [2.5, 3.0, 3.1]
                ]
            ],
            [
                [
                    [3.0, 2.0, 3.2],
                    [-1.0, 1.1, 5.0],
                    [4.5, 2.4, 2.1]
                ],
                [
                    [3.3, 2.3, -2.2],
                    [3.4, 2.1, -1.1],
                    [2.5, 6.8, 2.1]
                ]
            ]
        ])

        padding = 0
        stride = 1
        PH = 2
        PW = 2

        pooling = Pooling(PH, PW, stride=stride, padding=padding)

        actual = pooling.forward(test_input)

        expected = np.array([
            # set 1
            [
                # C1
                [
                    [3.0, 3.0], [3.0, 3.0]
                ],
                # C2
                [
                    [5.0, 4.0], [3.0, 3.1]
                ]

            ],
            # set 2
            [
                # C1
                [
                    [3.0, 5.0], [4.5, 5.0]
                ],
                # C2
                [
                    [3.4, 2.3], [6.8, 6.8]
                ]
            ]
        ])

        np.allclose(actual, expected)

    def test_backward(self):
        test_input = np.array([
            [
                [
                    [1.0, 2.0, 1.0],
                    [-1.0, 3.0, 2.2],
                    [5.0, -4.0, 2.5]
                ],
                [
                    [5.0, 4.0, 1.1],
                    [-2.0, 1.0, 2.0],
                    [2.5, 3.0, 3.1]
                ]
            ],
            [
                [
                    [3.0, 2.0, 3.2],
                    [-1.0, 1.1, 5.0],
                    [4.5, 2.4, 2.1]
                ],
                [
                    [3.3, 2.3, -2.2],
                    [3.4, 2.1, -1.1],
                    [2.5, 6.8, 2.1]
                ]
            ]
        ])

        test_dout = expected = np.array([
            # set 1
            [
                # C1
                [
                    [3.0, 3.0], [3.0, 3.0]
                ],
                # C2
                [
                    [5.0, 4.0], [3.0, 3.1]
                ]

            ],
            # set 2
            [
                # C1
                [
                    [3.0, 5.0], [4.5, 5.0]
                ],
                # C2
                [
                    [3.4, 2.3], [6.8, 6.8]
                ]
            ]
        ])

        expected = np.array([
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [3.0, 0.0, 0.0]
                ],
                [
                    [5.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 3.0, 3.1]
                ]
            ],
            [
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0],
                    [4.5, 0.0, 0.0]
                ],
                [
                    [0.0, 2.3, 0.0],
                    [3.4, 0.0, 0.0],
                    [0.0, 13.6, 0.0]
                ]
            ]
        ])

        padding = 0
        stride = 1
        PH = 2
        PW = 2

        pooling = Pooling(PH, PW, stride=stride, padding=padding)

        pooling.forward(test_input)
        actual = pooling.backward(test_dout)

        assert np.allclose(actual, expected)
