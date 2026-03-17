import numpy as np
import pytest

from ragnarok.core.function import (
    FunctionTensorError,
    NotSupportedOperationException,
    GetItem,
    GetItemGrad,
)
from ragnarok.core.function import (
    Split,
    Reshape,
    Transpose,
    SumTo,
    BroadcastTo,
    Sum,
    Comparison,
)
from ragnarok.core.function.common import Function
from ragnarok.core.tensor import Tensor, ones_like
from ragnarok.core.tensor.tensor import zeros_like
from ragnarok.core.util import numerical_diff, allclose


class FunctionForTest(Function):
    def backward(self, *douts: Tensor):
        return douts[0]

    def forward(self, *tensors: Tensor, **kwargs):
        return tensors[0]

    def _validate_tensors(self, *tensors: Tensor):
        pass


class TestComparison:
    @pytest.mark.parametrize(
        "operator, expected",
        [
            ("eq", [False, False, True]),
            ("ne", [True, True, False]),
            ("gt", [False, True, False]),
            ("ge", [False, True, True]),
            ("lt", [True, False, False]),
            ("le", [True, False, True]),
        ],
    )
    def test_forward(self, operator, expected):
        test_input1 = Tensor([0.1, 0.2, 0.3])
        test_input2 = Tensor([0.2, 0.1, 0.3])

        f = Comparison()

        actual = f.forward(test_input1, test_input2, operator=operator)
        expected_var = Tensor(expected)

        assert allclose(actual, expected_var)

    def test_backward(self):
        test_input1 = Tensor([0.1, 0.2, 0.3])
        test_input2 = Tensor([0.2, 0.1, 0.3])
        dout = Tensor([1.0, 1.0, 1.0])

        f = Comparison()
        f(test_input1, test_input2, operator="eq")

        Tensor([0.0, 0.0, 1.0])

        with pytest.raises(NotSupportedOperationException) as exc_info:
            f.backward(dout)

        print("error message: ", str(exc_info.value))

    @pytest.mark.parametrize(
        "operator, shapes",
        [
            # wrong operator
            ("db", [(2,), (2,)]),
            # the number of input tensors is not 2
            ("le", [(2,), (2,), (2,)]),
            ("le", [(2,)]),
        ],
    )
    def test_validation_error(self, operator, shapes):
        input_vars = []
        for shape in shapes:
            input_vars.append(Tensor(np.random.rand(*shape)))

        f = Comparison()
        with pytest.raises(FunctionTensorError) as exc_info:
            f(*input_vars, operator=operator)

        print("error message: ", str(exc_info.value))


class TestSplit:
    @pytest.mark.parametrize(
        "test_input,axis,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                0,
                [
                    Tensor(np.array([[1.0, 2.0, 3.0]])),
                    Tensor(np.array([[2.0, 4.0, 8.0]])),
                ],
            ),
            (
                Tensor(np.array([[1.0, 3.0], [2.0, 4.0]])),
                1,
                [
                    Tensor(np.array([[1.0], [2.0]])),
                    Tensor(np.array([[3.0], [4.0]])),
                ],
            ),
        ],
    )
    def test_forward(self, test_input, axis, expected):
        split = Split()
        actual = split.forward(test_input, axis=axis)
        assert len(actual) == len(expected)

        for i in range(len(actual)):
            assert allclose(actual[i], expected[i])

    @pytest.mark.parametrize(
        "test_input,axis,expected",
        [
            (
                [
                    Tensor(np.array([[1.0, 2.0, 3.0]])),
                    Tensor(np.array([[2.0, 4.0, 8.0]])),
                ],
                0,
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            (
                [
                    Tensor(np.array([[1.0], [2.0]])),
                    Tensor(np.array([[3.0], [4.0]])),
                ],
                1,
                Tensor(np.array([[1.0, 3.0], [2.0, 4.0]])),
            ),
        ],
    )
    def test_backward(self, test_input, axis, expected):
        output_shape = expected.data.shape
        forward_input = Tensor(np.random.rand(*output_shape))

        split = Split()
        split(forward_input, axis=axis)
        actual = split.backward(*test_input)

        assert allclose(actual, expected)


class TestReshape:
    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                (3, 2),
                Tensor(np.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]])),
            ),
            (
                Tensor(np.array([[1.0, 3.0], [2.0, 4.0]])),
                (4,),
                Tensor(np.array([1.0, 3.0, 2.0, 4.0])),
            ),
        ],
    )
    def test_forward(self, test_input, shape, expected):
        split = Reshape()
        actual = split.forward(test_input, shape=shape)
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]])),
                (3, 2),
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            (
                Tensor(np.array([1.0, 3.0, 2.0, 4.0])),
                (2, 2),
                Tensor(np.array([[1.0, 3.0], [2.0, 4.0]])),
            ),
        ],
    )
    def test_backward(self, test_input, shape, expected):
        output_shape = expected.data.shape
        forward_input = Tensor(np.random.rand(*output_shape))

        split = Reshape()
        split(forward_input, shape=shape)
        actual = split.backward(test_input)

        assert allclose(actual, expected)


class TestTranspose:
    @pytest.mark.parametrize(
        "test_input,transpose, expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                None,
                Tensor(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
            ),
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                (1, 0),
                Tensor(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
            ),
            # more complex case
            (
                Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])),
                (1, 0, 2),
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [5.0, 6.0]],
                            [[3.0, 4.0], [7.0, 8.0]],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_forward(self, test_input, transpose, expected):
        f = Transpose()
        if transpose:
            actual = f.forward(test_input, transpose=transpose)
        else:
            actual = f.forward(test_input)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,transpose, expected",
        [
            (
                Tensor(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
                None,
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            (
                Tensor(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
                (1, 0),
                Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            # more complex case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [5.0, 6.0]],
                            [[3.0, 4.0], [7.0, 8.0]],
                        ]
                    )
                ),
                (1, 0, 2),
                Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])),
            ),
        ],
    )
    def test_backward(self, test_input, transpose, expected):
        output_shape = expected.shape
        forward_input = Tensor(np.random.rand(*output_shape))

        f = Transpose()
        if transpose:
            f(forward_input, transpose=transpose)
        else:
            f(forward_input)
        actual = f.backward(test_input)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,transpose",
        [
            # multi inputs
            (
                [
                    Tensor(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
                    Tensor(1.0),
                ],
                (1, 0, 2),
            ),
            # transpose is not tuple
            (
                [Tensor(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]]))],
                1,
            ),
        ],
    )
    def test_validate_tensors(self, test_input, transpose):
        with pytest.raises(FunctionTensorError):
            f = Transpose()(*test_input, transpose=transpose)


class TestBroadcastTo:
    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            # (3, 2) -> (3, 2) case: no broadcast required
            (
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (3, 2) -> (2, 3, 2) case
            (
                Tensor(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (1, 3, 2) -> (2, 3, 2) case
            (
                Tensor(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (2, 1, 2) -> (2, 3, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[6.0, 9.0], [6.0, 9.0], [6.0, 9.0]],
                            [[15.0, 18.0], [15.0, 18.0], [15.0, 18.0]],
                        ]
                    )
                ),
            ),
            #  (2, 1, 2, 1) -> (2, 3, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                            ],
                            [
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (1, 2, 2) -> (2, 3, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (3, 1, 2) -> (2, 3, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[12.0, 16.0]],
                            [[16.0, 20.0]],
                            [[20.0, 24.0]],
                        ],
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_forward(self, test_input, shape, expected):
        f = BroadcastTo()
        actual = f.forward(test_input, shape=shape)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "from_shape, to_shape",
        [
            (
                (1, 2, 3),
                (2, 3),
            ),
            (
                (3, 3),
                (2, 3, 2),
            ),
            (
                (1, 3, 3),
                (2, 3, 2),
            ),
        ],
    )
    def test_forward_error(self, from_shape, to_shape):
        with pytest.raises(FunctionTensorError) as exc_info:
            f = BroadcastTo()
            f.forward(Tensor(np.random.rand(*from_shape)), shape=to_shape)

        print("error message: ", exc_info.value)

    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            # (3, 2) -> (3, 2) case: no change
            (
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (3, 2) -> (2, 3, 2) broadcast case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Tensor(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
            ),
            # (1, 3, 2) -> (2, 3, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Tensor(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
            ),
            # (2, 1, 2) -> (2, 3, 2) broadcast case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
            ),
            # (2, 3, 1) -> (2, 3, 2) broadcast case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[3.0], [5.0], [7.0]],
                            [[9.0], [11.0], [13.0]],
                        ]
                    )
                ),
            ),
            # (2, 1, 2, 1) -> (2, 3, 2, 2) broadcast case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (1, 2, 2) -> (2, 3, 2, 2) broadcast case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
            ),
            # (3, 1, 2) -> (2, 3, 2, 2) broadcast case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[12.0, 16.0]],
                                [[16.0, 20.0]],
                                [[20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_backward(self, test_input, shape, expected):
        output_shape = expected.data.shape
        forward_input = Tensor(np.random.rand(*output_shape))

        f = BroadcastTo()
        f(forward_input, shape=shape)
        actual = f.backward(test_input)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = BroadcastTo()
        shape = (3, 2)
        test_input = Tensor(np.random.rand(*shape))

        f(test_input, shape=(2, 3, 2))

        tensor = Tensor(
            np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ]
            )
        )

        actual = f.backward(tensor)

        expected = numerical_diff(f, test_input, shape=(2, 3, 2))

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input, shape",
        [
            # no shape
            ([Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], None),
            # shape is not tuple
            ([Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], 3),
            # multiple tensors
            (
                [
                    Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
                    Tensor(np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])),
                ],
                (3, 2),
            ),
        ],
    )
    def test_validate_tensor(self, test_input, shape):
        with pytest.raises(FunctionTensorError) as exc_info:
            f = BroadcastTo()
            if shape:
                f(*test_input, shape=shape)
            else:
                f(*test_input)

        print("error message: ", exc_info.value)


class TestSumTo:
    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            # (3, 2) -> (3, 2) case: no sum required
            (
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (2, 3, 2) -> (3, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (3, 2),
                Tensor(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
            ),
            # (2, 3, 2) -> (1, 3, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (1, 3, 2),
                Tensor(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
            ),
            # (2, 3, 2) -> (2, 1, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 1, 2),
                Tensor(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
            ),
            # (2, 3, 2) -> (2, 3, 1) case
            (
                Tensor(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 1),
                Tensor(
                    np.array(
                        [
                            [[3.0], [5.0], [7.0]],
                            [[9.0], [11.0], [13.0]],
                        ]
                    )
                ),
            ),
            # (2, 3, 2, 2) -> (2, 1, 2, 1) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 1, 2, 1),
                Tensor(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (2, 3, 2, 2) -> (1, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (1, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
            ),
            # (2, 3, 2, 2) -> (3, 1, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (3, 1, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[12.0, 16.0]],
                                [[16.0, 20.0]],
                                [[20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_forward(self, test_input, shape, expected):
        f = SumTo()
        actual = f.forward(test_input, shape=shape)
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "from_shape, to_shape",
        [
            # The length of the given shape is larger than the length of the input
            (
                (2, 3),
                (1, 2, 3),
            ),
            # The from_shape can not sum to to_shape
            (
                (2, 3, 2),
                (3, 3),
            ),
            (
                (2, 3, 2),
                (1, 3, 3),
            ),
        ],
    )
    def test_forward_error(self, from_shape, to_shape):
        test_input = Tensor(np.random.rand(*from_shape))
        with pytest.raises(FunctionTensorError) as exc_info:
            f = SumTo()
            f.forward(test_input, shape=to_shape)

        print("error message: ", exc_info.value)

    @pytest.mark.parametrize(
        "test_input, shape, expected",
        [
            # (3, 2) -> (3, 2) case: no broadcast required
            (
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Tensor(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (3, 2) -> (2, 3, 2) case
            (
                Tensor(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (1, 3, 2) -> (2, 3, 2) case
            (
                Tensor(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (2, 1, 2) -> (2, 3, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Tensor(
                    np.array(
                        [
                            [[6.0, 9.0], [6.0, 9.0], [6.0, 9.0]],
                            [[15.0, 18.0], [15.0, 18.0], [15.0, 18.0]],
                        ]
                    )
                ),
            ),
            #  (2, 1, 2, 1) -> (2, 3, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                            ],
                            [
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (1, 2, 2) -> (2, 3, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (3, 1, 2) -> (2, 3, 2, 2) case
            (
                Tensor(
                    np.array(
                        [
                            [[12.0, 16.0]],
                            [[16.0, 20.0]],
                            [[20.0, 24.0]],
                        ],
                    )
                ),
                (2, 3, 2, 2),
                Tensor(
                    np.array(
                        [
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_backward(self, test_input, shape, expected):
        output_shape = expected.data.shape
        forward_input = Tensor(np.random.rand(*output_shape))

        f = SumTo()
        f(forward_input, shape=shape)
        actual = f.backward(test_input)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = SumTo()
        shape = (2, 3, 2)
        test_input = Tensor(np.random.rand(*shape))

        f(test_input, shape=(3, 2))

        tensor = Tensor(
            np.array(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            )
        )

        actual = f.backward(tensor)

        expected = numerical_diff(f, test_input, shape=(3, 2))

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input, shape",
        [
            # no shape
            ([Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], None),
            # shape is not tuple
            ([Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], 3),
            # multiple tensors
            (
                [
                    Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
                    Tensor(np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])),
                ],
                (3, 2),
            ),
        ],
    )
    def test_validate_tensor(self, test_input, shape):
        with pytest.raises(FunctionTensorError) as exc_info:
            f = SumTo()
            if shape:
                f(*test_input, shape=shape)
            else:
                f(*test_input)

        print("error message: ", exc_info.value)


class TestSum:
    @pytest.mark.parametrize(
        "test_input, axis, keepdims, expected",
        [
            ([1.0, 2.0, 3.0], None, False, 6.0),
            ([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], None, False, 15.0),
            ([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], 0, False, [6.0, 9.0]),
            ([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], 0, True, [[6.0, 9.0]]),
        ],
    )
    def test_forward(self, test_input, axis, keepdims, expected):
        f = Sum()
        actual = f.forward(Tensor(test_input), axis=axis, keepdims=keepdims)
        expected_var = Tensor(expected)
        assert actual.shape == expected_var.shape
        assert allclose(actual, expected_var)

    @pytest.mark.parametrize(
        "test_input_shape, axis, keepdims, dout, expected",
        [
            ((3,), None, False, 1.0, [1.0, 1.0, 1.0]),
            ((2, 3), None, False, 1.0, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            ((3, 2), 0, False, [1.0, 1.0], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            ((3, 2), 0, True, [[1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        ],
    )
    def test_backward(self, test_input_shape, axis, keepdims, dout, expected):
        test_input = Tensor(np.random.rand(*test_input_shape))
        f = Sum()
        f(test_input, axis=axis, keepdims=keepdims)
        actual = f.backward(Tensor(dout))
        expected_var = Tensor(expected)
        assert actual.shape == expected_var.shape
        assert allclose(actual, expected_var)

    @pytest.mark.parametrize(
        "test_input_shape, axis, keepdims, dout, expected",
        [
            ((3,), None, False, 1.0, [1.0, 1.0, 1.0]),
            ((2, 3), None, False, 1.0, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            ((3, 2), 0, False, [1.0, 1.0], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            ((3, 2), 0, True, [[1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        ],
    )
    def test_gradient_check(self, test_input_shape, axis, keepdims, dout, expected):
        test_input = Tensor(np.random.rand(*test_input_shape))
        f = Sum()
        f(test_input, axis=axis, keepdims=keepdims)
        actual = f.backward(Tensor(dout))

        expected = numerical_diff(f, test_input, axis=axis, keepdims=keepdims)
        assert actual.shape == expected.shape
        assert allclose(actual, expected)


def sum(x, axis=None, keepdims=False):
    return Sum()(x, axis=axis, keepdims=keepdims)


class TestGetItem:
    @pytest.mark.parametrize(
        "test_input, index, expected",
        [
            ([1.0, 2.0, 3.0], 0, 1.0),
            ([[1.0, 2.0], [3.0, 4.0]], (0, 1), 2.0),
            ([[1.0, 2.0], [3.0, 4.0]], -1, [3.0, 4.0]),
            ([[1.0, 2.0], [3.0, 4.0]], 1, [3.0, 4.0]),
            ([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], (1, slice(0, 2)), [3.0, 4.0]),
        ],
    )
    def test_forward(self, test_input, index, expected):
        f = GetItem()
        actual = f.forward(Tensor(test_input), index=index)
        expected_var = Tensor(expected)
        assert actual.shape == expected_var.shape
        assert allclose(actual, expected_var)

    @pytest.mark.parametrize(
        "test_input, index, expected",
        [
            ([1.0, 2.0, 3.0], 0, [1.0, 0.0, 0.0]),
            ([1.0, 2.0, 3.0], -1, [0.0, 0.0, 1.0]),
            ([[1.0, 2.0], [3.0, 4.0]], (0, 1), [[0.0, 1.0], [0.0, 0.0]]),
            ([[1.0, 2.0], [3.0, 4.0]], 1, [[0.0, 0.0], [1.0, 1.0]]),
            (
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
                (1, slice(0, 2)),
                [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ),
        ],
    )
    def test_backward(self, test_input, index, expected):
        test_input = Tensor(test_input)
        f = GetItem()
        y = f(test_input, index=index)

        dout = ones_like(y)
        actual = f.backward(dout)
        expected = Tensor(expected)
        assert actual.shape == expected.shape
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input, index",
        [
            ([1.0, 2.0, 3.0], 0),
            ([[1.0, 2.0], [3.0, 4.0]], (0, 1)),
            ([[1.0, 2.0], [3.0, 4.0]], -1),
            ([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], (1, slice(0, 2))),
        ],
    )
    def test_gradient_check(self, test_input, index):
        test_input = Tensor(test_input)
        f = GetItem()
        y = f(test_input, index=index)

        dout = ones_like(y)
        actual = f.backward(dout)

        expected = numerical_diff(f, test_input, index=index)
        assert actual.shape == expected.shape
        assert allclose(actual, expected)


class TestGetItemGrad:
    @pytest.mark.parametrize(
        "test_input, to_shape, index, expected",
        [
            (1.0, (3,), 0, [1.0, 0.0, 0.0]),
            (1.0, (3,), -1, [0.0, 0.0, 1.0]),
            (1.0, (2, 2), (0, 1), [[0.0, 1.0], [0.0, 0.0]]),
            ([1.0, 1.0], (2, 2), 1, [[0.0, 0.0], [1.0, 1.0]]),
            ([1.0, 1.0], (2, 3), (1, slice(0, 2)), [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        ],
    )
    def test_forward(self, test_input, to_shape, index, expected):
        f = GetItemGrad()
        actual = f.forward(Tensor(test_input), to_shape=to_shape, index=index)
        expected_var = Tensor(expected)
        assert actual.shape == expected_var.shape
        assert allclose(actual, expected_var)

    @pytest.mark.parametrize(
        "test_input, to_shape, index, expected",
        [
            ([5.0, 3.0, 7.0], (3,), 0, 5.0),
            ([5.0, 3.0, 7.0], (3,), -1, 7.0),
            ([[1.0, 2.0], [3.0, 4.0]], (2, 2), (0, 1), 2.0),
            ([[1.0, 2.0], [3.0, 4.0]], (2, 2), 1, [3.0, 4.0]),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3), (1, slice(0, 2)), [4.0, 5.0]),
        ],
    )
    def test_backward(self, test_input, to_shape, index, expected):
        # x must have the shape of to_shape[index] (the indexed output shape)
        x = zeros_like(Tensor(expected))
        f = GetItemGrad()
        f(x, to_shape=to_shape, index=index)

        actual = f.backward(Tensor(test_input))
        expected = Tensor(expected)
        assert actual.shape == expected.shape
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input, to_shape, index",
        [
            (5.0, (3,), 0),
            (5.0, (3,), -1),
            ([2.0, 3.0], (2, 2), 1),
            (2.0, (2, 2), (0, 1)),
            ([4.0, 5.0], (2, 3), (1, slice(0, 2))),
        ],
    )
    def test_gradient_check(self, test_input, to_shape, index):
        x = Tensor(test_input)
        f = GetItemGrad()
        y = f(x, to_shape=to_shape, index=index)

        dout = ones_like(y)
        actual = f.backward(dout)

        expected = numerical_diff(f, x, to_shape=to_shape, index=index)
        assert actual.shape == expected.shape
        assert allclose(actual, expected)
