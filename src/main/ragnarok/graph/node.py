from typing import Optional


class DotVariableNode:
    def __init__(
        self, id_, *, name: str = "", shape: Optional[tuple] = None, dtype: str = None
    ):
        self._id = id_
        self._name = name
        self._shape = shape
        self._dtype = dtype

    def __str__(self):
        return self.to_str(verbose=True)

    def __repr__(self):
        return self.to_str(verbose=True)

    def to_str(self, *, verbose=False):
        str_format = '{} [label="{}", color=orange, style=filled]'
        name = self._name

        if verbose:
            label_subs = []
            if not self._name:
                pass
            elif self._name and (self._shape or self._dtype):
                name += ":"
                label_subs.append(name)
            else:
                label_subs.append(name)

            if self._shape is not None:
                label_subs.append(str(self._shape))

            if self._dtype is not None:
                label_subs.append(str(self._dtype))

            label = " ".join(label_subs)
        else:
            label = self._name

        formatted = str_format.format(self._id, label)
        return formatted

    def draw(self, verbose=False):
        return self.to_str(verbose=verbose)


class DotFunctionNode:
    def __init__(self, id_, name: str, *, input_ids: list[int], output_ids: list[int]):
        self._id = id_
        self._name = name
        self._input_ids = input_ids
        self._output_ids = output_ids

    def to_str(self):
        str_format = '{} [label="{}", color=lightblue, style=filled, shape=box]'
        formatted = str_format.format(self._id, self._name)
        return formatted

    def draw(self):
        str_list = [self.to_str()]

        edge_format = "{} -> {}"
        for input_id in self._input_ids:
            str_list.append(edge_format.format(input_id, self._id))
        for output_id in self._output_ids:
            str_list.append(edge_format.format(self._id, output_id))

        return "\n".join(str_list)

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return self.to_str()
