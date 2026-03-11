import contextlib


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def using_backprop(value: bool):
    return using_config("enable_backprop", value)


def eval_mode():
    return using_config("train", False)
