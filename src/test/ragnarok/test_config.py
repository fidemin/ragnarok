from src.main.ragnarok.core.config import using_config, Config


def test_using_config(mocker):
    MockConfig = mocker.patch('src.main.ragnarok.core.config.Config')
    MockConfig.mock_enable_backprop = True

    # Use the context manager to change the value
    with using_config('mock_enable_backprop', False):
        assert MockConfig.mock_enable_backprop is False

    # Check that the value has been reset to the original value
    assert MockConfig.mock_enable_backprop


def test_using_backprop():
    with using_config('enable_backprop', False):
        assert Config.enable_backprop is False

    assert Config.enable_backprop
