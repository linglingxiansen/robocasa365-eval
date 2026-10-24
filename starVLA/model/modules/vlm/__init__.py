def get_vlm_model(config):
    """This evaluation snapshot supports Qwen3.5, independent of directory name."""
    from .QWen3_5 import _QWen3_5_VL_Interface

    return _QWen3_5_VL_Interface(config)
