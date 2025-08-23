from pathlib import Path

import numpy as np
import torch

from stroke_segmentor.model_handler import ModelHandler


def test_modelhandler_init_cpu(mocker):
    """Check that ModelHandler initializes on CPU if force_cpu=True."""
    mocker.patch(
        "stroke_segmentor.model_handler.fetch_weights",
        return_value=Path("/fake/weights"),
    )
    handler = ModelHandler(force_cpu=True)

    assert handler.device.type == "cpu"
    assert len(handler.checkpoints) == 15
    assert all(str(c).startswith("/fake/weights/model") for c in handler.checkpoints)
