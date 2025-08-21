import numpy as np
import pytest
from pathlib import Path

import SimpleITK
from stroke_segmentor.inferer import Inferer


@pytest.fixture
def mock_model_handler(mocker):
    """Fixture to mock ModelHandler inside Inferer."""
    mock = mocker.patch("stroke_segmentor.inferer.ModelHandler")
    instance = mock.return_value
    instance.infer.return_value = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    return instance


def test_infer_without_save(mock_model_handler):
    inferer = Inferer(force_cpu=True)

    # Run inference
    adc = "fake_adc.nii.gz"
    dwi = "fake_dwi.nii.gz"
    result = inferer.infer(adc_path=adc, dwi_path=dwi)

    # Assertions
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=np.uint8))
    mock_model_handler.infer.assert_called_once_with(adc_path=adc, dwi_path=dwi)


def test_infer_with_save(tmp_path, mocker, mock_model_handler):
    inferer = Inferer(force_cpu=False)

    adc = "adc_fake.nii.gz"
    dwi = "dwi_fake.nii.gz"
    seg_out = tmp_path / "segmentation.nii.gz"

    # Mock SimpleITK functions
    mock_image = mocker.Mock()
    mock_image.GetOrigin.return_value = (0, 0, 0)
    mock_image.GetSpacing.return_value = (1, 1, 1)
    mock_image.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    mocker.patch(
        "stroke_segmentor.inferer.SimpleITK.ReadImage", return_value=mock_image
    )
    mocker.patch(
        "stroke_segmentor.inferer.SimpleITK.GetImageFromArray",
        return_value=mocker.Mock(),
    )
    write_mock = mocker.patch("stroke_segmentor.inferer.SimpleITK.WriteImage")

    # Run inference with save path
    result = inferer.infer(adc_path=adc, dwi_path=dwi, segmentation_path=seg_out)

    # Assertions
    assert isinstance(result, np.ndarray)
    write_mock.assert_called_once()
    # Check if file got written to correct path
    args, kwargs = write_mock.call_args
    assert str(seg_out) in args or kwargs.values()
