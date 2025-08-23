from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import SimpleITK
from numpy.typing import NDArray

from stroke_segmentor.model_handler import ModelHandler


class Inferer:
    """
    Handles inference for stroke segmentation using DWI and ADC images.
    This class wraps a model handler to perform inference on given images and can optionally
    save the resulting segmentation mask to disk.
    """

    def __init__(
        self,
        force_cpu: bool = False,
    ):
        self.model_handler = ModelHandler(force_cpu=force_cpu)

    def _save(
        self,
        dwi_path: Union[str, Path],
        prediction: NDArray,
        segmentation_path: Union[str, Path],
    ) -> None:
        """Save the prediction as a SimpleITK image.

        Args:
            dwi_path (str | Path): Path to the DWI image.
            prediction (NDArray): The predicted segmentation mask.
            segmentation_path (str | Path): Path to save the segmentation mask.

        Returns:
            None
        """
        dwi_image = SimpleITK.ReadImage(str(dwi_path))

        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = (
            dwi_image.GetOrigin(),
            dwi_image.GetSpacing(),
            dwi_image.GetDirection(),
        )

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction.astype(np.uint8))
        output_image.SetOrigin(origin)
        output_image.SetSpacing(spacing)
        output_image.SetDirection(direction)

        Path(segmentation_path).parent.mkdir(parents=True, exist_ok=True)
        SimpleITK.WriteImage(output_image, str(segmentation_path))

    def infer(
        self,
        adc_path: Union[str, Path],
        dwi_path: Union[str, Path],
        segmentation_path: Optional[Union[str, Path]] = None,
    ) -> NDArray:
        """Run inference on the provided ADC and DWI images.

        Args:
            adc_path (str | Path): Path to the ADC image.
            dwi_path (str | Path): Path to the DWI image.
            segmentation_path (Optional[str | Path], optional): Path to save the segmentation mask. Defaults to None.

        Returns:
            NDArray: The predicted segmentation mask.
        """

        prediction = self.model_handler.infer(
            adc_path=adc_path,
            dwi_path=dwi_path,
        )
        if segmentation_path:
            self._save(
                dwi_path=dwi_path,
                prediction=prediction,
                segmentation_path=segmentation_path,
            )
        return prediction
