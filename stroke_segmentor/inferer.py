from pathlib import Path
from typing import Optional

import SimpleITK
import numpy as np
from stroke_segmentor.model_handler import ModelHandler
from numpy.typing import NDArray


class Inferer:

    def __init__(
        self,
    ):
        self.model_weights_folder = ""
        self.device = "cuda"
        self.model_handler = ModelHandler(device=self.device)

    def _save(
        self,
        dwi_path: str | Path,
        prediction: NDArray,
        segmentation_path: str | Path,
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
        adc_path: str | Path,
        dwi_path: str | Path,
        segmentation_path: Optional[str | Path] = None,
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
