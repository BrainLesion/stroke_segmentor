from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.post.dictionary import Invertd
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.utility.dictionary import (
    CastToTyped,
    EnsureChannelFirstd,
    EnsureTyped,
)
from numpy.typing import NDArray
from torch.amp.autocast_mode import autocast

from stroke_segmentor.zenodo import fetch_weights


class ModelHandler:
    """Class for model loading, inference and post processing"""

    def __init__(self, force_cpu: bool = False):
        """Initialize the ModelHandler class.

        Args:
            force_cpu (bool): Whether to force the use of CPU for inference.

        Returns:
            ModelHandler: ModelHandler instance.
        """

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        )
        # get location of model weights
        self.model_weights_folder = fetch_weights()
        self.checkpoints = [
            self.model_weights_folder / f"model{i}.ts" for i in range(15)
        ]

        self.transforms = self._get_transforms()

    def _get_transforms(self) -> Compose:
        """Get the transforms to be applied to the input data.

        Returns:
            Compose: Composed transforms.
        """
        load_keys = ["image"]

        return Compose(
            [
                LoadImaged(keys=load_keys),
                EnsureChannelFirstd(keys=load_keys),
                CastToTyped(keys=["image"], dtype=np.float32),
                EnsureTyped(keys=load_keys, data_type="tensor"),
                Spacingd(keys=["image"], pixdim=[1, 1, 1], mode=["bilinear"]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    def _get_data_loader(
        self,
        adc_path: Union[str, Path],
        dwi_path: Union[str, Path],
    ) -> DataLoader:
        """Get the data loader for the input data.

        Args:
            adc_path (str | Path): Path to the ADC image.
            dwi_path (str | Path): Path to the DWI image.

        Returns:
            DataLoader: DataLoader for the input data.
        """

        files = [{"image": [adc_path, dwi_path]}]

        ds = Dataset(
            data=files,
            transform=self.transforms,
        )
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=None,
        )

    def infer(
        self,
        adc_path: Union[str, Path],
        dwi_path: Union[str, Path],
    ) -> NDArray:
        """Run inference on the provided ADC and DWI images.

        Args:
            adc_path (str | Path): Path to the ADC image.
            dwi_path (str | Path): Path to the DWI image.

        Returns:
            NDArray: The predicted segmentation mask.
        """

        dataloader = self._get_data_loader(adc_path, dwi_path)
        model_inferer = SlidingWindowInferer(
            roi_size=[192, 192, 128],
            overlap=0.625,
            mode="gaussian",
            cache_roi_weight_map=True,
            sw_batch_size=2,
        )

        with torch.no_grad():
            batch_data: Dict[str, Any] = next(iter(dataloader))
            image = batch_data["image"].to(self.device)

            all_probs = []
            for checkpoint in self.checkpoints:

                model = torch.jit.load(checkpoint).to(self.device)
                model.eval()

                with (
                    autocast("cuda", enabled=True)
                    if self.device.type == "cuda"
                    else nullcontext()
                ):
                    logits = model_inferer(inputs=image, network=model)

                probs = torch.softmax(logits.float(), dim=1)

                batch_data["pred"] = probs
                inverter = Invertd(
                    keys="pred",
                    transform=self.transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                )
                probs = [
                    inverter(x)["pred"] for x in decollate_batch(batch_data)
                ]  # invert resampling if any
                probs = torch.stack(probs, dim=0)

                all_probs.append(probs.cpu())

            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            labels = torch.argmax(avg_probs, dim=1).cpu().numpy()

            prediction = labels[0].copy()

            prediction = prediction.transpose((2, 1, 0))

            return prediction.astype(np.int8)
