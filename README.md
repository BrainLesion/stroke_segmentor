# Stroke Segmentor

[![Python Versions](https://img.shields.io/pypi/pyversions/stroke_segmentor)](https://pypi.org/project/stroke_segmentor/)
[![Stable Version](https://img.shields.io/pypi/v/stroke_segmentor?label=stable)](https://pypi.python.org/pypi/stroke_segmentor/)
[![Documentation Status](https://readthedocs.org/projects/stroke_segmentor/badge/?version=latest)](http://stroke_segmentor.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/stroke_segmentor/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/stroke_segmentor/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/stroke_segmentor/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/stroke_segmentor) -->

State-of-the-art ischemic stroke lesion segmentation in MRI


## Installation

With a Python 3.9+ environment, you can install `stroke_segmentor` directly from [PyPI](https://pypi.org/project/stroke_segmentor/):

```bash
pip install stroke_segmentor
```


## Use Cases and Tutorials

A minimal example to create a segmentation could look like this:

```python
from stroke_segmentor.inferer import Inferer

inferer = Inferer()
pred = inferer.infer(
    adc_path="path/to/adc.nii.gz",
    dwi_path="path/to/dwi.nii.gz",
    segmentation_path="seg.nii.gz", # optional. the numpy array is always returned for direct usage
)
```

## Logging
By default our package is silent, although we do use logging under the hood.  
If you want, you can enable it like this:
```py
from loguru import logger

logger.enable("stroke_segmentor")
```


## Citation

> [!IMPORTANT]
> `stroke_segmentor` is based on [DeepISLES](https://github.com/ezequieldlrosa/DeepIsles) and offers its NVAUTO algorithm
> as part of the [BrainLesion](https://github.com/BrainLesion) suite.  
> Please cite all relevant manuscripts!  


### DeepISLES
> de la Rosa, Ezequiel, et al. "DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES'22 challenge." Nature Communications 16.1 (2025): 7357.
```

@article{de2025deepisles,
  title={DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES'22 challenge},
  author={de la Rosa, Ezequiel and Reyes, Mauricio and Liew, Sook-Lei and Hutton, Alexandre and Wiest, Roland and Kaesmacher, Johannes and Hanning, Uta and Hakim, Arsany and Zubal, Richard and Valenzuela, Waldo and others},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={7357},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

### BrainLesion Suite
> Kofler, F., Rosier, M., Astaraki, M., Möller, H., Mekki, I. I., Buchner, J. A., ... & Menze, B. (2025). BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis. arXiv preprint arXiv:2507.09036.
```

@article{kofler2025brainlesion,
  title={BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis},
  author={Kofler, Florian and Rosier, Marcel and Astaraki, Mehdi and M{\"o}ller, Hendrik and Mekki, Ilhem Isra and Buchner, Josef A and Schmick, Anton and Pfiffer, Arianna and Oswald, Eva and Zimmer, Lucas and others},
  journal={arXiv preprint arXiv:2507.09036},
  year={2025}
}
```

### NVAUTO Algorithm
>Siddique, M. M. R., Yang, D., He, Y., Xu, D., & Myronenko, A. (2022). Automated ischemic stroke lesion segmentation from 3D MRI. arXiv preprint arXiv:2209.09546.
```
@article{siddique2022automated,
  title={Automated ischemic stroke lesion segmentation from 3D MRI},
  author={Siddique, Md Mahfuzur Rahman and Yang, Dong and He, Yufan and Xu, Daguang and Myronenko, Andriy},
  journal={arXiv preprint arXiv:2209.09546},
  year={2022}
}
```


## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/stroke_segmentor/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
