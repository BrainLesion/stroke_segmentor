# Stroke Segmentor

[![Python Versions](https://img.shields.io/pypi/pyversions/stroke_segmentor)](https://pypi.org/project/stroke_segmentor/)
[![Stable Version](https://img.shields.io/pypi/v/stroke_segmentor?label=stable)](https://pypi.python.org/pypi/stroke_segmentor/)
[![Documentation Status](https://readthedocs.org/projects/stroke_segmentor/badge/?version=latest)](http://stroke_segmentor.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/stroke_segmentor/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/stroke_segmentor/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/stroke_segmentor/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/stroke_segmentor) -->

State-of-the-art ischemic stroke lesion segmentation in MRI


## Installation

With a Python 3.8+ environment, you can install `stroke_segmentor` directly from [PyPI](https://pypi.org/project/stroke_segmentor/):

```bash
pip install stroke_segmentor
```


## Use Cases and Tutorials

A minimal example to create a segmentation could look like this:

```python
from stroke_segmentor.inferer import Inferer

inferer = Inferer()
pred = inferer.infer(
    dwi_path="path/to/dwi.nii.gz",
    adc_path="path/to/adc.nii.gz",
    segmentation_path="seg.nii.gz", # optional. the numpy array is always returned for direct usage
)
```

## Citation

> [!IMPORTANT]
> `stroke_segmentor` is part of the [BrainLesion](https://github.com/BrainLesion) suite.  
> Please cite it support the development!  
> https://github.com/BrainLesion#-citing-brainlesion-suite

 This project is based on [DeepISLES](https://github.com/ezequieldlrosa/DeepIsles) and offers its NVAUTO algorithm.
 Please cite the [original manuscript](https://github.com/ezequieldlrosa/DeepIsles?tab=readme-ov-file#citations).

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/stroke_segmentor/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
