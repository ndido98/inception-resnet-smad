# UBO-SMAD-R3: an Inception-ResNet-based model for Single-image Morphing Attack Detection

This is the repository that contains UBO-SMAD-R3, a standalone version of an Inception-ResNet V1, fine-tuned for Single-image Morphing Attack Detection (S-MAD).
This model was trained using the [Revelio framework](https://github.com/ndido98/revelio).


## Requirements

The required packages are present in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

The inception_resnet_smad package exposes a `get_prediction` function which, in its simplest form, takes in input a document image, and returns a morphing prediction.
0 means that the document image is bona fide, while 1 means that the document image is morphed.

```python
from ubo_smad_r3 import get_prediction
import cv2 as cv

# Load the document image
document = cv.imread("document.png")

# Get the prediction
prediction = get_prediction(document)
```

This function also allows the user to specify the device to use for the computation (i.e. CPU or GPU) with the optional `device` parameter. The default value is `cpu`.

```python
from ubo_smad_r3 import get_prediction
import cv2 as cv

# Load the document image
document = cv.imread("document.png")

# Get the prediction
prediction = get_prediction(document, device="cuda:0")
```

Finally, the function supports computing batched predictions, by passing a list containing the document images. The function will return a list of predictions.

```python
from ubo_smad_r3 import get_prediction
import cv2 as cv

# Load the document images
documents = [cv.imread("document1.png"), cv.imread("document2.png")]

# Get the predictions
predictions = get_prediction(documents, device="cuda:0")
```

## Acknowledgements

When using the code from this repository, please cite the following work:

```
@article{borghi2023revelio,
  title={Revelio: a Modular and Effective Framework for Reproducible Training and Evaluation of Morphing Attack Detectors},
  author={Borghi, Guido and Di Domenico, Nicol{\`o} and Franco, Annalisa and Ferrara, Matteo and Maltoni, Davide},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```
