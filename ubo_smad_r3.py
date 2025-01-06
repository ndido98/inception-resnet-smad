from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


WEIGHTS_URL = "https://miatbiolab.csr.unibo.it/wp-content/uploads/2023/inception-resnet-smad-4a96dd25.ckpt"
WEIGHTS_FILE = "./weights/inception-resnet-smad-4a96dd25.ckpt"

def _crop_face(image_rgb: np.ndarray, mtcnn: MTCNN) -> np.ndarray:
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected.")
    biggest_box = np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))
    box = boxes[biggest_box].astype(int)
    x1, y1, x2, y2 = (
        max(0, box[0]),
        max(0, box[1]),
        min(image_rgb.shape[1], box[2]),
        min(image_rgb.shape[0], box[3]),
    )
    cropped = image_rgb[y1:y2, x1:x2]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError("No face detected.")
    return cropped


def _preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    # Resize
    new_size = (299, 299)
    old_size = image_rgb.shape[:2]
    scale_factor = min(n / o for n, o in zip(new_size, old_size))
    rescaled = cv.resize(image_rgb, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    if rescaled.shape[0] == 0 or rescaled.shape[1] == 0:
        raise ValueError("Rescaling failed.")
    top_bottom, left_right = tuple(d - s for d, s in zip(new_size, rescaled.shape[:2]))
    top = top_bottom // 2
    bottom = top_bottom - top
    left = left_right // 2
    right = left_right - left
    resized = cv.copyMakeBorder(rescaled, top, bottom, left, right, cv.BORDER_CONSTANT, (0, 0, 0))
    # To float
    resized = resized.astype(np.float32) / 255.0
    # Normalize
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (resized - mean) / std
    # To tensor
    chw = torch.from_numpy(normalized.transpose((2, 0, 1)))
    if chw.ndim != 3:
        raise ValueError(f"Invalid image ndim: expected 3, got {chw.ndim}.")
    return chw


def get_prediction(
    document_bgr: np.ndarray | list[np.ndarray],
    device: str | torch.device = "cpu",
) -> float | list[float]:
    """
    Get the prediction score(s) for the given document image(s).
    If a lists of images is passed as input, the output will be a list of corresponding scores.

    :param document_bgr: The document image(s) in BGR format.
    :param device: The device to use for the prediction. Can be either a string representing the device or a torch.device object.
    :return: The prediction score(s).
    """

    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(document_bgr, np.ndarray):
        document_bgr = [document_bgr]
    # Download the Siamese weights or load them from file
    if Path(WEIGHTS_FILE).exists():
        state_dict = torch.load(WEIGHTS_FILE)
    else:
        state_dict = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", check_hash=True)
    # Load the SMAD classifier
    smad = InceptionResnetV1(
        pretrained=None,
        classify=True,
        num_classes=1,
        dropout_prob=0.6,
    ).eval()
    smad.load_state_dict(state_dict)
    smad = smad.to(device)
    # Load the MTCNN face detector
    mtcnn = MTCNN(select_largest=True, device=device)
    # Compute the prediction(s)
    with torch.no_grad():
        preprocessed = []
        for doc in document_bgr:
            document_rgb = cv.cvtColor(doc, cv.COLOR_BGR2RGB)
            document_face = _crop_face(document_rgb, mtcnn)
            preprocessed.append(_preprocess_image(document_face))
        batch = torch.stack(preprocessed)
        device_batch = batch.to(device)
        scores = smad(device_batch).squeeze()
        scores = torch.sigmoid_(scores).cpu().tolist()
        
        # Return the score(s) as a list 
        scores = [scores] if isinstance(scores, float) else scores 

        return scores if len(scores) > 1 else scores[0]
