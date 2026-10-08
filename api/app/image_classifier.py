from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps
from torchvision import transforms

from app.cnn_model import AssignmentCNN, CIFAR10_CLASSES, CNN_INPUT_SIZE


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.pth"
)
MODEL_PATH = Path(os.getenv("CNN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

preprocess = transforms.Compose(
    [
        transforms.Resize(CNN_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def _build_model() -> tuple[AssignmentCNN, bool, str | None]:
    model = AssignmentCNN(num_classes=len(CIFAR10_CLASSES)).to(DEVICE)
    if not MODEL_PATH.exists():
        model.eval()
        return model, False, f"Model checkpoint not found at {MODEL_PATH}"

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, True, None


model, MODEL_LOADED, MODEL_WARNING = _build_model()


async def classify_upload(file: UploadFile) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image file is empty.")

    try:
        image = Image.open(BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not read the image file.") from exc

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_index = int(predicted_idx.item())
    class_probabilities = {
        class_name: float(probabilities[i].item())
        for i, class_name in enumerate(CIFAR10_CLASSES)
    }

    return {
        "filename": file.filename,
        "predicted_class": CIFAR10_CLASSES[predicted_index],
        "predicted_index": predicted_index,
        "confidence": float(confidence.item()),
        "model_loaded": MODEL_LOADED,
        "model_path": str(MODEL_PATH),
        "warning": MODEL_WARNING,
        "probabilities": class_probabilities,
    }
