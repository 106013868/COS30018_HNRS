import os
import numpy as np
import tensorflow as tf
from hnrs.preprocessing import CLASSES

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MODULE_DIR)

MODEL_PATHS = {
    "LeNet": os.path.join(_REPO_ROOT, "models", "lenet_symbols.keras"),
    "Conv2D": os.path.join(_REPO_ROOT, "models", "conv2d.keras"),
    "Dense": os.path.join(_REPO_ROOT, "models", "dense.keras"),
}

_loaded_models = {}


def get_model(model_name):
    """Load a model once and reuse it."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name not in _loaded_models:
        model_path = MODEL_PATHS[model_name]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        _loaded_models[model_name] = tf.keras.models.load_model(
            model_path
        )

    return _loaded_models[model_name]


def load_and_predict(image, model_name="LeNet"):
    model = get_model(model_name)
    image = np.asarray(image, dtype=np.float32)

    prediction = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )

    predicted_label = CLASSES[int(np.argmax(prediction[0]))]
    confidence = float(np.max(prediction[0]))

    return predicted_label, confidence