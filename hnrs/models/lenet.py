import os
import keras
import numpy as np
import tensorflow as tf
from hnrs.preprocessing import load_preprocessed_mnist

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
MODEL_PATH = os.path.join(_REPO_ROOT, "models", "lenet.keras")

_model = None

def build_model():
    keras.utils.set_random_seed(42) 
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model

def train_model(model, train_images, train_labels, epochs=5):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1)

    model.save(MODEL_PATH)

    return history

def load_and_predict(image):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    prediction = _model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    return predicted_label, confidence

if __name__ == "__main__":
    x_train, y_train, _, x_test, y_test, _ = load_preprocessed_mnist()

    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)

    model = build_model()
    history = train_model(model, x_train, y_train, epochs=5)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")