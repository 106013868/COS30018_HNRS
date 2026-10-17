import os
import keras
import numpy as np
import tensorflow as tf
from hnrs.preprocessing import load_preprocessed_mnist

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
MODEL_PATH = os.path.join(_REPO_ROOT, "models", "lenet.keras")
FINETUNED_PATH = os.path.join(_REPO_ROOT, "models", "lenet_swiss.keras")
MIXED_PATH = os.path.join(_REPO_ROOT, "models", "lenet_mixed.keras")
SYMBOLS_PATH = os.path.join(_REPO_ROOT, "models", "lenet_symbols.keras")

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

def fine_tune_model(train_images, train_labels, epochs=5, save_path=FINETUNED_PATH):
    """Transfer learning: start from the MNIST-trained LeNet and adapt it to new handwriting.

    The convolution layers have already learned to detect strokes and shapes,
    so they are frozen. Only the dense layers are retrained on the new data.
    Set save_path=None to skip saving (e.g. for quick experiments).
    """
    keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(MODEL_PATH)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = False

    # Freezing layers only takes effect after compiling again
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1)

    if save_path:
        model.save(save_path)

    return model, history

def fine_tune_symbol_model(train_images, train_labels, num_classes, epochs=5, save_path=SYMBOLS_PATH):
    """
    In: training images and labels for digits and symbols, number of classes
    Out: model that predicts digits and arithmetic symbols, training history
    """
    keras.utils.set_random_seed(42)

    # start from the mixed model since it already handles real handwriting
    model = tf.keras.models.load_model(MIXED_PATH)

    # freeze conv layers, strokes and curves look the same in digits and symbols
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = False

    # swap the 10 digit output layer for one with an output per class
    model.pop()
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='symbol_output'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1)

    if save_path:
        model.save(save_path)

    return model, history

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