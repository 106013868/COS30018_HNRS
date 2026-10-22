"""
transfer.py
transfer learning for any saved model on any mix of datasets from preprocessing.DATASETS
build a training mix, fine tune a saved model on it, then compare models on each test set
"""

import numpy as np
import keras
import tensorflow as tf

from hnrs.preprocessing import load_dataset


def match_input(model, x):
    """
    adds the channel dimension if the model expects (28, 28, 1) images, e.g. lenet
    """
    if len(model.input_shape) == 4 and x.ndim == 3:
        return x[..., None]
    return x


def build_mix(names, per_class=None, seed=42):
    """
    In: dataset names, training images per class from each dataset
    Out: shuffled training images and labels
    """
    rng = np.random.default_rng(seed)
    data = {name: load_dataset(name)[:2] for name in names}

    # default to the average class size of the smallest dataset
    # so each dataset adds about the same amount per class and no style dominates
    if per_class is None:
        per_class = min(int(np.bincount(y)[np.bincount(y) > 0].mean()) for _, y in data.values())

    xs, ys = [], []
    for x, y in data.values():
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            pick = rng.choice(idx, min(per_class, len(idx)), replace=False)
            xs.append(x[pick])
            ys.append(y[pick])

    # shuffle so batches contain every dataset and class
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = rng.permutation(len(x))
    return x[order], y[order]


def fine_tune(base_path, x, y, num_classes=None, epochs=5, save_path=None, seed=42):
    """
    In: path of a trained model, training images and labels, number of output classes
    Out: fine tuned model and training history, saved to save_path if given
    """
    keras.utils.set_random_seed(seed)
    model = tf.keras.models.load_model(base_path)

    # conv layers detect strokes and shapes, which carry over to new handwriting, so freeze them
    # models without conv layers (dense) freeze their first dense layer instead
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if conv_layers:
        for layer in conv_layers:
            layer.trainable = False
    else:
        first_dense = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
        first_dense.trainable = False

    # new classes (e.g. symbols) need a bigger output layer
    # defaults to enough outputs for every label in y
    current = model.layers[-1].units
    if num_classes is None:
        num_classes = max(current, int(y.max()) + 1)
    if num_classes != current:
        model.pop()
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name=f"output_{num_classes}"))

    # freezing only takes effect after compiling again
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(match_input(model, x), y, epochs=epochs, validation_split=0.1)

    if save_path:
        model.save(save_path)

    return model, history


def evaluate(model, names):
    """
    In: model, dataset names
    Out: dict of dataset name -> accuracy on its test set
    only counts classes the model can predict, e.g. a digits only model skips symbols
    """
    outputs = model.layers[-1].units
    results = {}
    for name in names:
        _, _, x_test, y_test = load_dataset(name)
        keep = y_test < outputs
        pred = model.predict(match_input(model, x_test[keep]), verbose=0).argmax(axis=1)
        results[name] = (pred == y_test[keep]).mean()
    return results


def compare(models, names):
    """
    In: dict of label -> model, dataset names
    Out: prints a table of test accuracy per model and dataset, returns the results
    """
    results = {label: evaluate(model, names) for label, model in models.items()}

    print(f"{'':22s}" + "".join(f"{name:>9s}" for name in names))
    for label, accs in results.items():
        print(f"{label:22s}" + "".join(f"{accs[name]:9.1%}" for name in names))

    return results