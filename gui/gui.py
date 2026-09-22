import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model


# Get the project folder so the model paths work when running the GUI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONV2D_PATH = os.path.join(BASE_DIR, "models", "conv2d.keras")
DENSE_PATH = os.path.join(BASE_DIR, "models", "dense.keras")


# Load the models that have already been trained
conv2d_model = load_model(CONV2D_PATH)
dense_model = load_model(DENSE_PATH)


# Finds the sections of the image that contain digits
def find_digit_spans(profile, threshold=0, min_width=1):
    spans = []
    start = None

    for i, value in enumerate(profile):

        if value > threshold and start is None:
            start = i

        elif value <= threshold and start is not None:

            if i - start >= min_width:
                spans.append((start, i))

            start = None

    if start is not None and len(profile) - start >= min_width:
        spans.append((start, len(profile)))

    return spans


# Splits a number image into separate digits
def segment_image(image):
    image = image.convert("L")
    image_array = np.array(image)

    # The models use white digits on a black background
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Add up the pixels in each column to find where the digits are
    profile = image_array.sum(axis=0)

    spans = find_digit_spans(profile)

    crops = [
        image_array[:, start:end]
        for start, end in spans
    ]

    return crops


# Changes a digit into the format used by the models
def preprocess_digit(crop):
    image = Image.fromarray(crop)

    # MNIST images are 28x28
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32")

    # Change pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    return image_array


# Runs the selected model on the uploaded image
def predict_image(image, model):
    crops = segment_image(image)

    if len(crops) == 0:
        return ""

    processed = [
        preprocess_digit(crop)
        for crop in crops
    ]

    # Put all of the digits into one batch
    batch = np.array(processed)

    predictions = model.predict(batch, verbose=0)

    # Get the digit with the highest probability
    digits = predictions.argmax(axis=1)

    # Join the digits together for multi-digit numbers
    result = "".join(str(digit) for digit in digits)

    return result


# Opens the file picker and displays the selected image
def upload_image():
    global current_image

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        current_image = Image.open(file_path)

        display_image = current_image.copy()
        display_image.thumbnail((300, 200))

        photo = ImageTk.PhotoImage(display_image)

        image_label.config(image=photo)
        image_label.image = photo

        prediction_label.config(text="Prediction: ")

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Could not open image:\n{e}"
        )


# Removes the current image and prediction
def clear_image():
    global current_image

    current_image = None

    image_label.config(image="")
    image_label.image = None

    prediction_label.config(text="Prediction: ")


# Runs the prediction when the button is pressed
def predict():
    if current_image is None:
        messagebox.showwarning(
            "No image",
            "Please upload an image first."
        )
        return

    if model_choice.get() == "Conv2D":
        model = conv2d_model
    else:
        model = dense_model

    try:
        result = predict_image(current_image, model)

        if result == "":
            prediction_label.config(
                text="Prediction: No digits detected"
            )
        else:
            prediction_label.config(
                text=f"Prediction: {result}"
            )

    except Exception as e:
        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# Set up the main window
root = tk.Tk()

root.title("MNIST Digit Recognition")
root.geometry("500x500")


title_label = tk.Label(
    root,
    text="MNIST Digit Recognition",
    font=("Arial", 20)
)

title_label.pack(pady=20)


# Where the uploaded image is shown
image_label = tk.Label(root)

image_label.pack(pady=10)


# Model selection
model_choice = tk.StringVar(value="Conv2D")

model_label = tk.Label(
    root,
    text="Select Model:"
)

model_label.pack()


model_menu = tk.OptionMenu(
    root,
    model_choice,
    "Conv2D",
    "Dense"
)

model_menu.pack(pady=5)


upload_button = tk.Button(
    root,
    text="Upload Image",
    command=upload_image
)

upload_button.pack(pady=5)


clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_image
)

clear_button.pack(pady=5)


predict_button = tk.Button(
    root,
    text="Predict",
    command=predict
)

predict_button.pack(pady=10)


prediction_label = tk.Label(
    root,
    text="Prediction: ",
    font=("Arial", 16)
)

prediction_label.pack(pady=20)


current_image = None

# Start the GUI
root.mainloop()

