import os
import sys

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow Python to find the hnrs package
sys.path.insert(0, BASE_DIR)
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from hnrs.preprocess_image import preprocess_image
from hnrs.models.lenet import load_and_predict
# Get the project folder so the model paths work when running the GUI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV2D_PATH = os.path.join(BASE_DIR, "models", "conv2d.keras")
DENSE_PATH = os.path.join(BASE_DIR, "models", "dense.keras")

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


# Predict the digit in the uploaded image
def predict():
    if current_image is None:
        messagebox.showwarning(
            "No Image",
            "Please upload an image first."
        )
        return

    try:
        # Preprocess the uploaded image for the model
        processed_image = preprocess_image(current_image)

        # Get the predicted digit and confidence
        label, confidence = load_and_predict(processed_image)

        # Display the prediction and confidence
        prediction_label.config(
            text=f"Prediction: {label}\nConfidence: {confidence:.2%}"
        )

    except Exception as e:
        messagebox.showerror(
            "Prediction Error",
            f"Could not predict the image:\n{e}"
        )

# Removes the current image 
def clear_image():
    global current_image

    current_image = None

    image_label.config(image="")
    image_label.image = None

    prediction_label.config(text="Prediction: ")

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
    "Dense",
    "Lenet"
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

