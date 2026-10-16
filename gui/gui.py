import os
import sys

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow Python to find the hnrs package
sys.path.insert(0, BASE_DIR)
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from hnrs.preprocess_image import preprocess_image
from hnrs.models.lenet import load_and_predict
# Get the project folder so the model paths work when running the GUI

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
        display_image.thumbnail((100, 100))

        photo = ImageTk.PhotoImage(display_image)

        image_label.config(image=photo)
        image_label.image = photo

        prediction_label.config(
            text="Predicted Digit: -\nConfidence: -"
        )

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

    selected_model = model_choice.get()

    # Only LeNet prediction is implemented for now
    if selected_model != "Lenet":
        prediction_label.config(
            text=f"{selected_model} prediction is not implemented yet."
        )
        return

    try:
        processed_image = preprocess_image(current_image)
        label, confidence = load_and_predict(processed_image)

        prediction_label.config(
            text=f"Predicted Digit: {label}\nConfidence: {confidence:.1%}"
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

    prediction_label.config(text="Predicted Digit: -\nConfidence: -" )

# Set up the main window
root = tk.Tk()
root.title("MNIST Digit Recognition")
root.geometry("430x530")
root.configure(bg="#F4F7FB")

BG = "#F7F8FA"
NAVY = "#172B4D"
BLUE = "#527DB8"
GREEN = "#579B7B"
TEXT = "#374151"
MUTED = "#64748B"
WHITE = "#FFFFFF"
GRAY = "#CBD5E1"

# Smaller, less prominent heading
title_label = tk.Label(
    root,
    text="MNIST Digit Recognition",
    font=("Arial", 20),
    bg=BG,
    fg=NAVY
)
title_label.pack(pady=(20, 4))

subtitle_label = tk.Label(
    root,
    text="Upload a handwritten digit to get a prediction",
    font=("Arial", 10),
    bg=BG,
    fg=MUTED
)
subtitle_label.pack(pady=(0, 16))

image_label = tk.Label(
    root,
    bg=BG,
    fg=MUTED,
    text="No image uploaded",
    font=("Arial", 10)
)
image_label.pack(pady=10)


# Model selection
model_choice = tk.StringVar(value="Lenet")

model_label = tk.Label(
    root,
    text="Select Model",
    font=("Arial", 11, "bold"),
    bg=BG,
    fg=TEXT
)
model_label.pack(pady=(10, 5), anchor="center")

model_menu = tk.OptionMenu(
    root,
    model_choice,
    "Conv2D",
    "Dense",
    "Lenet"
)

# Style the dropdown button
model_menu.config(
    font=("Arial", 11),
    bg=WHITE,
    fg=TEXT,
    activebackground="#E2E8F0",
    activeforeground=TEXT,
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#CBD5E1",
    cursor="hand2",
    width=16
)

# Style the options inside the dropdown
model_menu["menu"].config(
    font=("Arial", 11),
    bg=WHITE,
    fg=TEXT,
    activebackground="#DBEAFE",
    activeforeground=NAVY,
    bd=0,
    relief="flat"
)

model_menu.pack(pady=(0, 12), anchor="center")


# Shared button styling
def create_button(parent, text, command, bg, fg="white",
                  active_bg=None, border="#CBD5E1"):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Arial", 11),
        bg=bg,
        fg=fg,
        activebackground=active_bg or bg,
        activeforeground=fg,
        relief="flat",
        bd=1,
        highlightthickness=0,
        highlightbackground=border,
        highlightcolor=border,
        cursor="hand2",
        padx=16,
        pady=7,
        width=18
    )


# Upload button
upload_button = create_button(
    root, "Upload Image", upload_image,
    BLUE, active_bg="#2563EB"
)
upload_button.pack(pady=6)

# Clear button
clear_button = create_button(
    root, "Clear", clear_image,
    WHITE, fg=TEXT, active_bg=GRAY
)
clear_button.pack(pady=6)

# Predict button
predict_button = create_button(
    root, "Predict", predict,
    WHITE, fg=TEXT, active_bg=GRAY
)
predict_button.pack(pady=6)

prediction_label = tk.Label(
    root,
    text="Predicted Digit: -\nConfidence: -",
    font=("Arial", 11),
    bg=BG,
    fg=TEXT,
    padx=15,
    pady=12
)
prediction_label.pack(
    fill="x",
    padx=60,
    pady=(12, 20)
)

current_image = None

# Start the GUI
root.mainloop()

