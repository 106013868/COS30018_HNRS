import os
import sys

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow Python to find the hnrs package
sys.path.insert(0, BASE_DIR)
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageTk
from hnrs.preprocess_image import preprocess_image
from hnrs.model_loader import load_and_predict
from hnrs.segmentation import split_characters
from hnrs.arithmetic import calculate
from hnrs.compose_numbers import load_digit_bank, generate_number
# Get the project folder so the model paths work when running the GUI

# Shows an image scaled up or down to fit the same 300x100 box, so uploads and composed numbers match
def show_image(img):
    img = ImageOps.exif_transpose(img)  # phone photos can be stored sideways
    scale = min(300 / img.width, 100 / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    display_image = img.resize(size, Image.NEAREST if scale > 1 else Image.LANCZOS)

    photo = ImageTk.PhotoImage(display_image)
    image_label.config(image=photo)
    image_label.image = photo

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

        show_image(current_image)

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

    try:
        characters = split_characters(current_image)
        if not characters:
            raise ValueError("No characters found in the image")

        results = [load_and_predict(preprocess_image(c), model_name=selected_model) for c in characters]
        text = "".join(label for label, _ in results)
        confidence = min(conf for _, conf in results)

        if any(ch in "+-*/" for ch in text):
            try:
                text += f" = {calculate(text):g}"
            except ValueError:
                text += "  (can't calculate)"

        prediction_label.config(
            text=f"Prediction: {text}\nConfidence: {confidence:.1%}"
        )

    except Exception as e:
        messagebox.showerror(
            "Prediction Error",
            f"Could not predict the image:\n{e}"
        )

# Builds one random number from the digit images and loads it ready to predict
def compose_number():
    global current_image

    try:
        image, label = generate_number(load_digit_bank())
        current_image = Image.fromarray(image)

        show_image(current_image)

    except Exception as e:
        messagebox.showerror(
            "Compose Error",
            f"Could not compose a number:\n{e}"
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
root.title("Handwritten Digit Recognition")
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
    text="Handwritten Digit Recognition",
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
model_choice = tk.StringVar(value="LeNet")

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
    "LeNet"
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
    if sys.platform == "darwin":
        fg = TEXT
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


# Upload and compose buttons, side by side
button_row = tk.Frame(root, bg=BG)
button_row.pack(pady=6)

upload_button = create_button(
    button_row, "Upload", upload_image,
    BLUE, active_bg="#2563EB"
)
upload_button.config(width=1, padx=4)
upload_button.place(relx=0, rely=0, relwidth=0.5, relheight=1, width=-4)

compose_button = create_button(
    button_row, "Compose", compose_number,
    WHITE, fg=TEXT, active_bg=GRAY
)
compose_button.config(width=1, padx=4)
compose_button.place(relx=0.5, x=4, rely=0, relwidth=0.5, relheight=1, width=-4)

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

root.update_idletasks()
button_row.config(width=clear_button.winfo_reqwidth(), height=clear_button.winfo_reqheight())

# Start the GUI
root.mainloop()

