# HNRS: Handwritten Number Recognition System

COS30018 Intelligent Systems, Option B. Recognises handwritten numbers and arithmetic expressions (+ - × ÷ ()) from an image, and calculates the result.

Team: Tim Johnson (106013868), Amanuial Ashagrie (105871858), MD Rakibul Islam (105208278), Oliver Brand (105350085)

See the project report, section 3, for the full installation and user guide.

## Setup

Python 3.12.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the GUI

```
python gui/gui.py
```

Upload an image, or press Compose to build a random number from a set of digits. Choose a model and press Predict. The trained models are in `models/`, so no data or training is needed to run the GUI. The first Compose downloads MNIST (internet needed).

## Data (only needed for the notebooks)

MNIST downloads automatically. The other datasets are not included. Download them from Kaggle and unzip them into `data/`:

| Dataset | Folder |
|---|---|
| Handwritten Digits 0 - 9 | `data/swiss_digits/` (folders 0 to 9) |
| Handwritten Math Symbols | `data/math_symbols/` (folders 0 to 9, add, sub, mul, div) |
| Basic arithmetics dataset | copy only `left_bracket` and `right_bracket` into `data/math_symbols/` |

Then build the caches:

```
python -m hnrs.preprocessing
```

## Project structure

```
gui/gui.py                  user interface
hnrs/preprocess_image.py    converts one character image to MNIST format
hnrs/preprocessing.py       loads and caches all datasets, shared class list
hnrs/segmentation.py        splits an image into one crop per character
hnrs/model_loader.py        loads the saved models for the GUI
hnrs/transfer.py            fine-tunes any saved model on any mix of datasets
hnrs/arithmetic.py          calculates a recognised expression
hnrs/compose_numbers.py     builds multi-digit numbers from MNIST digits
models/                     trained models (.keras)
notebooks/                  model training and experiments
```

## Datasets

- MNIST: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
- Handwritten Digits 0 - 9 (European style): https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9
- Handwritten Math Symbols: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
- Basic arithmetics dataset: https://www.kaggle.com/datasets/simepavlic/basic-arithmetics-dataset
