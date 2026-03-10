<div align="center">

# Deep Learning

<p>
	<strong>A deep learning study repository built around PyTorch experiments, mathematical intuition, and LaTeX notes</strong>
</p>

<p>
	<img src="https://img.shields.io/badge/Python-3.11.9-1F6FEB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11.9" />
	<img src="https://img.shields.io/badge/PyTorch-Practice-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Practice" />
	<img src="https://img.shields.io/badge/TorchVision-Datasets-0A7E8C?style=for-the-badge" alt="TorchVision Datasets" />
	<img src="https://img.shields.io/badge/LaTeX-Notes-2F855A?style=for-the-badge&logo=latex&logoColor=white" alt="LaTeX Notes" />
	<img src="https://img.shields.io/badge/License-CC0%201.0-7C3AED?style=for-the-badge" alt="CC0 1.0" />
</p>

<p>
	From linear regression to multilayer perceptrons, MNIST classification, and structured lecture-style notes.<br/>
	This repository combines runnable code with notes that are meant to be read continuously.
</p>

</div>

---

## Overview

This repository is not just a collection of isolated scripts. It is organized around two parallel tracks for learning deep learning:

- Runnable PyTorch examples that validate core ideas with small but complete experiments.
- Continuously refined LaTeX and Markdown notes that connect formulas, intuition, and implementation into a single map of knowledge.

If you want to see how concepts become code, or you prefer to keep experiments and structured notes in the same place, this repository is designed for that workflow.

## Highlights

| Module | Content | Focus |
| --- | --- | --- |
| `Examples` | Linear regression, MNIST digit recognition, Fashion-MNIST softmax classification | Built around hands-on implementation and direct execution |
| `Notes` | Main LaTeX document, chapter sources, exported PDF, Markdown draft | Built around structure, clarity, and readability |
| Learning path | From tensors and data processing to linear regression and MLPs | Suitable for building fundamentals step by step |

## Repository Structure

```text
Deep-Learning/
├─ Examples/
│  ├─ linear/
│  │  └─ test.py                  # Handwritten linear regression with minibatch SGD in PyTorch
│  ├─ mnist/
│  │  └─ handwriting_recog.py     # CNN-based MNIST handwritten digit recognition
│  └─ softmax/
│     └─ fashion_mnist.py         # Softmax regression on Fashion-MNIST
├─ Notes/
│  ├─ chapters/
│  │  ├─ preface.tex              # Preface
│  │  ├─ chapter1.tex             # Basics: Python, tensors, and data preparation
│  │  ├─ chapter2.tex             # Linear regression
│  │  └─ chapter3.tex             # Multilayer perceptrons
│  ├─ markdown/
│  │  ├─ note.md                  # Markdown version of the study notes
│  │  └─ figures/                 # Figures used by the Markdown notes
│  ├─ main.tex                    # Main LaTeX entry point
│  └─ main.pdf                    # Compiled PDF notes
├─ LICENSE
└─ README.md
```

## Current Coverage

| Topic | Description | Location |
| --- | --- | --- |
| Basics | Python environment, common AI libraries, tensor basics, and data preprocessing | `Notes/chapters/chapter1.tex` |
| Linear Regression | Linear models, loss functions, analytical solutions, SGD, and probabilistic interpretation | `Notes/chapters/chapter2.tex` |
| Multilayer Perceptrons | The starting chapter for MLPs and the transition toward neural networks | `Notes/chapters/chapter3.tex` |
| Linear Demo | A handwritten linear regression training pipeline on synthetic data | `Examples/linear/test.py` |
| MNIST Demo | Handwritten digit recognition with a CNN | `Examples/mnist/handwriting_recog.py` |
| Fashion-MNIST Demo | Softmax regression, training flow, and prediction visualization | `Examples/softmax/fashion_mnist.py` |

## Quick Start

### 1. Create a Python Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

This repository does not currently maintain a standalone `requirements.txt`, so you can begin with the dependencies required by the existing examples:

```bash
pip install --upgrade pip
pip install torch torchvision matplotlib numpy pandas seaborn
```

If you are using CUDA, it is better to install PyTorch with the official command that matches your driver and platform.

### 3. Run the Examples

```bash
python Examples/linear/test.py
python Examples/mnist/handwriting_recog.py
python Examples/softmax/fashion_mnist.py
```

After running them, you should see:

- The linear regression parameters gradually approaching the ground truth.
- Training logs and test accuracy for MNIST.
- The Fashion-MNIST training process and sample prediction visualizations.

## Reading and Building the Notes

### Read Directly

- If you want the full lecture-style document, open `Notes/main.pdf`.
- If you prefer a lighter format or want to continue drafting content, check `Notes/markdown/note.md`.

### Compile the LaTeX Document

If LaTeX is installed on your machine, run the following inside the `Notes` directory:

```bash
cd Notes
latexmk -pdf main.tex
```

If `latexmk` is not available, you can use the traditional workflow:

```bash
cd Notes
pdflatex main.tex
pdflatex main.tex
```

After compilation, `Notes/main.pdf` will be generated or updated.

## Suggested Reading Order

1. Start with `Notes/chapters/chapter1.tex` to build the foundations of tensors and data processing.
2. Then run `Examples/linear/test.py` to understand the most basic training loop and parameter updates.
3. Continue with `Notes/chapters/chapter2.tex` to connect linear regression to its mathematical and probabilistic background.
4. Finally, run the MNIST and Fashion-MNIST examples to move into real image classification tasks.

## Who This Is For

- Beginners who want to build deep learning fundamentals from scratch.
- Learners who want to keep experiments and structured notes side by side.
- Anyone turning their AI study process into a long-term repository.

## Possible Next Directions

- Convolutional neural networks and more complete computer vision experiments.
- Optimizers, regularization, overfitting, and generalization.
- Sequence models, attention mechanisms, and Transformers.
- More structured dataset management, experiment tracking, and result comparison.

## License

This repository is released under `CC0 1.0 Universal`. See `LICENSE` for details.

---

> This repository records more than code outputs. It captures the process of gradually aligning formulas, implementation, and understanding.
