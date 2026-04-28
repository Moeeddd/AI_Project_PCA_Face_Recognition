Here is a complete and professional `README.md` file for your project. You can copy and paste this directly into a new file named `README.md` in your GitHub repository.

***

# PCA-Based Face & Expression Recognition

## Overview
This project implements a Principal Component Analysis (PCA) algorithm from scratch to perform two distinct computer vision tasks using the Yale Face Database:
1.  **Person Identification:** Identifying an individual based on their facial features.
2.  **Expression Recognition:** Categorizing the facial expression (e.g., happy, sad, surprised) regardless of the person.

The system uses dimensionality reduction and reconstruction error to classify unseen images by comparing them against trained PCA models.

---

## Prerequisites
To run this project, you need Python installed on your system along with the following libraries:

* `numpy` (for matrix operations and SVD)
* `opencv-python` (for image processing and reading)

You can install the required dependencies using pip:
```bash
pip install numpy opencv-python
```

---

## Dataset Configuration
This project relies on the **Yale Face Database**. 

**Important:** Before running the script, you must update the `DATASET_PATH` variable in `AI Project02.py` to point to the directory where your dataset is extracted.

```python
# Update this line in the code to match your local or repository path
DATASET_PATH = "path/to/your/yale/dataset"
```

The script expects the dataset images to be in `.jpg` format and adhere to the standard Yale naming convention, where the filename contains the subject and expression (e.g., `subject01_happy.jpg`, `subject02_sad.jpg`).

---

## Features & Implementation Details

### 1. Person Identification
* **Data Splitting:** Groups images by subject ID (extracted from the filename prefix).
* **Training:** Uses the first 10 images of each person to compute a specific PCA basis (eigenfaces) for that individual.
* **Testing:** Evaluates the last image of each person by calculating the reconstruction error against all trained models. The model yielding the lowest error is selected as the prediction.

### 2. Expression Recognition
* **Data Splitting:** Groups images by facial expression (extracted from the filename suffix).
* **Training:** Uses the first 10 images of each expression category to compute the PCA basis.
* **Testing:** Evaluates all remaining images in the dataset, categorizing them based on the lowest reconstruction error among the expression models.

---

## Mathematical Approach (PCA)
Instead of relying on pre-built machine learning libraries like `scikit-learn`, this project calculates PCA using Singular Value Decomposition (SVD). 

For a given matrix of centered image data $\mathbf{X}$, SVD is computed to extract the orthogonal basis vectors $\mathbf{V}^T$. The algorithm retains enough basis vectors to explain a specified variance threshold (defaulted to $0.99$ or 99%).

**Classification via Reconstruction Error**
Classification is performed by projecting a test image $\mathbf{x}$ into the PCA space of a specific class and then reconstructing it. The reconstruction error $E$ is calculated using the Frobenius norm:

$E = \|\mathbf{x} - \mathbf{\hat{x}}\|$

Where the reconstructed image $\mathbf{\hat{x}}$ is defined as:

$\mathbf{\hat{x}} = ((\mathbf{x} - \mathbf{\mu}) \mathbf{B}) \mathbf{B}^T + \mathbf{\mu}$

* $\mathbf{\mu}$ = Mean face of the class
* $\mathbf{B}$ = PCA basis vectors for the class

The class model that produces the lowest error $E$ is chosen as the match.

---

## Usage
To execute the script, simply run it from your terminal:

```bash
python "AI Project02.py"
```

**Output:**
The terminal will display the live identification results for individual subjects, followed by the final accuracy scores for both Person Identification and Expression Recognition.
