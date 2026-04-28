import os
import cv2
import numpy as np

DATASET_PATH = "D:/Semester_3/Artificial Intelligence/yale"
VARIANCE_THRESHOLD = 0.99

def load_images_by_person(path):
    data = {}
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            subject = file.split("_")[0]
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            img = img.flatten()

            if subject not in data:
                data[subject] = []
            data[subject].append(img)
    return data


def load_images_by_expression(path):
    data = {}
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            parts = file.replace(".jpg", "").split("_")
            expression = parts[-1]
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            img = img.flatten()

            if expression not in data:
                data[expression] = []
            data[expression].append(img)
    return data


def compute_pca(X, variance=0.99):
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = (S ** 2) / (X.shape[0] - 1)
    total_variance = np.sum(eigenvalues)

    cumulative = 0
    k = 0
    for val in eigenvalues:
        cumulative += val
        k += 1
        if cumulative / total_variance >= variance:
            break

    basis = Vt[:k].T

    return mean, basis


def reconstruction_error(image, mean, basis):
    proj = (image - mean) @ basis
    recon = proj @ basis.T + mean
    return np.linalg.norm(image - recon)

def train_person_models(data):
    models = {}
    for person, images in data.items():
        train_images = images[:10]
        X = np.array(train_images)
        mean, basis = compute_pca(X)
        models[person] = (mean, basis)
    return models


def test_person_identification(data, models):
    correct = 0
    total = 0

    for person, images in data.items():
        test_img = images[-1]
        min_error = float("inf")
        predicted = None

        for p, (mean, basis) in models.items():
            err = reconstruction_error(test_img, mean, basis)
            if err < min_error:
                min_error = err
                predicted = p

        print(f"Actual: {person} | Predicted: {predicted}")
        if predicted == person:
            correct += 1
        total += 1

    print("\nPerson Identification Accuracy:", correct / total)

# Part 2
def train_expression_models(data):
    models = {}
    for exp, images in data.items():
        X = np.array(images[:10])
        mean, basis = compute_pca(X)
        models[exp] = (mean, basis)
    return models


def test_expression_recognition(data, models):
    correct = 0
    total = 0

    for exp, images in data.items():
        for img in images[10:]:
            min_error = float("inf")
            predicted = None

            for e, (mean, basis) in models.items():
                err = reconstruction_error(img, mean, basis)
                if err < min_error:
                    min_error = err
                    predicted = e

            if predicted == exp:
                correct += 1
            total += 1

    print("Expression Recognition Accuracy:", correct / total)

def main():
    person_data = load_images_by_person(DATASET_PATH)
    expression_data = load_images_by_expression(DATASET_PATH)

    print("\nTraining Person Identification Model...")
    person_models = train_person_models(person_data)
    test_person_identification(person_data, person_models)

    print("\nTraining Expression Recognition Model...")
    expression_models = train_expression_models(expression_data)
    test_expression_recognition(expression_data, expression_models)


if __name__ == "__main__":
    main()
