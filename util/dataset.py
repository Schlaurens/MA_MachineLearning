import json
import os

from . import image as u_image


def get_label_path(directory):
    """Just a helper function to get the label path.

    Args:
        directory: directory of the dataset

    Returns:
        path to the JSON file with the labels
    """
    return os.path.join(directory, "labels.json")


def get_image_path(directory, name):
    """Get the path to an image with a given name from a given directory.

    Args:
        directory: directory of the dataset
        name: name of the image

    Returns:
        path to the image
    """
    return os.path.join(directory, f"{name}.jpg")


def save_labels(directory, labels):
    with open(get_label_path(directory), "w") as f:
        json.dump(labels, f, indent=0)


def load_labels(directory):
    with open(get_label_path(directory), "r") as f:
        labels = json.load(f)
    return labels


def load_image(directory, label, **kwargs):
    """Load image from a given directory and label.

    Args:
        directory: directory of the dataset
        label: corresponding label of the image

    Returns:
        the image
    """
    with open(get_image_path(directory, label["name"]), "rb") as f:
        image = u_image.load_bhuman_jpeg_image(f.read(), **kwargs)
    return image


def load_image_direct(path, **kwargs):
    with open(path, "rb") as f:
        image = u_image.load_bhuman_jpeg_image(f.read(), **kwargs)
    return image
