import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from train.models import FullModel
from util import image as u_image
from util import dataset as u_dataset


def camera_from_label(label):
    alpha = np.arccos(label["cpose"]["z"][2])
    if np.abs(alpha) < 0.01:
        roll = pitch = 0
    else:
        sin_alpha = np.sqrt(1 - label["cpose"]["z"][2]*label["cpose"]["z"][2])
        roll = -label["cpose"]["z"][1] / sin_alpha * alpha
        pitch = label["cpose"]["z"][0] / sin_alpha * alpha
    height = label["cpose"]["h"] * 0.001
    return (roll, pitch, height)

def intrinsics_from_label(label):
    return (label["cintr"]["cx"], label["cintr"]["cy"], label["cintr"]["fx"], label["cintr"]["fy"])

def get_dataset(directory):
    labels = u_dataset.load_labels(directory)[:100]

    images = []
    cameras = []
    intrinsics = []

    for label in labels:
        images.append(u_dataset.load_image(directory, label, image_format=u_image.ImageFormat.YUYV))
        cameras.append(camera_from_label(label))
        intrinsics.append(intrinsics_from_label(label))

    return tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.reshape(tf.constant(images), (-1, 480, 320, 4)),
            "camera": tf.constant(cameras, dtype=tf.float32),
            "intrinsics": tf.constant(intrinsics, dtype=tf.float32),
        }
    )


def main():

    train_ds = get_dataset("/Users/arne/Projects/Perception/data/MarcUwe_MarcUwe_CompetitionWalk_RoboCup2023__HTWK-Robots_1stHalf_1")
    train_ds = train_ds.shuffle(32)
    train_ds = train_ds.batch(32)

    model = FullModel(480, 320)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.fit(x=train_ds, epochs=10)

    """
    results = model(image, camera, intrinsics)

    fig, axes = plt.subplots(5)
    axes[0].imshow(results["ball"][0][0, 0, ...].numpy() / 255)
    axes[1].imshow(results["ball"][0][0, 1, ...].numpy() / 255)
    axes[2].imshow(results["ball"][0][0, 2, ...].numpy() / 255)
    axes[3].imshow(results["ball"][0][0, 3, ...].numpy() / 255)
    axes[4].imshow(results["ball"][0][0, 4, ...].numpy() / 255)
    
    plt.show()
    print(results)
    """

if __name__ == "__main__":
    main()
