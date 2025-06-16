import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf

from util import dataset as u_dataset
from util import image as u_image


def make_example(directory, label):
    masks = u_dataset.get_masks(label, "ball")
    image_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.constant(
                            u_dataset.load_image(
                                directory, label, image_format=u_image.ImageFormat.YUYV
                            )
                        ),
                        (480, 320, 4),
                    )
                ).numpy(),
            ]
        )
    )
    camera_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.constant(u_dataset.camera_from_label(label), dtype=tf.float32)
                ).numpy(),
            ]
        )
    )
    intrinsics_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.constant(u_dataset.intrinsics_from_label(label), dtype=tf.float32)
                ).numpy(),
            ]
        )
    )
    objectness_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks[1], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )
    offset_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(tf.reshape(masks[0], (15, 20, 2))).numpy(),
            ]
        )
    )
    loss_mask_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks[2], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )

    # Create a Features dictionary
    features = tf.train.Features(
        feature={
            "image": image_feature,
            "camera": camera_feature,
            "intrinsics": intrinsics_feature,
            "objectness": objectness_feature,
            "offsets": offset_feature,
            "loss_mask": loss_mask_feature,
        }
    )

    example = tf.train.Example(features=features)
    # print(example)

    return example


def write_file(directory):
    # Load the dataset
    # TODO: must be divisible by 32
    labels = u_dataset.load_labels(directory)
    print("Dataset loaded.")

    record_file = directory + ".tfrecords"
    print("Writing file...")
    with tf.io.TFRecordWriter(record_file) as writer:
        for label in labels:
            example = make_example(directory, label)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script saves labels into a TFRecord file.")
    parser.add_argument("directory")
    args = parser.parse_args()

    write_file(args.directory)
    print("Done!")
