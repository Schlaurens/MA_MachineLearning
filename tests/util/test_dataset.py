import tensorflow as tf

from src.util import dataset as u_dataset

dataset_config = u_dataset.DatasetConfig()
dataset_utils = u_dataset.DatasetUtils(dataset_config)


class TestCoordsInCell:
    def test_zeros(self):
        coords_zero = tf.constant([0, 0], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_zero, coords_zero) == True

    def test_same_cell(self):
        coords_a = tf.constant([32, 32], tf.float32)
        coords_b = tf.constant([35, 35], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == True

    def test_different_cell(self):
        coords_a = tf.constant([100, 35], tf.float32)
        coords_b = tf.constant([32, 35], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == False
