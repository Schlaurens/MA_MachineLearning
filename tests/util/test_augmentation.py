import pytest
import tensorflow as tf

from src.util import augmentation as u_augmentation
from src.util import dataset as u_dataset


class TestAugmentCellIndices:
    @pytest.fixture
    def dataset_config(self):
        dataset_config = u_dataset.DatasetConfig()

        return dataset_config

    @pytest.fixture
    def indices(self):
        # Example indices of shape (B, N)
        return tf.constant([[0, 1, 200, 3], [4, 57, 66, 70]], dtype=tf.int32)  # (B, 4)

    def test_output_shape(self, dataset_config, indices):
        """Test that the output shape matches the input shape."""
        result = u_augmentation.augment_cell_indices(dataset_config, indices)
        assert result.shape == indices.shape

    def test_clipping_lower_bound(self, dataset_config, indices):
        """Test that values are clipped to >= 0."""
        result = u_augmentation.augment_cell_indices(dataset_config, indices, minval=-3, maxval=-1)
        assert tf.reduce_all(result >= 0)

    def test_clipping_upper_bound(self, dataset_config: u_dataset.DatasetConfig, indices):
        """Test that values are clipped to <= H*W - 1."""
        max_val = tf.reduce_prod(dataset_config.output_dims) - 1
        result = u_augmentation.augment_cell_indices(dataset_config, indices, minval=max_val, maxval=max_val+4)
        assert tf.reduce_all(result <= max_val)

    def test_noise_application(self, dataset_config, indices):
        """Test that noise is applied and values change (unless clipped)."""
        # Set seed for reproducibility
        result1 = u_augmentation.augment_cell_indices(dataset_config, indices, seed=42)
        result2 = u_augmentation.augment_cell_indices(dataset_config, indices, seed=43)
        # Results should differ due to different random seeds
        assert not tf.reduce_all(tf.equal(result1, result2))

    def test_edge_case_empty_indices(self, dataset_config):
        """Test with empty indices tensor."""
        empty_indices = tf.constant([], shape=(0, 0), dtype=tf.int32)
        result = u_augmentation.augment_cell_indices(dataset_config, empty_indices)
        assert result.shape == empty_indices.shape

    def test_edge_case_single_element(self, dataset_config):
        """Test with a single element."""
        single_indices = tf.constant([[0]], dtype=tf.int32)
        result = u_augmentation.augment_cell_indices(dataset_config, single_indices)
        assert result.shape == single_indices.shape
        assert tf.reduce_all(result >= 0)
        assert tf.reduce_all(result <= tf.reduce_prod(dataset_config.output_dims) - 1)
