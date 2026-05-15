import pytest
import tensorflow as tf

from src.util import keypoint as u_keypoint
from src.util import layers as u_layers


class TestAreCoordsInPatch:
    def test_single_patch(self):
        coordinates_normalized = tf.constant([[0.343, 0.89]], tf.float32)
        box = tf.constant([[0.1, 0.24, 0.7, 0.5]], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.0)

        assert tf.reduce_all(result == expected)

    def test_multiple_patches(self):
        coordinates_normalized = tf.constant([[0.8, 0.1], [0.343, 0.89]], tf.float32)
        boxes = tf.constant([[0.01, 0.7, 0.3, 0.9], [0.1, 0.24, 0.7, 0.5]], tf.float32)

        expected = tf.constant([True, False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, boxes, padding=0.0)

        assert tf.reduce_all(result == expected)

    def test_coord_not_in_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.75, 0.75]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([True])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.0)

        assert tf.reduce_all(result == expected)

    def test_coord_in_left_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.6, 0.75]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_coord_in_right_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.9, 0.75]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_coord_in_top_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.75, 0.6]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_coord_in_bottom_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.75, 0.9]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_single_coord_multiple_boxes(self):
        coords = tf.constant([[0.75, 0.9], [0.75, 0.9]], tf.float32)  # (2, 2) — B=2
        boxes = tf.constant(
            [
                [[0.5, 0.5, 1.0, 1.0], [3, 3, 2.0, 2.0], [0.5, 0.5, 1.0, 1.0]],  # 3 candidates
                [[0.5, 0.5, 1.0, 1.0], [3, 3, 2.0, 2.0], [0.5, 0.5, 1.0, 1.0]],
            ],
            tf.float32,
        )  # (2, 3, 4)
        expected = tf.constant([[True, False, True], [True, False, True]])  # (2, 3)
        result = u_keypoint.are_coords_in_patch(
            coords[:, tf.newaxis, :],  # (2, 1, 2)
            boxes,  # (2, 3, 4)
            padding=0,
        )
        assert tf.reduce_all(result == expected)

    def test_negative_coords(self):
        coordinates_normalized = tf.constant([[-0.0015625, -0.00208333]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)


class TestPatchCoordsToImageCoords:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.patch_dims = tf.constant([32, 32], tf.float32)
        self.patch_center = tf.constant([15.5, 15.5], tf.float32)
        self.extractor = u_layers.PatchExtractor()
        self.camera = tf.constant(
            [[-0.0220113918, 0.0786367953, 0.493571222]],
            tf.float32,
        )  # (B, 2, 3)
        self.camera_intr = tf.constant([[320, 240, 618.663391, 617.159]], tf.float32)  # (B, 2, 4)

    def test_base_case(self):
        coords = tf.constant([[[30, 380], [400, 400]]], tf.float32)  # (B, 2, 2)

        (_, _, _, pixel_sizes) = self.extractor(
            tf.zeros((1, 480, 640, 1)), coords, self.camera, self.camera_intr
        )

        patch_coords = tf.constant([[15.5, 15.5], [15.5, 15.5]])

        results = u_keypoint.patch_coords_to_image_coords(
            self.patch_dims, pixel_sizes, patch_coords, coords
        )

        tf.print(results)

        assert tf.reduce_all(coords == results)

    def _run(self, patch_coords, image_coords, pixel_sizes):
        return u_keypoint.patch_coords_to_image_coords(
            self.patch_dims,
            pixel_sizes,
            patch_coords,
            image_coords,
        )

    def test_patch_center_returns_image_coords(self):
        """Patch center should map exactly to the image center coords."""
        image_coords = tf.constant([[[100.0, 200.0]]])  # (1, 1, 2)
        patch_coords = tf.constant([[[15.5, 15.5]]])  # (1, 1, 2)
        pixel_sizes = tf.constant([[2.0]])  # (1, 1)

        result = self._run(patch_coords, image_coords, pixel_sizes)
        tf.debugging.assert_near(result, image_coords)

    def test_top_left_corner(self):
        """Patch (0,0) should be offset by -patch_center * pixel_size."""
        image_coords = tf.constant([[[100.0, 200.0]]])
        patch_coords = tf.constant([[[0.0, 0.0]]])
        pixel_sizes = tf.constant([[2.0]])

        expected = image_coords + (tf.constant([[[0.0, 0.0]]]) - self.patch_center) * 2.0
        result = self._run(patch_coords, image_coords, pixel_sizes)
        tf.debugging.assert_near(result, expected)

    def test_bottom_right_corner(self):
        """Patch (31,31) should be offset by +patch_center * pixel_size."""
        image_coords = tf.constant([[[100.0, 200.0]]])
        patch_coords = tf.constant([[[31.0, 31.0]]])
        pixel_sizes = tf.constant([[2.0]])

        expected = image_coords + (tf.constant([[[31.0, 31.0]]]) - self.patch_center) * 2.0
        result = self._run(patch_coords, image_coords, pixel_sizes)
        tf.debugging.assert_near(result, expected)

    def test_top_left_and_bottom_right_are_symmetric(self):
        """Top-left and bottom-right offsets should be equal and opposite."""
        image_coords = tf.constant([[[100.0, 200.0], [100.0, 200.0]]])  # (1, 2, 2)
        patch_coords = tf.constant([[[0.0, 0.0], [31.0, 31.0]]])
        pixel_sizes = tf.constant([[2.0, 2.0]])

        result = self._run(patch_coords, image_coords, pixel_sizes)
        offset_tl = result[:, 0, :] - image_coords[:, 0, :]
        offset_br = result[:, 1, :] - image_coords[:, 1, :]
        tf.debugging.assert_near(offset_tl, -offset_br)

    def test_pixel_size_scales_offset(self):
        """Doubling pixel_size should double the offset from center."""
        image_coords = tf.constant([[[100.0, 200.0], [100.0, 200.0]]])  # (1, 2, 2)
        patch_coords = tf.constant([[[0.0, 0.0], [0.0, 0.0]]])
        pixel_sizes_1x = tf.constant([[1.0, 1.0]])
        pixel_sizes_2x = tf.constant([[2.0, 2.0]])

        result_1x = self._run(patch_coords, image_coords, pixel_sizes_1x)
        result_2x = self._run(patch_coords, image_coords, pixel_sizes_2x)

        offset_1x = result_1x - image_coords
        offset_2x = result_2x - image_coords
        tf.debugging.assert_near(offset_2x, 2.0 * offset_1x)
