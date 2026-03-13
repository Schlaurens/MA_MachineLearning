import numpy as np
import pytest
import tensorflow as tf

from src.util import dataset as u_dataset

CONFIGS = [
    (None, (240, 320), (16, 16)),
    (None, (288, 384), (16, 16)),
    (None, (360, 480), (24, 24)),
    (None, (432, 576), (24, 24)),
    (None, (480, 640), (32, 32)),
]


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestFilterCoordinates:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.cell_dims = cell_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_single_coordinate(self):
        coordinates = tf.constant([[32, 54]], tf.float32)
        expected = coordinates

        result = self.dataset_utils.filter_coordinates(coordinates)

        # assert result.shape == expected.shape
        assert tf.reduce_all(result == expected)

    def test_unique_coordinates(self):
        # No need for filtering.
        coordinates = tf.constant([[32, 54], [100, 102]], tf.float32)
        expected = coordinates

        result = self.dataset_utils.filter_coordinates(coordinates)

        # assert result.shape == expected.shape
        assert tf.reduce_all(result == expected)

    def test_duplicate_coordinates(self):
        coordinates = tf.constant([[32, 54], [32, 54]], tf.float32)
        expected = tf.constant([[32, 54]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)

        # assert result.shape == expected.shape
        assert tf.reduce_all(result == expected)

    def test_same_cell_both_axes(self):
        # One coordinate pair is on both axes bigger than the other one.
        coordinates = tf.constant([[33, 40], [34, 43]], tf.float32)
        expected = tf.constant([[34, 43]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_same_cell_bigger_x_axis(self):
        # One coordinate pair is only on the x axis bigger than the other. On the y axis they are equal.
        coordinates = tf.constant([[33, 43], [34, 43]], tf.float32)
        expected = tf.constant([[34, 43]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_same_cell_bigger_y_axis(self):
        # One coordinate pair is only on the y axis bigger than the other. On the x axis they are equal.
        coordinates = tf.constant([[33, 43], [33, 45]], tf.float32)
        expected = tf.constant([[33, 45]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_negative_values(self):
        coordinates = tf.constant([[-1, -1], [-1, -1], [35, 30]], tf.float32)
        expected = tf.constant([[-1, -1], [35, 30]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_close_floating_values(self):
        # floats that are very close to each other
        coordinates = tf.constant(
            [[33, 34], [33.0000002, 34.000003], [67.03125, 78.436558]], tf.float32
        )
        expected = tf.constant([[33.0000002, 34.000003], [67.03125, 78.436558]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_values_at_cell_edge(self):
        grid_size = self.dataset_config.cell_dims[0]
        coordinates = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )
        expected = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_multiple_zero_coords(self):
        coordinates = tf.constant([[0, 0], [0, 0], [0, 0]], tf.float32)
        expected = tf.constant([[0, 0]], tf.float32)

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_close_floating_values_v2(self):
        cell_h, cell_w = self.cell_dims

        # Lone coordinate in cell (0, 0)
        lone = tf.constant([cell_h // 3, cell_w // 3], tf.float32)

        # Two coordinates in cell (1, 1) — coord_b has smaller y, so it's retained
        cell1_coord_a = tf.constant([float(cell_h + 1), float(cell_w + 1)], tf.float32)
        cell1_coord_b = tf.constant([float(cell_h), float(cell_w + 2)], tf.float32)

        # Two coordinates in cell (9, 10) — coord_a has smaller y, so it's retained
        cell9_coord_a = tf.constant(
            [float(9 * cell_h + cell_h // 3), float(10 * cell_w + cell_w // 8)], tf.float32
        )
        cell9_coord_b = tf.constant(
            [float(9 * cell_h + 2 * (cell_h // 3)), float(10 * cell_w + cell_w // 32)], tf.float32
        )

        coordinates = tf.stack([lone, cell1_coord_a, cell1_coord_b, cell9_coord_a, cell9_coord_b])
        expected = tf.stack([lone, cell1_coord_b, cell9_coord_a])

        result = self.dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestCoordsInSameCell:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.cell_dims = np.array(cell_dims)
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_zeros(self):
        coords_zero = tf.constant([0, 0], tf.float32)
        assert self.dataset_utils.are_coords_in_same_cell(coords_zero, coords_zero) == True

    def test_same_cell(self):
        cell_h, cell_w = self.cell_dims
        # Both coords placed inside cell (1, 1) by anchoring to its origin
        cell_origin = tf.constant([cell_h * 1, cell_w * 1], tf.float32)
        coords_a = cell_origin
        coords_b = cell_origin + tf.constant([3, 3], tf.float32)
        assert self.dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == True

    def test_same_cell_close_to_edge(self):
        cell_h, cell_w = self.cell_dims

        # Both coords near the far edge of cell (9, 10)
        cell_origin = np.array([cell_h * 9, cell_w * 10], dtype=np.float32)
        coords_a = tf.constant(cell_origin + [cell_h - 4, cell_w - 2], tf.float32)
        coords_b = tf.constant(cell_origin + [cell_h - 1, cell_w - 3], tf.float32)
        assert self.dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == True

        # Both coords near the start edge of cell (1, 1)
        cell_origin = np.array([cell_h * 1, cell_w * 1], dtype=np.float32)
        coords_c = tf.constant(cell_origin + [1, 1], tf.float32)
        coords_d = tf.constant(cell_origin + [0, 2], tf.float32)
        assert self.dataset_utils.are_coords_in_same_cell(coords_c, coords_d) == True

    def test_different_cell(self):
        cell_h, cell_w = self.cell_dims
        # coords_a is in cell (3, 1), coords_b is in cell (1, 1) — different rows
        coords_a = tf.constant([cell_h * 3 + 4, cell_w * 1 + 3], tf.float32)
        coords_b = tf.constant([cell_h * 1 + 0, cell_w * 1 + 3], tf.float32)
        assert self.dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == False


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestGetCoordsFromOffset:
    # Assumes that the generation of offset_masks works!
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_single_coordinate_pair(self):
        # Only one object in the sample (all cells point to one cell)
        coordinates = tf.constant([[34.0534, 67.432]], tf.float32)
        offset_mask = self.dataset_utils._generate_offset_mask(coordinates)

        result = self.dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(coordinates, axis=0), axis=1)
            )
        )

    def test_multiple_coordinate_pairs(self):
        # Multiple objects in the sample (each in their own cell)
        coordinates = tf.constant(
            [[34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        # Filter coord to account for differing cell_dims. Because they could be in the same cell.
        coordinates_filtered = self.dataset_utils.filter_coordinates(coordinates)

        offset_mask = self.dataset_utils._generate_offset_mask(coordinates)

        result = self.dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1),
                tf.sort(tf.expand_dims(coordinates_filtered, axis=0), axis=1),
            )
        )

    def test_multiple_coordinate_pairs_with_negative(self):
        coordinates = tf.constant(
            [[-1, -1], [34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        # Filter coord to account for differing cell_dims. Because they could be in the same cell.
        coordinates_filtered = self.dataset_utils.filter_coordinates(coordinates)

        offset_mask = self.dataset_utils._generate_offset_mask(coordinates)

        result = self.dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1),
                tf.sort(tf.expand_dims(coordinates_filtered, axis=0), axis=1),
            )
        )

    def test_no_coordinates(self):
        # The offset_mask of an empty sample points to [-1.0, -1.0]
        coordinates = tf.constant([[]], tf.float32)
        offset_mask = self.dataset_utils.get_masks(coordinates=coordinates)["offsets"]
        expected = tf.constant([[-1, -1]], tf.float32)

        result = self.dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(expected, axis=0), axis=1)
            )
        )

    def test_coords_at_cell_edge(self):
        grid_size = self.dataset_config.cell_dims[0]

        coordinates = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )
        expected = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )
        offset_mask = self.dataset_utils._generate_offset_mask(coordinates)
        result = self.dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(expected, axis=0), axis=1)
            )
        )

    def test_coords_batched(self):
        coordinates = tf.constant(
            [[34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        # Filter coord to account for differing cell_dims. Because they could be in the same cell.
        coordinates_filtered = self.dataset_utils.filter_coordinates(coordinates)

        offset_mask = tf.expand_dims(
            self.dataset_utils._generate_offset_mask(coordinates), axis=0
        )  # (H, W, 2)
        coordinates_empty = tf.constant([[]], tf.float32)
        offset_mask_empty = tf.expand_dims(
            self.dataset_utils.get_masks(coordinates=coordinates_empty)["offsets"], axis=0
        )

        input_mask = tf.concat(
            [offset_mask, offset_mask_empty, offset_mask], axis=0
        )  # (2, H, W, 2)

        result = self.dataset_utils.get_coords_from_offsets(input_mask)  # (B, N, 2)
        tf.print(coordinates_filtered)
        expected = tf.stack(
            [
                coordinates_filtered,
                tf.fill(tf.shape(coordinates_filtered), [-1.0]),
                coordinates_filtered,
            ],
            axis=0,
        )  # (B, N, 2)

        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1),
                tf.sort(expected, axis=1),
            )
        )


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestObjectMask:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_basic(self):
        coordinates = tf.constant([[0.5, 0.5], [100.3, 106.06]], tf.float32)
        object_mask = tf.cast(
            self.dataset_utils.get_masks(coordinates=coordinates)["object_mask"], tf.int32
        )

        coordinates_scaled = coordinates * self.dataset_config.image_res_scale

        mask_indices = tf.cast(coordinates_scaled // self.dataset_config.cell_dims, tf.int32)[
            ..., ::-1
        ]  # (N, 2)
        mask_values = tf.gather_nd(object_mask, mask_indices)

        assert tf.reduce_sum(object_mask) == coordinates_scaled.shape[0]
        assert tf.reduce_all(mask_values == 1)

    def test_same_cell(self):
        cell_h = float(self.dataset_config.cell_dims[0])
        cell_w = float(self.dataset_config.cell_dims[1])

        # Two coordinates in the same cell (0, 0) — near opposite corners
        coord_a = tf.constant([0.5, 0.5], tf.float32)
        coord_b = tf.constant([cell_h - cell_h / 4, cell_w - cell_w / 4], tf.float32)

        coordinates = tf.stack([coord_a, coord_b])

        object_mask = tf.cast(
            self.dataset_utils.get_masks(coordinates=coordinates)["object_mask"], tf.int32
        )
        mask_indices = tf.cast(coordinates // self.dataset_config.cell_dims, tf.int32)[
            ..., ::-1
        ]  # (N, 2)
        mask_values = tf.gather_nd(object_mask, mask_indices)

        assert tf.reduce_sum(object_mask) == coordinates.shape[0] - 1
        assert tf.reduce_all(mask_values == 1)

    def test_out_of_bounds(self):
        coordinates = tf.constant(
            [
                [self.dataset_config.input_dims[1] + 10, 70],
                [32, self.dataset_config.input_dims[0] + 10],
                [-5, 80],
                [50, -5],
            ],
            tf.float32,
        )
        object_mask = tf.cast(
            self.dataset_utils.get_masks(coordinates=coordinates)["object_mask"], tf.int32
        )

        coordinates_scaled = coordinates * self.dataset_config.image_res_scale[::-1]

        clipped_coords = tf.clip_by_value(
            coordinates_scaled,
            clip_value_min=tf.constant([0.0, 0.0], dtype=coordinates_scaled.dtype),
            clip_value_max=tf.constant(
                [self.dataset_config.input_dims[1] - 1, self.dataset_config.input_dims[0] - 1],
                dtype=coordinates_scaled.dtype,
            ),
        )  # (N, 2)

        mask_indices = tf.cast(clipped_coords // self.dataset_config.cell_dims, tf.int32)[
            ..., ::-1
        ]  # (N, 2)
        mask_values = tf.gather_nd(object_mask, mask_indices)

        assert tf.reduce_sum(object_mask) == coordinates_scaled.shape[0]
        assert tf.reduce_all(mask_values == 1)


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestGetCellOfCoordinates:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.cell_dims = cell_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_basic_case(self):
        ch, cw = self.cell_dims  # cell height, cell width

        # coords_a: middle of cell (0, 0)
        coords_a = tf.constant([ch / 2, cw / 2], tf.float32)
        expected_a = tf.constant([0, 0], tf.int32)

        # coords_b: middle of cell (1, 2): one cell down, two cells right
        coords_b = tf.constant([ch + ch / 2, 2 * cw + cw / 2], tf.float32)
        expected_b = tf.constant([1, 2], tf.int32)

        result_a = self.dataset_utils.get_cell_of_coordinate(coords_a)
        result_b = self.dataset_utils.get_cell_of_coordinate(coords_b)
        assert tf.reduce_all(expected_a == result_a)
        assert tf.reduce_all(expected_b == result_b)

    def test_small_coords(self):
        coords = tf.constant([[1, 1], [0.5, 0.5], [-1, -1]], tf.float32)
        expected = tf.constant([[0, 0], [0, 0], [-1, -1]], tf.int32)
        results = self.dataset_utils.get_cell_of_coordinate(coords, clip=False)
        assert tf.reduce_all(expected == results)

    def test_batched_coords(self):
        ch, cw = float(self.cell_dims[0]), float(self.cell_dims[1])

        # (B, N, 2) batch: cells (0,0), (1,2), (2,0), (1,9)
        coords = tf.constant(
            [
                [[ch / 2, cw / 2], [ch + ch / 2, 2 * cw + cw / 2]],
                [[2 * ch + 1, cw / 2], [ch + ch / 2, 9 * cw + cw / 2]],
            ],
            tf.float32,
        )
        expected = tf.constant(
            [
                [[0, 0], [1, 2]],
                [[2, 0], [1, 9]],
            ],
            tf.int32,
        )
        results = self.dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_negative_coords(self):
        ch, cw = float(self.cell_dims[0]), float(self.cell_dims[1])

        # -1 cell in y  -> floordiv(-ch/2, ch) = -1
        # -2 cells in x -> floordiv(-ch/2 - cw, cw) = -2  (one full cell + a bit over)
        coords = tf.constant([[-ch / 2, -cw - cw / 2]], tf.float32)
        expected = tf.constant([[-1, -2]], tf.int32)
        results = self.dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_coords_at_cell_edge(self):
        ch, cw = float(self.cell_dims[0]), float(self.cell_dims[1])

        # Just inside cell (0,0) vs. exactly on the boundary -> steps into cell (0,1)
        coords = tf.constant(
            [
                [ch - 1, cw - 1],  # still in (0, 0)
                [ch - 1, cw],  # crosses into (0, 1)
            ],
            tf.float32,
        )
        expected = tf.constant([[0, 0], [0, 1]], tf.int32)
        results = self.dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_coords_at_grid_edge(self):
        ch = float(self.cell_dims[0])

        coords = tf.constant([[ch - 1, self.dataset_config.input_dims[1] - 1]], tf.float32)
        expected = tf.constant([[0, self.dataset_config.output_dims[1] - 1]], tf.int32)
        results = self.dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_coords_at_grid_edge_with_clip(self):
        ch = float(self.cell_dims[1])

        coords = tf.constant([[ch - 1, self.dataset_config.input_dims[0] - 1]], tf.float32)
        expected = tf.constant([[0, self.dataset_config.output_dims[0] - 1]], tf.int32)
        results = self.dataset_utils.get_cell_of_coordinate(coords, clip=True)

        assert tf.reduce_all(expected == results)

    def test_negative_coords_with_clip(self):
        cw = float(self.cell_dims[1])

        # [-1, 40] -> raw (-1, 40//cw), clipped to (0, 40//cw)
        # [3, -50] -> raw (0, -2), clipped to (0, 0)
        # [-1, -1] -> raw (-1, -1), clipped to (0, 0)
        coords = tf.constant(
            [
                [-1, cw + cw / 2],  # lands in column 1 after clip
                [3, -50],
                [-1, -1],
            ],
            tf.float32,
        )
        expected = tf.constant([[0, 1], [0, 0], [0, 0]], tf.int32)
        results = self.dataset_utils.get_cell_of_coordinate(coords, clip=True)

        assert tf.reduce_all(expected == results)

    def test_too_large_coords_with_clip(self):
        cw = float(self.cell_dims[1])

        coords = tf.constant(
            [
                [self.dataset_config.input_dims[1], 40],
                [3, self.dataset_config.input_dims[0]],
                [700000, 1000000],
            ],
            tf.float32,
        )
        # the function returns the indices in x,y not y,x
        expected = tf.constant(
            [
                [self.dataset_config.output_dims[1] - 1, int(40 // cw)],
                [0, self.dataset_config.output_dims[0] - 1],
                self.dataset_config.output_dims[::-1] - 1,
            ],
            tf.int32,
        )

        results = self.dataset_utils.get_cell_of_coordinate(coords, clip=True)
        assert tf.reduce_all(expected == results)


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestFlattenCellIndices:
    # These tests are done with the default cell_grid size.
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_index_in_first_row(self):
        indices = tf.constant([0, 5], tf.int32)
        expected = tf.constant([self.dataset_config.output_dims[1] * 5])
        result = self.dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_index_in_second_row(self):
        indices = tf.constant([2, 0], tf.int32)
        expected = tf.constant([2], tf.int32)
        result = self.dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_index_in_lower_right_corner_of_grid(self):
        indices = tf.constant(self.dataset_config.output_dims[::-1] - 1, tf.int32)
        expected = tf.reduce_prod(indices + 1) - 1
        result = self.dataset_utils.flatten_cell_indices(indices)
        assert tf.reduce_all(expected == result)

    def test_batched_index(self):
        last = self.dataset_config.output_dims[1] - 1, self.dataset_config.output_dims[0] - 1
        indices = tf.constant(
            [
                [[last[0], last[1]], [0, 0]],
                [[last[0], last[1]], [0, 0]],
            ],
            tf.int32,
        )

        expected = tf.constant(
            [[(last[0] + 1) * (last[1] + 1) - 1, 0], [(last[0] + 1) * (last[1] + 1) - 1, 0]],
            tf.int32,
        )  # (B, N)
        result = self.dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_negative_index(self):
        indices = tf.constant([[[-1, -1], [0, -4]]], tf.int32)  # (B, N, 2)
        # expected = tf.constant([[-21, -80]], tf.int32)  # (B, N)

        expected = tf.constant(
            [[-self.dataset_config.output_dims[1] - 1, self.dataset_config.output_dims[1] * -4]],
            tf.int32,
        )  # (B, N)

        result = self.dataset_utils.flatten_cell_indices(indices)
        assert tf.reduce_all(expected == result)


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestGetDistanceMaskFromOffsets:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

        self.camera = tf.constant(
            [[-0.0220113918, 0.0786367953, 0.493571222], [0.0550611168, 0.298894078, 0.481570929]],
            tf.float32,
        )  # (B, 3)
        self.camera_intr = tf.constant(
            [[320, 240, 618.663391, 617.159], [320, 240, 618.663391, 617.159]], tf.float32
        )  # (B, 4)

        self.empty_offset_mask = tf.expand_dims(
            tf.fill((*self.dataset_config.output_dims, 2), -1.0), axis=0
        )
        self.non_empty_offset_mask = tf.expand_dims(
            self.dataset_utils._generate_offset_mask(tf.constant([[100, 400]], tf.float32)), axis=0
        )
        self.offset_mask_batch = tf.concat(
            [self.empty_offset_mask, self.non_empty_offset_mask], axis=0
        )

    def test_shape(self):
        offset_mask_batched = tf.expand_dims(
            self.dataset_utils._generate_offset_mask(tf.constant([[100, 400]], tf.float32)), axis=0
        )

        distance_mask = self.dataset_utils.get_distance_mask_from_offsets(
            offset_mask_batched, self.camera[0:1, :], self.camera_intr[0:1, :], object_height=0.0
        )
        tf.print(tf.unique(tf.reshape(distance_mask, [-1])))
        tf.print("Distances: ", tf.shape(distance_mask))
        assert tf.reduce_all(
            tf.shape(distance_mask) == tf.constant([1, *self.dataset_config.output_dims, 1])
        )

    def test_mult_coords_one_offset_mask(self):
        # Index 0 and 1 are valid. Index 3 is invalid
        coords = tf.constant([[225, 400], [400, 300], [1, 4]], tf.float32)
        offset_mask_batched = tf.expand_dims(
            self.dataset_utils._generate_offset_mask(coords), axis=0
        )  # (B, W_out, H_out, 2)

        number_of_valid_coords = 2

        # Round the tensor to a tolerance to reduce values that are very close to each other.
        # These values a rounding errors and should be the same
        tolerance = 1e-4

        distance_mask = (
            tf.round(
                self.dataset_utils.get_distance_mask_from_offsets(
                    offset_mask_batched,
                    self.camera[0:1, :],
                    self.camera_intr[0:1, :],
                    object_height=0.0,
                )
                / tolerance
            )
            * tolerance
        )  # (B, W_out, H_out)

        _, _, unique_count_distances = tf.unique_with_counts(tf.reshape(distance_mask, [-1]))

        coords, _, unique_count_coordinates = tf.unique_with_counts(
            tf.round(
                tf.reshape(
                    self.dataset_utils.get_coordinate_mask(offset_mask_batched)[..., 0], [-1]
                )
                / tolerance
            )
            * tolerance
        )
        y, _ = tf.unique(tf.reshape(distance_mask, [-1]))
        tf.print(y)
        # tf.print(number_of_valid_coords)
        tf.print(unique_count_distances)
        tf.print(unique_count_coordinates)
        tf.print(coords)
        # Are distribution of distances and coordinates the same?
        assert tf.reduce_all(unique_count_distances == unique_count_coordinates)
        assert len(y) == number_of_valid_coords + 1

    def test_empty_offset_mask(self):
        distance_mask = self.dataset_utils.get_distance_mask_from_offsets(
            self.empty_offset_mask, self.camera[0:1, :], self.camera_intr[0:1, :], object_height=0.0
        )

        assert tf.reduce_all(distance_mask == -1.0)

    def test_mult_offset_masks(self):
        empty_offset_mask = tf.expand_dims(
            tf.fill((*self.dataset_config.output_dims, 2), -1.0), axis=0
        )
        non_empty_offset_mask = tf.expand_dims(
            self.dataset_utils._generate_offset_mask(tf.constant([[300, 400]], tf.float32)), axis=0
        )

        offset_mask_batch = tf.concat([empty_offset_mask, non_empty_offset_mask], axis=0)

        distance_mask = self.dataset_utils.get_distance_mask_from_offsets(
            offset_mask_batch, self.camera, self.camera_intr, object_height=0.0
        )

        # Round the tensor to a tolerance to reduce values that are very close to each other.
        # These values a rounding errors and should be the same
        tolerance = 1e-4
        y0, _ = tf.unique(tf.reshape(distance_mask[0], [-1]))
        y1, _ = tf.unique(tf.round(tf.reshape(distance_mask[1], [-1]) / tolerance) * tolerance)

        # First distance mask should be -1.0
        assert tf.reduce_all(distance_mask[0] == -1.0)
        # Second distance mask should be valid
        assert tf.reduce_all(distance_mask[1] >= 0)
        # Test number of unique elements in distance masks
        assert len(y0) == 1
        assert len(y1) == 1


@pytest.mark.parametrize("output_dims,input_dims, cell_dims", CONFIGS)
class TestGetCoordinateMask:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = input_dims
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

    def test_empty_mask(self):
        empty_offset_mask = tf.expand_dims(
            tf.fill((*self.dataset_config.output_dims, 2), -1.0), axis=0
        )
        result = self.dataset_utils.get_coordinate_mask(empty_offset_mask)

        assert tf.reduce_all(result == empty_offset_mask)

    def test_single_coordinate_pair(self):
        coord_pair = tf.constant([[400, 300]], tf.float32)  # (1, 2)

        offset_mask = tf.expand_dims(self.dataset_utils._generate_offset_mask(coord_pair), axis=0)

        expected = tf.tile(
            coord_pair[:, tf.newaxis, tf.newaxis, :],  # (1, 1, 1, 2)
            [1, *self.dataset_config.output_dims, 1],
        )  # (1, 15, 20, 2)
        result = self.dataset_utils.get_coordinate_mask(offset_mask)

        assert tf.reduce_all(tf.keras.ops.isclose(result, expected))


FULL_RES = np.array([480, 640])  # h, w


@pytest.mark.parametrize("output_dims, input_dims, cell_dims", CONFIGS)
class TestGetMasksClassification:
    @pytest.fixture(autouse=True)
    def setup(self, output_dims, input_dims, cell_dims):
        self.output_dims = np.array(output_dims)
        self.input_dims = np.array(input_dims)
        self.cell_dims = np.array(cell_dims)
        self.dataset_config = u_dataset.DatasetConfig(
            input_dims=input_dims, output_dims=output_dims, cell_dims=cell_dims
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

        # Cell size in full-resolution (480x640) coordinate space
        self.image_res_scale = self.input_dims / FULL_RES  # [h_scale, w_scale]
        self.full_ch = self.cell_dims[0] / self.image_res_scale[0]  # cell height in full-res pixels
        self.full_cw = self.cell_dims[1] / self.image_res_scale[1]  # cell width  in full-res pixels

    # ===== HELPER FUNCTIONS =====
    def _make_label(self, l_coords=None, t_coords=None, x_coords=None, ignore_sample=False):
        """Build a label dict with coordinates in full 480x640 resolution space."""

        def to_entry(coords):
            return [{"x": float(c[0]), "y": float(c[1])} for c in (coords or [])]

        return {
            u_dataset.CategoryNames.INTERSECTIONS.value: {
                "ignore_sample": ignore_sample,
                "L": to_entry(l_coords),
                "T": to_entry(t_coords),
                "X": to_entry(x_coords),
            }
        }

    def _get_classification_mask(self, label):
        return self.dataset_utils.get_masks(
            label=label, object_name=u_dataset.CategoryNames.INTERSECTIONS.value
        )["classification_mask"]

    def _cell_value(self, mask, full_res_coord):
        """Read the classification value for a coordinate given in full-res space."""
        scaled = tf.constant(full_res_coord, tf.float32) * self.dataset_config.image_res_scale[::-1]
        idx = self.dataset_utils.get_cell_of_coordinate(scaled)
        return mask[idx[1], idx[0]]

    # ===== TESTS =====
    def test_non_intersection_object_has_zero_classification_mask(self):
        label = {"some_object": {"x": self.full_cw / 2, "y": self.full_ch / 2, "radius": 5.0}}
        result = self.dataset_utils.get_masks(label=label, object_name="some_object")
        assert tf.reduce_all(result["classification_mask"] == 0)

    def test_empty_intersection_label_returns_zero_mask(self):
        label = self._make_label()
        mask = self._get_classification_mask(label)
        assert tf.reduce_all(mask == 0)

    def test_ignore_sample_returns_zero_mask(self):
        label = self._make_label(
            l_coords=[[self.full_cw / 2, self.full_ch / 2]], ignore_sample=True
        )
        mask = self._get_classification_mask(label)
        assert tf.reduce_all(mask == 0)

    def test_single_l_intersection(self):
        coord = [self.full_cw / 2, self.full_ch / 2]  # center of cell (0, 0)
        label = self._make_label(l_coords=[coord])
        mask = self._get_classification_mask(label)

        assert self._cell_value(mask, coord) == u_dataset.IntersectionType.L.value

    def test_single_t_intersection(self):
        coord = [self.full_cw + self.full_cw / 2, self.full_ch / 2]  # center of cell (0, 1)
        label = self._make_label(t_coords=[coord])
        mask = self._get_classification_mask(label)

        assert self._cell_value(mask, coord) == u_dataset.IntersectionType.T.value

    def test_single_x_intersection(self):
        coord = [2 * self.full_cw + self.full_cw / 2, self.full_ch / 2]  # center of cell (0, 2)
        label = self._make_label(x_coords=[coord])
        mask = self._get_classification_mask(label)

        assert self._cell_value(mask, coord) == u_dataset.IntersectionType.X.value

    def test_multiple_different_types_in_different_cells(self):
        l_coord = [self.full_cw / 2, self.full_ch / 2]  # cell (0, 0)
        t_coord = [self.full_cw + self.full_cw / 2, self.full_ch / 2]  # cell (0, 1)
        x_coord = [self.full_cw / 2, self.full_ch + self.full_ch / 2]  # cell (1, 0)

        label = self._make_label(l_coords=[l_coord], t_coords=[t_coord], x_coords=[x_coord])
        mask = self._get_classification_mask(label)

        tf.print(label)
        tf.print(mask)
        assert self._cell_value(mask, l_coord) == u_dataset.IntersectionType.L.value
        assert self._cell_value(mask, t_coord) == u_dataset.IntersectionType.T.value
        assert self._cell_value(mask, x_coord) == u_dataset.IntersectionType.X.value

    def test_cells_without_intersections_remain_zero(self):
        coord = [self.full_cw / 2, self.full_ch / 2]
        label = self._make_label(l_coords=[coord])
        mask = self._get_classification_mask(label)

        assert tf.reduce_sum(tf.cast(mask != 0, tf.int32)) == 1

    def test_two_coords_same_cell_only_one_entry(self):
        coord_a = [self.full_cw * 0.25, self.full_ch * 0.25]  # both in cell (0, 0)
        coord_b = [self.full_cw * 0.75, self.full_ch * 0.75]
        label = self._make_label(l_coords=[coord_a, coord_b])
        mask = self._get_classification_mask(label)

        assert tf.reduce_sum(tf.cast(mask != 0, tf.int32)) == 1

    def test_classification_mask_shape(self):
        coord = [self.full_cw / 2, self.full_ch / 2]
        label = self._make_label(l_coords=[coord])
        mask = self._get_classification_mask(label)

        assert tf.reduce_all(mask.shape == self.dataset_config.output_dims)

    def test_coordinates_input_returns_zero_classification_mask(self):
        coords = tf.constant([[self.full_cw / 2, self.full_ch / 2]], tf.float32)
        result = self.dataset_utils.get_masks(coordinates=coords)
        assert tf.reduce_all(result["classification_mask"] == 0)

    def test_multiple_different_intersection_in_single_cell(self):
        coord_a = [self.full_cw * 0.25, self.full_ch * 0.25]  # both in cell (0, 0)
        coord_b = [self.full_cw * 0.75, self.full_ch * 0.75]  # coord_b is lower

        label = self._make_label(l_coords=[coord_a], t_coords=[coord_b])
        mask = self._get_classification_mask(label)

        assert tf.reduce_sum(tf.cast(mask != 0, tf.int32)) == 1
        assert self._cell_value(mask, coord_a) == u_dataset.IntersectionType.T.value
