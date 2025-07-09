import numpy as np
import scipy
import tensorflow as tf

from . import camera as u_camera
from . import dataset as u_dataset


class MAE(tf.keras.metrics.Metric):
    def __init__(self, name="custom_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.abs_error = self.add_weight(name="abs_error", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold = self.get_threshold()

        img_coords_true = u_dataset.get_coords_from_offsets(y_true["offsets_ball"])
        img_coords_pred = u_dataset.get_coords_from_offsets(y_pred["offsets_ball"])

        wrld_coords_true = tf.expand_dims(
            u_camera.image_to_world(y_true["camera"], y_true["intrinsics"], img_coords_true), axis=0
        ).numpy()
        wrld_coords_pred = tf.expand_dims(
            u_camera.image_to_world(y_pred["camera"], y_pred["intrinsics"], img_coords_pred), axis=0
        ).numpy()

        # print("Img coords pred: ", img_coords_pred)
        # print("Img coords true: ", img_coords_true)
        # print("wrld coords pred: ", wrld_coords_pred)
        # print("wrld coords true: ", wrld_coords_true)

        match = self.match_keypoints(wrld_coords_pred, wrld_coords_true, e_max=threshold)

        for pair in match:
            # If all element is in pair are 0.0, do not count them towards the metric
            if (pair == 0).all():
                continue

            distance = tf.norm(pair[0] - pair[1])
            self.abs_error.assign_add(distance)
            self.num_samples.assign_add(1.0)

        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, "float32")
        #     values = tf.multiply(values, sample_weight)

    def get_threshold(self):
        return 0.2

    def result(self):
        print(self.abs_error)
        print(self.num_samples)
        return self.abs_error / self.num_samples

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.abs_error.assign(0.0)
        self.num_samples.assign(0.0)

    def match_keypoints(self, kps, pts, e_max):
        """Matches predicted and labeled points optimally. Points are only matched as
        long as their distance is below or equal to e_max. It does not match two
        points from kps to the same point in pts or vice versa. Correspondingly it can
        happen that some points remain unmatched.

        A matrix is constructed that contains a score for each pair of detected and
        labeled point. The score is 0 for pairs that are at least e_max pixels apart
        and 1 for points with distance 0 (cf. the error metric for point detectors from
        lecture 11a).

        Then scipy.optimize.linear_sum_assignment calculates the assignment that maximizes
        the overall score. Dummy assignments (with score 0) are filtered out before the
        matched points are stacked together to obtain the result.

        :param kps: numpy array of predicted keypoints (Mx2)
        :param pts: numpy array of labeled points (Nx2)
        :param e_max: max pixel distance to consider a match
        :return: numpy array of matches (k, p), where k is from kps and p the matched point from pts (Kx2x2)
        """
        if kps.size == 0 or pts.size == 0:
            return np.zeros((0, 2, 2), dtype=kps.dtype)
        diffs = kps[:, np.newaxis] - pts[np.newaxis]
        score_matrix = np.maximum(1 - np.linalg.norm(diffs, axis=-1) / e_max, 0)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(score_matrix, maximize=True)
        assigned = score_matrix[row_ind, col_ind] > 0
        return np.stack([kps[row_ind[assigned]], pts[col_ind[assigned]]], axis=1)
