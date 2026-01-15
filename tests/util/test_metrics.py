import tensorflow as tf

from src.util import metrics as u_metrics


class TestGetThresholdingMask:
    def testBaseCase(self):
        classifier_threshold = 0.8
        encoder_threshold = 0.1

        encoder_preds = tf.constant([[0.1, 0.4, 0.04], [0.9, 1, 0]], tf.float32)  # (2, 3)
        classifier_preds = tf.constant([[0.9, 0.8, 1], [0.3, 0, 0]], tf.float32)  # (2, 3)

        expected = tf.cast(tf.constant([[1, 1, 0], [0, 0, 0]]), tf.bool)
        results = u_metrics.get_thresholding_mask(
            classifier_preds, classifier_threshold, encoder_preds, encoder_threshold
        )

        assert tf.reduce_all(expected == results)

    def testFlatInput(self):
        classifier_threshold = 0.8
        encoder_threshold = 0.1

        encoder_preds = tf.constant([0.1, 0.4, 0.04, 0.9, 1, 0], tf.float32)  # (6)
        classifier_preds = tf.constant([0.9, 0.8, 1, 0.3, 0, 0], tf.float32)  # (6)

        expected = tf.cast(tf.constant([1, 1, 0, 0, 0, 0]), tf.bool)
        results = u_metrics.get_thresholding_mask(
            classifier_preds, classifier_threshold, encoder_preds, encoder_threshold
        )

        assert tf.reduce_all(expected == results)

    def testHighDimensions(self):
        classifier_threshold = 0.8
        encoder_threshold = 0.1

        encoder_preds = tf.constant([[[[[0.1], [0.4], [0.04]]], [[[0.9], [1], [0]]]]], tf.float32)
        classifier_preds = tf.constant([[[[[0.9], [0.8], [1]]], [[[0.3], [0], [0]]]]], tf.float32)

        expected = tf.cast(tf.constant([[[[[1], [1], [0]]], [[[0], [0], [0]]]]]), tf.bool)
        results = u_metrics.get_thresholding_mask(
            classifier_preds, classifier_threshold, encoder_preds, encoder_threshold
        )

        assert tf.reduce_all(expected == results)

    def testNoEncoderThresholds(self):
        classifier_threshold = 0.8

        classifier_preds = tf.constant([[0.9, 0.8, 1], [0.3, 0, 0]], tf.float32)  # (2, 3)

        expected = tf.cast(tf.constant([[1, 1, 1], [0, 0, 0]]), tf.bool)
        results = u_metrics.get_thresholding_mask(classifier_preds, classifier_threshold)

        assert tf.reduce_all(expected == results)
