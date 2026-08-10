import tensorflow as tf


class KNNTensorFlow:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # คำนวณระยะ Euclidean Distance
        r_train = tf.reduce_sum(tf.square(self.X_train), axis=1, keepdims=True)
        r_test = tf.reduce_sum(tf.square(X_test), axis=1, keepdims=True)

        distances = (
            r_test
            - 2 * tf.matmul(X_test, self.X_train, transpose_b=True)
            + tf.transpose(r_train)
        )

        distances = tf.maximum(distances, 0.0)
        distances = tf.sqrt(distances)

        # หา K ตัวที่ใกล้ที่สุด
        _, top_k_indices = tf.nn.top_k(-distances, k=self.k)

        # ดึง Label ของเพื่อนบ้าน
        top_k_labels = tf.gather(self.y_train, top_k_indices)

        predictions = []

        for labels in top_k_labels.numpy():
            values, _, counts = tf.unique_with_counts(tf.constant(labels))
            predictions.append(values[tf.argmax(counts)].numpy())

        return tf.constant(predictions, dtype=tf.int32)