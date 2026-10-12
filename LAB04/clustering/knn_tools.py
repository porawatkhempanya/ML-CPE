import tensorflow as tf


def find_nearest_neighbors(X_tf, k_neighbors=2):
    """ค้นหานักเตะที่มีค่าพลังใกล้เคียงกันมากที่สุด"""

    num_samples = int(tf.shape(X_tf)[0])

    r = tf.reduce_sum(tf.square(X_tf), axis=1, keepdims=True)

    distances = r - 2 * tf.matmul(X_tf, X_tf, transpose_b=True) + tf.transpose(r)
    distances = tf.maximum(distances, 0.0)
    distances = tf.sqrt(distances)

    _, top_k_idx = tf.nn.top_k(
        -distances,
        k=min(k_neighbors + 1, num_samples)
    )

    # ตัดตัวเองออก
    return top_k_idx.numpy()[:, 1:]