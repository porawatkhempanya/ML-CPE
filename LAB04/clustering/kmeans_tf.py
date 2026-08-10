import tensorflow as tf

class KMeansTensorFlow:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        num_samples = tf.shape(X)[0]
        
        # สุ่มเลือก Centroids เริ่มต้น
        random_indices = tf.random.shuffle(tf.range(num_samples))[:self.k]
        self.centroids = tf.gather(X, random_indices)

        for _ in range(self.max_iters):
            distances = tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - tf.expand_dims(self.centroids, 0)), axis=2)
            cluster_assignments = tf.argmin(distances, axis=1)

            new_centroids = []
            for i in range(self.k):
                mask = tf.equal(cluster_assignments, i)
                cluster_points = tf.boolean_mask(X, mask)
                
                if tf.shape(cluster_points)[0] > 0:
                    new_centroids.append(tf.reduce_mean(cluster_points, axis=0))
                else:
                    new_centroids.append(self.centroids[i])

            new_centroids = tf.stack(new_centroids)
            center_shift = tf.reduce_sum(tf.square(self.centroids - new_centroids))
            self.centroids = new_centroids

            if center_shift < self.tol:
                break

    def predict(self, X):
        distances = tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - tf.expand_dims(self.centroids, 0)), axis=2)
        return tf.argmin(distances, axis=1)

    def calculate_wcss(self, X):
        distances = tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - tf.expand_dims(self.centroids, 0)), axis=2)
        min_distances = tf.reduce_min(distances, axis=1)
        return tf.reduce_sum(min_distances).numpy()