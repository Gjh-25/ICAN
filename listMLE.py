import tensorflow as tf

def listMLE(scores, labels):
    _, ranked_idx = tf.nn.top_k(tf.reshape(labels, [-1]),
                                k=tf.shape(labels)[0])

    y_pred = tf.gather(scores, ranked_idx)
    max_values = tf.reduce_max(y_pred, axis=0, keepdims=True)
    y_pred = y_pred - max_values
    exp_scores = tf.exp(y_pred)
    reverse_cumsum = tf.cumsum(exp_scores[::-1], axis=0)[::-1]
    log_prob = y_pred - tf.math.log(reverse_cumsum)
    loss = -tf.reduce_sum(log_prob)

    return loss

