import tensorflow as tf
import numpy as np
from utils import set_seed, MBFeatures
from listMLE import listMLE
import os


def normalize_adj_tf(A):
    N = tf.shape(A)[0]
    I = tf.eye(N, dtype=A.dtype)
    A_tilde = A + I
    d = tf.reduce_sum(A_tilde, axis=1)
    d_safe = tf.where(d > 0, d, tf.ones_like(d))
    d_inv_sqrt = tf.pow(d_safe, -0.5)
    D_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    return tf.matmul(tf.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)


def model_ICAN(xs, adj, ys, seed, iter1_max=20, iter2_max=10,
              lambda1=1.0, lambda2=1.0, lambda3=1.0,
              save_path='model_params'):
    tf.reset_default_graph()
    n, F = xs.shape
    set_seed(seed)

    n_hidden_1 = 32
    n_hidden_2 = 32
    n_hidden_3 = 33
    n_hidden_4 = 33
    gnn_hidden = 32

    gamma = 0.25
    beta = 10
    learning_rate = 0.001
    display_step = 100
    tol = 1e-4

    X = tf.placeholder(tf.float32, shape=[None, F])
    A = tf.placeholder(tf.float32, shape=[None, None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    alpha = tf.placeholder(tf.float32)
    rho = tf.placeholder(tf.float32)
    W_mb = tf.placeholder('float', shape=[1, n_hidden_1])
    A_norm = normalize_adj_tf(A)

    weights = {
        'encoder_w1': tf.Variable(tf.glorot_uniform_initializer()([F, n_hidden_1])),
        'encoder_w2': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_1, n_hidden_2])),
        'encoder_w3': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_2 + 1, n_hidden_3])),
        'encoder_w4': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_3, n_hidden_4])),
        'decoder_w1': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_4, n_hidden_3])),
        'decoder_w2': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_3, n_hidden_2 + 1])),
        'gnn_w1': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_2, gnn_hidden])),
        'gnn_w2': tf.Variable(tf.glorot_uniform_initializer()([gnn_hidden, gnn_hidden])),
        'fc_w': tf.Variable(tf.glorot_uniform_initializer()([gnn_hidden, 1]))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_4])),
        'decoder_b1': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_3])),
        'decoder_b2': tf.Variable(tf.glorot_uniform_initializer()([n_hidden_2 + 1])),
        'gnn_b1': tf.Variable(tf.zeros([gnn_hidden])),
        'gnn_b2': tf.Variable(tf.zeros([gnn_hidden])),
        'fc_b': tf.Variable(tf.zeros([1]))
    }

    def encoder(x, y, A_norm):
        layer_1 = tf.nn.relu(tf.matmul(A_norm, tf.matmul(x, weights['encoder_w1'])) + biases['encoder_b1'])
        layer_2 = tf.nn.relu(tf.matmul(A_norm, tf.matmul(layer_1, weights['encoder_w2'])) + biases['encoder_b2'])
        layer_2_1 = tf.concat([layer_2, y], axis=1)
        layer_3 = tf.nn.relu(tf.matmul(A_norm, tf.matmul(layer_2_1, weights['encoder_w3'])) + biases['encoder_b3'])
        layer_4 = tf.nn.relu(tf.matmul(A_norm, tf.matmul(layer_3, weights['encoder_w4'])) + biases['encoder_b4'])
        return layer_4, layer_2, layer_2_1

    def decoder(z):
        layer_1 = tf.nn.relu(tf.matmul(A_norm, tf.matmul(z, weights['decoder_w1'])) + biases['decoder_b1'])
        layer_2 = tf.nn.relu(tf.matmul(A_norm, tf.matmul(layer_1, weights['decoder_w2'])) + biases['decoder_b2'])
        layer_2_1 = layer_2[:, :-1]
        recon_logits = tf.matmul(layer_2_1, layer_2_1, transpose_b=True)
        recon_prob = tf.sigmoid(recon_logits)
        return recon_logits, recon_prob
    W_init = tf.Variable(tf.random_uniform([n_hidden_4, n_hidden_4], -0.1, 0.1))
    W = tf.linalg.set_diag(W_init, tf.zeros(n_hidden_4))  #消除自环

    Z, H2, H2_y = encoder(X, y, A_norm)
    Z_causal = tf.matmul(Z, W)
    logits_A_hat, A_hat = decoder(Z_causal)

    loss_adj_recon = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=A, logits=logits_A_hat))
    loss_l2w = tf.add_n([tf.reduce_mean(tf.square(v)) for v in weights.values()])
    H_mb =tf.multiply(H2, W_mb)
    loss_ranking_scores = listMLE(gnn_model(A,H_mb, weights, biases), y)
    h = tf.linalg.trace(tf.linalg.expm(W * W)) - n_hidden_4
    loss_fun = lambda1 * loss_adj_recon + lambda2 * loss_l2w + lambda3 * loss_ranking_scores + alpha * h + 0.5 * rho * h * h
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_fun)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        alpha_val = 0.0
        rho_val = 1.0
        h_val = 0
        W_mb_val = np.ones([1, n_hidden_1])
        for i in range(iter2_max):
            print('the number of iter2 is:', i)
            loss_1 = 0
            for j in range(iter1_max):
                _, loss, l_ar, l2_w, l_k, W_curr, h_curr = sess.run(
                    [optimizer, loss_fun, loss_adj_recon, loss_l2w, loss_ranking_scores, W, h],
                    feed_dict={X: xs, A: adj, y: ys, alpha: alpha_val,
                               rho: rho_val,
                               W_mb: W_mb_val})
                if np.abs(loss - loss_1) <= tol:
                    break
                loss_1 = loss
                if j % display_step == 0 or j == 1:
                    print('the number of iter1 is:', j)
                    print('loss is:', loss_1)
            X_new, w1, w2, b1, b2, W_new, h_new = sess.run(
                [H2, weights['encoder_w1'], weights['encoder_w2'], biases['encoder_b1'],
                 biases['encoder_b2'], W, h], feed_dict={X: xs, A: adj, y: ys, alpha: alpha_val, rho: rho_val, W_mb: W_mb_val})
            W_new1 = np.copy(W_new)
            _, W_mb_array = MBFeatures(W_new1)
            W_mb_val = W_mb_array.T
            alpha_new = alpha_val + rho_val * h_new
            if np.abs(h_new) >= gamma * np.abs(h_val):
                rho_new = beta * rho_val
            else:
                rho_new = rho_val
            alpha_val = alpha_new
            rho_val = rho_new
            h_val = h_new

        save_dict = {k: sess.run(v) for k, v in weights.items()}
        save_dict.update({k: sess.run(v) for k, v in biases.items()})
        save_dict['W'] = sess.run(W)


        pa, mb = MBFeatures(save_dict['W'])
        feature_mask_val = W_mb_val.astype(np.int32)  # numpy 数组
        np.save(os.path.join(save_path, "feature_mask.npy"), feature_mask_val)
        print("Saved feature_mask.npy with shape:", feature_mask_val.shape)
        H2_val = sess.run(H2, feed_dict={X: xs, A: adj, y: ys})
        return H2_val * feature_mask_val, feature_mask_val

def gnn_model(A,features, weights, biases):
    h1=tf.matmul(A, tf.matmul(features, weights['gnn_w1'])) + biases['gnn_b1']
    h1 = tf.sigmoid(h1)
    h2 = tf.matmul(A, tf.matmul(h1, weights['gnn_w2'])) + biases['gnn_b2']
    h2 = tf.sigmoid(h2)
    out = tf.matmul(h2, weights['fc_w']) + biases['fc_b']
    return out
