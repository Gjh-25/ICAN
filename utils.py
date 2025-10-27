import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing
import random
import os
import tensorflow as tf



def construct_adjacency(features, mb_mask):
    selected_indices = tf.where(tf.equal(mb_mask[0], 1))[:, 0]
    num_selected = tf.shape(selected_indices)[0]
    X_mb_features = tf.gather(features, selected_indices, axis=1)
    adjacency_matrix = tf.cond(
        tf.equal(num_selected, 0),
        lambda: tf.eye(1, dtype=tf.float32),  # 无选中特征时返回单位矩阵
        lambda: tf.matmul(X_mb_features, X_mb_features, transpose_b=True)  # 相似度矩阵
    )

    adjacency_matrix = tf.matrix_set_diag(
        adjacency_matrix,
        tf.zeros(tf.shape(adjacency_matrix)[0], dtype=tf.float32)
    )
    return adjacency_matrix

def knn_classify(xs, ys, xt, yt, k=1):
    model = KNeighborsClassifier(n_neighbors=k)
    ys = ys.ravel()
    yt = yt.ravel()
    model.fit(xs, ys)
    yt_pred = model.predict(xt)
    acc = accuracy_score(yt, yt_pred)
    return acc


def svm_classify(xs, ys, xt, yt, c=1, gamma='auto'):
    model = svm.SVC(C=c, kernel='rbf', gamma=gamma, decision_function_shape='ovr')
    ys = ys.ravel()
    yt = yt.ravel()
    model.fit(xs, ys)
    yt_pred = model.predict(xt)
    acc = accuracy_score(yt, yt_pred)
    return acc

def linearsvm_classify(xs, ys, xt, yt):
    model = svm.LinearSVC(random_state=42, tol=1e-3)
    ys = ys.ravel()
    yt = yt.ravel()
    model.fit(xs, ys)
    yt_pred = model.predict(xt)
    acc = accuracy_score(yt, yt_pred)
    return acc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def MBFeatures(array):
    n, p = array.shape
    array_new = np.copy(array)
    array_new[np.abs(array_new) >= 0.1] = 1
    array_new[np.abs(array_new) < 0.1]= 0
    p = p - 1
    parents = np.copy(array_new[0:p, p:p + 1])
    children = np.copy(array_new[ p:p + 1, 0:p])
    MB = parents + np.transpose(children)
    for j in range(p):
        if children[0, j] == 1:
            MB = MB + array_new[0:p, j:j + 1]
    MB[np.abs(MB) >= 1] = 1
    return parents, MB


def MBFeatures_thres(array, threshold=0.3):
    n, p = array.shape
    array_new = np.copy(array)
    array_new[np.abs(array_new) >= threshold] = 1
    array_new[np.abs(array_new) < threshold] = 0
    p = p - 1
    parents = np.copy(array_new[0:p, p:p + 1])
    children = np.copy(array_new[ p:p + 1, 0:p])
    MB = parents + np.transpose(children)
    for j in range(p):
        if children[0, j] == 1:
            MB = MB + array_new[0:p, j:j + 1]
    MB[np.abs(MB) >= 1] = 1
    return parents, MB, array_new


def Y_New(Y_in):
    cluster = np.unique(Y_in)
    Y = np.zeros((Y_in.shape[0], cluster.__len__()), dtype=int)
    for i in range(Y_in.shape[0]):
        for j in range(cluster.__len__()):
            if Y_in[i] == cluster[j]:
                Y[i][j] = 1
    return Y, cluster


def mysigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def getnew_xt(xt, w1, w2, b1, b2):
    layer_1 = mysigmoid(np.matmul(xt, w1) + b1)
    layer_2 = mysigmoid(np.matmul(layer_1, w2) + b2)
    return layer_2


def getnew_xt_m1(xt, w1, b1):
    layer_1 = mysigmoid(np.matmul(xt, w1) + b1)
    # layer_2 = mysigmoid(np.matmul(layer_1, w2) + b2)
    return layer_1


def getnew_xt_m3(xt, w1, w2, w2_1, b1, b2, b2_1):
    layer_1 = mysigmoid(np.matmul(xt, w1) + b1)
    layer_2 = mysigmoid(np.matmul(layer_1, w2) + b2)
    layer_3 = mysigmoid(np.matmul(layer_2, w2_1) + b2_1)
    return layer_3


def getnew_xt_m4(xt, w1, w2, w2_1, w2_2, b1, b2, b2_1, b2_2):
    layer_1 = mysigmoid(np.matmul(xt, w1) + b1)
    layer_2 = mysigmoid(np.matmul(layer_1, w2) + b2)
    layer_3 = mysigmoid(np.matmul(layer_2, w2_1) + b2_1)
    layer_4 = mysigmoid(np.matmul(layer_3, w2_2) + b2_2)
    return layer_4


def getnew_xt_m5(xt, w1, w2, w2_1, w2_2, w2_3, b1, b2, b2_1, b2_2, b2_3):
    layer_1 = mysigmoid(np.matmul(xt, w1) + b1)
    layer_2 = mysigmoid(np.matmul(layer_1, w2) + b2)
    layer_3 = mysigmoid(np.matmul(layer_2, w2_1) + b2_1)
    layer_4 = mysigmoid(np.matmul(layer_3, w2_2) + b2_2)
    layer_5 = mysigmoid(np.matmul(layer_4, w2_3) + b2_3)
    return layer_5


def normalize_adj(A):
    """Â = D^{-1/2}(A+I)D^{-1/2}. Works with dynamic N."""
    N = tf.shape(A)[0]
    I = tf.eye(N)
    A_tilde = A + I
    d = tf.reduce_sum(A_tilde, axis=1)
    d_inv_sqrt = tf.pow(d, -0.5)
    D_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    return tf.matmul(tf.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)


