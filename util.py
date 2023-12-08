import time

import numpy as np
import pickle
from keras.datasets import mnist
import keras.utils
import shamir

P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = 10000
Accuracy_image = 2


def load_weights(filename="weights.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 重みの最小値を探索
    min_weight_value = np.min(data['weights'])

    # 最小値が負の場合、すべての重みにその値を加える
    if min_weight_value < 0:
        data['weights'] -= min_weight_value  # ここで負の最小値を加えることで、すべての重みが0以上になります

    return data['weights'], data['biases']

    # #重みの最小値を探索
    # min_weight_value = np.min(data['weights'])
    #
    # # 最小値が(-P/2)を下回った場合、エラーを出す
    # if min_weight_value < -P / 2:
    #     raise ValueError("Minimum weight value is less than -P/2.", min_weight_value)
    #
    # # 重みの最大値を探索
    # max_weight_value = np.max(data['weights'])
    #
    # # 最大値が(P/2)を上回った場合、エラーを出す
    # if max_weight_value > P / 2:
    #     raise ValueError("Maximum weight value is greater than P/2.", max_weight_value)
    #
    # return data['weights'], data['biases']


def load_encrypted_weight(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data['weights'], data['biases']


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print("load_data: OK")
    return (x_train, y_train), (x_test, y_test)


# dataすべてAccuracy_image倍してintに変換する
def transform_data(x_train, x_test):
    x_train_scaled_images = (x_train * Accuracy_image).astype(int)
    x_test_scaled_images = (x_test * Accuracy_image).astype(int)
    print("transform_data: OK")

    return x_train_scaled_images, x_test_scaled_images



def array_max(array):
    max = array[0]
    max_idx = 0
    for i in range(len(array)):
        if max < array[i]:
            max = array[i]
            max_idx = i
    return max_idx


def predict(x, weights, biases):
    # 入力ベクトルの長さを確認
    if len(x) != 784:
        raise ValueError("Input vector must have a length of 784.", len(x))

    # 重み行列の寸法を確認
    if len(weights) != 784 or len(weights[0]) != 10:
        raise ValueError("Weights must be a 784x10 dimensional matrix.")

    # バイアスの長さを確認
    if len(biases[0]) != 10:
        raise ValueError("Biases must have a length of 10: ", len(biases[0]))

    # 出力値を格納する配列を初期化（10個の出力ノードに対応）
    output_values = [0] * 10

    # 各出力ノードに対して線形結合を計算
    for i in range(10):  # 出力ノードの数だけループ
        # 重みと入力の積の合計を求める
        sum_weighted_inputs = 0
        for j in range(784):  # 各入力ノードについて
            sum_weighted_inputs += x[j] * weights[j][i]
        # バイアスを追加
        sum_weighted_inputs += biases[0][i]
        # 計算された値を出力値に設定
        output_values[i] = sum_weighted_inputs
        # print(output_values)

    return output_values


def debug_predict_1(x0, weight0, weight1, weight2, weight3, biases):
    # 出力値を格納する配列を初期化（10個の出力ノードに対応）
    output_values = [0] * 10

    for i in range(10):
        sum_weighted_inputs0 = 0
        for j in range(784):
            sum_weighted_inputs0 += x0 * weight0[j][i]
        output_values0 = sum_weighted_inputs0

        sum_weighted_inputs1 = 0
        for j in range(784):
            sum_weighted_inputs1 += x0 * weight1[j][i]
        output_values1 = sum_weighted_inputs1

        sum_weighted_inputs2 = 0
        for j in range(784):
            sum_weighted_inputs2 += x0 * weight2[j][i]
        output_values2 = sum_weighted_inputs2

        sum_weighted_inputs3 = 0
        for j in range(784):
            sum_weighted_inputs3 += x0 * weight3[j][i]
        output_values3 = sum_weighted_inputs3

        output_values123 = shamir.decrypt([output_values1, output_values2], P)

        print("秘密分散前:", int(output_values0))
        print("秘密分散後:", output_values123)

        time.sleep(5)

    return output_values


def debug_predict(x0, x1, x2, x3, weight0, weight1, weight2, weight3, biases):
    print("x0", x0[:3])
    print("x1", x1[:3])
    print("x2", x2[:3])
    print("x3", x3[:3])
    print("weight0", weight0[0][:3])
    print("weight1", weight1[0][:3])
    print("weight2", weight2[0][:3])
    print("weight3", weight3[0][:3])

    # 重みの復元チェック
    dec_weight = shamir.decrypt([weight1, weight2], P)

    output_values = [0] * 10

    for i in range(10):
        sum_weighted_inputs0 = 0
        for j in range(784):
            sum_weighted_inputs0 += x0[j] * weight0[j][i]
        # sum_weighted_inputs0 += biases[0][i]
        output_values0 = sum_weighted_inputs0

        sum_weighted_inputs1 = 0
        for j in range(784):
            sum_weighted_inputs1 += x1[j] * weight1[j][i]
        # sum_weighted_inputs1 += biases[0][i]
        output_values1 = sum_weighted_inputs1

        sum_weighted_inputs2 = 0
        for j in range(784):
            sum_weighted_inputs2 += x2[j] * weight2[j][i]
        # sum_weighted_inputs2 += biases[0][i]
        output_values2 = sum_weighted_inputs2

        sum_weighted_inputs3 = 0
        for j in range(784):
            sum_weighted_inputs3 += x3[j] * weight3[j][i]
        # sum_weighted_inputs3 += biases[0][i]
        output_values3 = sum_weighted_inputs3

        output_values123 = shamir.decrypt([output_values1, output_values2, output_values3], P)
        print("秘密分散前:", int(output_values0))
        print("秘密分散後:", output_values123)

        if output_values0 != output_values123:
            raise ValueError("error", output_values0, output_values123)

    time.sleep(1)
    return output_values


def dot(x, y):
    if len(x) != len(y):
        print(len(x), len(y))
        raise ValueError("Both input lists must have the same length.")
    return sum(i * j for i, j in zip(x, y))


def outer(x, y):
    result = []
    for i in x:
        for j in y:
            result.append(i * j)

    # dw[7840] から dw[784][10]に変換
    new_result = []
    for i in range(0, len(result), 10):
        new_result.append(result[i:i+10])

    return np.array(new_result, dtype=np.int64)
