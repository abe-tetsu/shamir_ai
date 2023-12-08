import numpy as np
import pickle
import time
import util
import shamir

P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight
Accuracy_image = util.Accuracy_image


def relu(x):
    return np.maximum(0, x)


def train_network(x_train, y_train, epochs, learning_rate):
    print("training start")
    print("k:", K)
    print("n:", N)
    print("p:", P)

    input_size = 784
    output_size = 10

    # 重みの初期値を秘密分散
    weights = (np.random.randn(input_size, output_size) + 100) * np.sqrt(2. / input_size) * Accuracy_weight
    weights = weights.astype('int64')

    weights1 = []
    weights2 = []
    weights3 = []
    for weight_row in weights:
        weights1_row = []
        weights2_row = []
        weights3_row = []
        for weight in weight_row:
            shares = shamir.encrypt(int(weight), K, N, P)
            weights1_row.append(shares[0])
            weights2_row.append(shares[1])
            weights3_row.append(shares[2])
        weights1.append(weights1_row)
        weights2.append(weights2_row)
        weights3.append(weights3_row)

    weights1 = np.array(weights1, dtype=np.int64)
    weights2 = np.array(weights2, dtype=np.int64)
    weights3 = np.array(weights3, dtype=np.int64)

    # バイアスの初期値を秘密分散
    # バイアスは全て1000にする
    # biases = np.array([10000] * output_size, dtype=np.int64)
    # biases_share = shamir.array_encrypt(biases, K, N, P)
    # biases1 = []
    # biases2 = []
    # biases3 = []
    # for bias in biases_share:
    #     biases1.append(int(bias[0]))
    #     biases2.append(int(bias[1]))
    #     biases3.append(int(bias[2]))
    #
    # biases1 = np.array(biases1, dtype=np.int64)
    # biases2 = np.array(biases2, dtype=np.int64)
    # biases3 = np.array(biases3, dtype=np.int64)

    # 学習開始
    for epoch in range(epochs):
        counter = 0
        for x, y in zip(x_train, y_train):
            counter += 1
            if counter % 500 == 0:
                print("now:", counter)
            # -------------------------------------------
            # 順伝播の計算
            # z = util.dot(x, weights) + biases
            z = np.dot(x, weights)
            a = relu(z)

            # 誤差の計算
            dz = a - y
            dw = np.outer(x, dz)
            # dw = util.outer(x, dz)

            # 重みとバイアスの更新
            weights = (weights - learning_rate * dw).astype(np.int64)
            # biases = (biases - learning_rate * dz).astype(np.int64)

            # -------------------------------------------
            # 順伝播の計算
            # z1 = util.dot(x, weights1) + biases1
            # z2 = util.dot(x, weights2) + biases2
            # z3 = util.dot(x, weights3) + biases3
            z1 = np.dot(x, weights1)
            z2 = np.dot(x, weights2)
            z3 = np.dot(x, weights3)
            a1 = relu(z1)
            a2 = relu(z2)
            a3 = relu(z3)

            # 掛け算を1回したので、(3, 3)閾値分散法になっている。
            # (2, 3)閾値分散法に変換し、再分配する。
            a1_converted, a2_converted, a3_converted = shamir.array_convert_shamir(a1, a2, a3, K, N, P)

            # 誤差の計算
            dz1 = a1_converted - y
            dz2 = a2_converted - y
            dz3 = a3_converted - y

            # dw1 = util.outer(x, dz1)
            # dw2 = util.outer(x, dz2)
            # dw3 = util.outer(x, dz3)
            dw1 = np.outer(x, dz1)
            dw2 = np.outer(x, dz2)
            dw3 = np.outer(x, dz3)

            # 掛け算を1回したので、(3, 3)閾値分散法になっている。
            # (2, 3)閾値分散法に変換し、再分配する。
            dw1_converted = []
            dw2_converted = []
            dw3_converted = []
            for dw1_row, dw2_row, dw3_row in zip(dw1, dw2, dw3):
                dw1_row_converted, dw2_row_converted, dw3_row_converted = shamir.array_convert_shamir(dw1_row, dw2_row,
                                                                                                      dw3_row, K, N, P)
                dw1_converted.append(dw1_row_converted)
                dw2_converted.append(dw2_row_converted)
                dw3_converted.append(dw3_row_converted)

            dw1_converted = np.array(dw1_converted, dtype=np.float64)
            dw2_converted = np.array(dw2_converted, dtype=np.float64)
            dw3_converted = np.array(dw3_converted, dtype=np.float64)

            # 重みとバイアスの更新
            weights1 = (weights1 - learning_rate * dw1_converted).astype(np.int64)
            weights2 = (weights2 - learning_rate * dw2_converted).astype(np.int64)
            weights3 = (weights3 - learning_rate * dw3_converted).astype(np.int64)

            # biases1 = (biases1 - learning_rate * dz1).astype(np.int64)
            # biases2 = (biases2 - learning_rate * dz2).astype(np.int64)
            # biases3 = (biases3 - learning_rate * dz3).astype(np.int64)

            # 重みの復元チェック
            # random_index = np.random.randint(0, 784)
            # dec = shamir.array_decrypt33(weights1[random_index], weights2[random_index], weights3[random_index], P)
            # print("秘密分散前:", weights[random_index][0], weights[random_index][1], weights[random_index][2], weights[random_index][3], weights[random_index][4], weights[random_index][5], weights[random_index][6], weights[random_index][7], weights[random_index][8], weights[random_index][9])
            # print("秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
            # time.sleep(1)
            # print("----------------------------")

        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")

    # dec = shamir.array_decrypt33(biases1, biases2, biases3, P)
    # print("バイアス, 秘密分散前:", biases[0], biases[1], biases[2], biases[3], biases[4], biases[5], biases[6], biases[7], biases[8], biases[9])
    # print("バイアス, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    # print("----------------------------")

    random_index = 0
    dec = shamir.array_decrypt33(weights1[random_index], weights2[random_index], weights3[random_index], P)
    print("重み, 秘密分散前:", weights[random_index][0], weights[random_index][1], weights[random_index][2],
          weights[random_index][3], weights[random_index][4], weights[random_index][5], weights[random_index][6],
          weights[random_index][7], weights[random_index][8], weights[random_index][9])
    print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    print("----------------------------")

    random_index = 10
    dec = shamir.array_decrypt33(weights1[random_index], weights2[random_index], weights3[random_index], P)
    print("重み, 秘密分散前:", weights[random_index][0], weights[random_index][1], weights[random_index][2],
          weights[random_index][3], weights[random_index][4], weights[random_index][5], weights[random_index][6],
          weights[random_index][7], weights[random_index][8], weights[random_index][9])
    print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    print("----------------------------")

    random_index = 20
    dec = shamir.array_decrypt33(weights1[random_index], weights2[random_index], weights3[random_index], P)
    print("重み, 秘密分散前:", weights[random_index][0], weights[random_index][1], weights[random_index][2],
          weights[random_index][3], weights[random_index][4], weights[random_index][5], weights[random_index][6],
          weights[random_index][7], weights[random_index][8], weights[random_index][9])
    print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    print("----------------------------")

    return weights, weights1, weights2, weights3


def save_weights(weights, biases, filename="weights.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases}, f)


def main():
    np.set_printoptions(100000)
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, weights1, weights2, weights3 = train_network(x_train[:10000], y_train[:10000], epochs=2, learning_rate=0.001)
    save_weights(weights, "")
    save_weights(weights1, "", "weights1_convert.pkl")
    save_weights(weights2, "", "weights2_convert.pkl")
    save_weights(weights3, "", "weights3_convert.pkl")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
