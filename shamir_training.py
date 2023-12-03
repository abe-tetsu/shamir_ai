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
    biases = np.array([10000] * output_size, dtype=np.int64)
    biases_share = shamir.array_encrypt(biases, K, N, P)
    biases1 = []
    biases2 = []
    biases3 = []
    for bias in biases_share:
        biases1.append(int(bias[0]))
        biases2.append(int(bias[1]))
        biases3.append(int(bias[2]))

    biases1 = np.array(biases1, dtype=np.int64)
    biases2 = np.array(biases2, dtype=np.int64)
    biases3 = np.array(biases3, dtype=np.int64)

    # 学習開始
    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            def decrypt_array_shamir23(shamir1, shamir2, shamir3, secret, label):
                dec_shamir = shamir.array_decrypt23(shamir1, shamir2, P)
                for i in range(len(shamir1)):
                    if int(dec_shamir[i]) != int(secret[i]):
                        print(label, "の値: 秘密分散前と後が一致しません。")
                        print("秘密分散前:", secret[i])
                        print("秘密分散後:", dec_shamir[i])
                        print()
                        print("秘密分散前:", secret)
                        print("秘密分散後:", dec_shamir)
                        exit()
                print(label, "passed!")

            def decrypt_array_shamir33(shamir1, shamir2, shamir3, secret, label):
                dec_shamir = shamir.array_decrypt33(shamir1, shamir2, shamir3, P)
                for i in range(len(shamir1)):
                    if int(dec_shamir[i]) != int(secret[i]):
                        print(label, "の値: 秘密分散前と後が一致しません。")
                        print("秘密分散前:", secret[i])
                        print("秘密分散後:", dec_shamir[i])
                        # print()
                        # print("秘密分散前:", secret)
                        # print("秘密分散後:", dec_shamir)
                        exit()
                print(label, "passed!")

            # decrypt_array_shamir23(weights1[0], weights2[0], weights3[0], weights[0], "重み[0]")

            # 画像データとラベルを秘密分散
            x_shares = shamir.array_encrypt(x, K, N, P)
            y_shares = shamir.array_encrypt(y, K, N, P)

            x0 = [subarray[0] for subarray in x_shares]
            x1 = [subarray[1] for subarray in x_shares]
            x2 = [subarray[2] for subarray in x_shares]

            y0 = [subarray[0] for subarray in y_shares]
            y1 = [subarray[1] for subarray in y_shares]
            y2 = [subarray[2] for subarray in y_shares]

            # decrypt_array_shamir23(x0, x1, x2, x, "x")
            # decrypt_array_shamir23(y0, y1, y2, y, "y")
            # decrypt_array_shamir23(biases1, biases2, biases3, biases, "バイアス")

            # 順伝播の計算
            z = util.dot(x, weights) + biases
            z0 = util.dot(x0, weights1) + biases1
            z1 = util.dot(x1, weights2) + biases2
            z2 = util.dot(x2, weights3) + biases3

            # decrypt_array_shamir33(z0, z1, z2, z, "順伝播")

            # 掛け算を1回したので、(3, 3)閾値分散法になっている。
            # (2, 3)閾値分散法に変換し、再分配する。
            z0_transformed, z1_transformed, z2_transformed = shamir.array_convert_shamir(z0, z1, z2, K, N, P)

            # decrypt_array_shamir23(z0_transformed, z1_transformed, z2_transformed, z, "再分配後の順伝播")

            # 誤差の計算
            # dz1 = x1 * w1 + x2 * w2 + x3 * w3 + b1 - y1
            dz = z - y
            dz0 = []
            for z, y in zip(z0_transformed, y0):
                dz0.append(z-y)
            dz1 = []
            for z, y in zip(z1_transformed, y1):
                dz1.append(z-y)
            dz2 = []
            for z, y in zip(z2_transformed, y2):
                dz2.append(z-y)

            dw = util.outer(x, dz)
            dw0 = util.outer(x0, dz0)
            dw1 = util.outer(x1, dz1)
            dw2 = util.outer(x2, dz2)

            # dwをweightsと同じ形状にリシェイプ
            dw_reshaped = dw.reshape(weights.shape)
            dw0_reshaped = dw0.reshape(weights1.shape)
            dw1_reshaped = dw1.reshape(weights2.shape)
            dw2_reshaped = dw2.reshape(weights3.shape)

            # 重みとバイアスの更新
            weights = (weights - learning_rate * dw_reshaped).astype(np.int64)
            biases = (biases - learning_rate * dz).astype(np.int64)

            weights1 = (weights1 - learning_rate * dw0_reshaped).astype(np.int64)
            weights2 = (weights2 - learning_rate * dw1_reshaped).astype(np.int64)
            weights3 = (weights3 - learning_rate * dw2_reshaped).astype(np.int64)

            for i in range(len(biases1)):
                biases1[i] -= learning_rate * dz0[i]
            for i in range(len(biases2)):
                biases2[i] -= learning_rate * dz1[i]
            for i in range(len(biases3)):
                biases3[i] -= learning_rate * dz2[i]


            #重みの復元チェック
            # dec = shamir.array_decrypt33(weights1[0], weights2[0], weights3[0], P)
            # print("秘密分散前:", int(weights[0][0]), int(weights[0][1]), int(weights[0][2]), int(weights[0][3]),
            #       int(weights[0][4]), int(weights[0][5]), int(weights[0][6]), int(weights[0][7]), int(weights[0][8]),
            #       int(weights[0][9]))
            # print("秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
            # print("----------------------------------")
            # time.sleep(1)

        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")
    return weights, weights1, weights2, weights3, biases, biases1, biases2, biases3


def save_weights(weights, biases, filename="weights.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases}, f)


def main():
    np.set_printoptions(100000)
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, weights1, weights2, weights3, biases, biases1, biases2, biases3 = train_network(x_train[:10000], y_train[:10000], epochs=3, learning_rate=0.001)
    save_weights(weights, biases)
    save_weights(weights1, biases1, "weights1.pkl")
    save_weights(weights2, biases2, "weights2.pkl")
    save_weights(weights3, biases3, "weights3.pkl")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
