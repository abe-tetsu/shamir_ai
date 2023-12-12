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
    # weights = np.zeros((input_size, output_size))
    # weights = np.ones((input_size, output_size)) * Accuracy_weight
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

    # 学習開始
    for epoch in range(epochs):
        counter = 0
        for x, y in zip(x_train, y_train):
            counter += 1
            if counter % 500 == 0:
                print("now:", counter)

            # 順伝播の計算
            z = np.dot(x, weights)
            a = relu(z)

            # 誤差の計算
            dz = a - y
            dw = np.outer(x, dz)

            # 重みの更新
            weights = (weights - learning_rate * dw).astype(np.int64)

            # -------------------------------------------

            # dzを秘密分散
            dz1 = []
            dz2 = []
            dz3 = []
            for i in range(len(dz)):
                shares = shamir.encrypt(int(dz[i]), K, N, P)
                dz1.append(shares[0])
                dz2.append(shares[1])
                dz3.append(shares[2])

            # xを秘密分散
            x1 = []
            x2 = []
            x3 = []
            for i in range(len(x)):
                shares = shamir.encrypt(int(x[i]), K, N, P)
                x1.append(shares[0])
                x2.append(shares[1])
                x3.append(shares[2])

            # 誤差の計算
            dw1 = np.outer(x1, dz1)
            dw2 = np.outer(x2, dz2)
            dw3 = np.outer(x3, dz3)

            # 重みとバイアスの更新
            weights1 = (weights1 - learning_rate * dw1).astype(np.int64)
            weights2 = (weights2 - learning_rate * dw2).astype(np.int64)
            weights3 = (weights3 - learning_rate * dw3).astype(np.int64)

        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")

    return weights, weights1, weights2, weights3


def save_weights(weights, filename="weights.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'weights': weights}, f)


def main():
    np.set_printoptions(100000)
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, weights1, weights2, weights3 = train_network(x_train[:1000], y_train[:1000], epochs=1, learning_rate=0.001)

    for index in range(len(weights)):
        dec = shamir.array_decrypt23(weights1[index], weights2[index], P)
        print("重み, 秘密分散前:", weights[index][0], weights[index][1], weights[index][2], weights[index][3], weights[index][4], weights[index][5], weights[index][6], weights[index][7], weights[index][8], weights[index][9])
        print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
        print("----------------------------")
        if not util.compare_arrays(weights[index], dec):
            print("error")
            print("index:", index)
            print("weights[index]:", weights[index])
            print("dec_weight:", dec)

    # for i in range(5):
    #     index = np.random.randint(0, len(weights))
    #     print("index:", index)
    #     dec = shamir.array_decrypt23(weights1[index], weights2[index], P)
    #     print("重み, 秘密分散前:", weights[index][0], weights[index][1], weights[index][2], weights[index][3], weights[index][4], weights[index][5], weights[index][6], weights[index][7], weights[index][8], weights[index][9])
    #     print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    #     print("----------------------------")

    save_weights(weights, "weights.pkl")
    save_weights(weights1, "weights1.pkl")
    save_weights(weights2, "weights2.pkl")
    save_weights(weights3, "weights3.pkl")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
