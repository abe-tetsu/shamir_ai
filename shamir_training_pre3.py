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
    weights = np.ones((input_size, output_size)) * Accuracy_weight
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

            weights1_share = []
            weights2_share = []
            weights3_share = []
            for weight_row in weights:
                weights1_row = []
                weights2_row = []
                weights3_row = []
                for weight in weight_row:
                    shares = shamir.encrypt(int(weight), K, N, P)
                    weights1_row.append(shares[0])
                    weights2_row.append(shares[1])
                    weights3_row.append(shares[2])
                weights1_share.append(weights1_row)
                weights2_share.append(weights2_row)
                weights3_share.append(weights3_row)

            weights1 = np.array(weights1_share, dtype=np.int64)
            weights2 = np.array(weights2_share, dtype=np.int64)
            weights3 = np.array(weights3_share, dtype=np.int64)

            # dwを秘密分散
            # dw1 = []
            # dw2 = []
            # dw3 = []
            # for i in range(len(dw)):
            #     dw1_row = []
            #     dw2_row = []
            #     dw3_row = []
            #     for j in range(len(dw[i])):
            #         shares = shamir.encrypt(int(dw[i][j]), K, N, P)
            #         dw1_row.append(shares[0])
            #         dw2_row.append(shares[1])
            #         dw3_row.append(shares[2])
            #     dw1.append(dw1_row)
            #     dw2.append(dw2_row)
            #     dw3.append(dw3_row)
            #
            # dw1 = np.array(dw1, dtype=np.int64)
            # dw2 = np.array(dw2, dtype=np.int64)
            # dw3 = np.array(dw3, dtype=np.int64)

            # 重みとバイアスの更新
            # weights1 = (weights1 - learning_rate * dw1).astype(np.int64)
            # weights2 = (weights2 - learning_rate * dw2).astype(np.int64)
            # weights3 = (weights3 - learning_rate * dw3).astype(np.int64)

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

    weights, weights1, weights2, weights3 = train_network(x_train[:10000], y_train[:10000], epochs=2, learning_rate=0.001)
    save_weights(weights,  "weights_convert.pkl")
    save_weights(weights1,  "weights1_convert.pkl")
    save_weights(weights2, "weights2_convert.pkl")
    save_weights(weights3, "weights3_convert.pkl")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
