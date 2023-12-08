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

    # バイアスの初期値を秘密分散
    # バイアスは全て1000にする
    biases = np.zeros((1, output_size))
    biases = biases[0]


    # 学習開始
    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            # 順伝播の計算
            z = np.dot(x, weights) + biases
            a = relu(z)

            # if z != a:
            if np.any(z != a):
                print(z)
                print(a)
                time.sleep(2)

            # 誤差の計算
            ## バックプロパゲーションによる誤差の計算
            dz = a - y
            dw = np.outer(x, dz)

            # 重みとバイアスの更新
            weights = (weights - learning_rate * dw).astype(np.int64)
            biases = (biases - learning_rate * dz).astype(np.int64)


        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")

    # 重みとバイアスを秘密分散
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

    return weights, weights1, weights2, weights3, biases, biases1, biases2, biases3


def save_weights(weights, biases, filename="weights.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases}, f)


def main():
    np.set_printoptions(100000)
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, weights1, weights2, weights3, biases, biases1, biases2, biases3 = train_network(x_train, y_train, epochs=2, learning_rate=0.001)
    save_weights(weights, biases)
    save_weights(weights1, biases1, "weights1.pkl")
    save_weights(weights2, biases2, "weights2.pkl")
    save_weights(weights3, biases3, "weights3.pkl")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
