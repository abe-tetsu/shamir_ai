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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def train_network(x_train, y_train, epochs, learning_rate):
    input_size = 784
    output_size = 10

    weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
    weights1 = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
    weights2 = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
    weights3 = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
    biases = np.zeros((1, output_size))
    biases1 = np.zeros((1, output_size))
    biases2 = np.zeros((1, output_size))
    biases3 = np.zeros((1, output_size))

    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            # 画像データとラベルを秘密分散
            x_shares = shamir.array_encrypt(x, K, N, P)
            y_shares = shamir.array_encrypt(y, K, N, P)

            x0 = [subarray[0] for subarray in x_shares]
            x1 = [subarray[1] for subarray in x_shares]
            x2 = [subarray[2] for subarray in x_shares]

            y0 = [subarray[0] for subarray in y_shares]
            y1 = [subarray[1] for subarray in y_shares]
            y2 = [subarray[2] for subarray in y_shares]

            # 順伝播の計算
            z = util.dot(x, weights) + biases
            z0 = util.dot(x0, weights1) + biases1
            z1 = util.dot(x1, weights2) + biases2
            z2 = util.dot(x2, weights3) + biases3

            # 誤差の計算
            dz = z - y
            dz0 = z0 - y0
            dz1 = z1 - y1
            dz2 = z2 - y2

            dw = util.outer(x, dz)
            dw0 = util.outer(x0, dz0)
            dw1 = util.outer(x1, dz1)
            dw2 = util.outer(x2, dz2)

            # 重みとバイアスの更新
            weights -= learning_rate * dw
            biases -= learning_rate * dz

            weights1 -= learning_rate * dw0
            weights2 -= learning_rate * dw1
            weights3 -= learning_rate * dw2
            biases1 -= learning_rate * dz0
            biases2 -= learning_rate * dz1
            biases3 -= learning_rate * dz2

            # if weights == shamir.decrypt([weights1, weights2, weights3], P):
            #     print("秘密分散前と後で重みが一致しません")
            #     print("秘密分散前:", weights)
            #     print("秘密分散後:", [weights1, weights2, weights3], shamir.decrypt([weights1, weights2, weights3], P))
            #     time.sleep(5)

    print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")
    return weights, biases


def save_weights(weights, biases, filename="weights.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases}, f)


def load_weights(filename="weights.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['weights'], data['biases']


def main():
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, biases = train_network(x_train, y_train, epochs=3, learning_rate=0.001)
    save_weights(weights, biases)

    loaded_weights, loaded_biases = load_weights()

    accuracy = 0
    for i in range(len(x_test)):
        prediction = util.predict(x_test[i], loaded_weights, loaded_biases)
        if util.array_max(prediction) == util.array_max(y_test[i]):
            accuracy += 1

    print(f"Accuracy: {accuracy / len(x_test) * 100:.2f}%")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
