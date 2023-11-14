import numpy as np
import pickle
import time
import util

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
    biases = np.zeros((1, output_size))

    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            # 順伝播の計算
            ## z = x * w + b
            z = np.dot(x, weights) + biases
            # z = util.dot(x, weights) + biases
            # a = softmax(z)
            a = relu(z)

            # 誤差の計算
            ## バックプロパゲーションによる誤差の計算
            dz = a - y
            dw = np.outer(x, dz)
            # dw = util.outer(x, dz)
            db = dz

            # 重みとバイアスの更新
            weights -= learning_rate * dw
            biases -= learning_rate * db

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
