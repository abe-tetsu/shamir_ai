# 画像認識 ver1
import time
import numpy as np
import matplotlib.pyplot as plt
import shamir
import util

P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight
Accuracy_image = util.Accuracy_image


def recognition(random_idx, x_test, loaded_weights, loaded_biases):
    # 784*10個の重みに対して、それぞれ秘密分散する
    loaded_weights1 = np.zeros((784, 10))
    loaded_weights2 = np.zeros((784, 10))
    loaded_weights3 = np.zeros((784, 10))

    for i in range(len(loaded_weights)):
        for j in range(len(loaded_weights[0])):
            # 重みは float64 型なので、10000000倍して、int型に変換する
            int_weights = int(loaded_weights[i][j] * Accuracy_weight)

            shares = shamir.encrypt(int_weights, K, N, P)
            # if int_weights != shamir.decrypt(shares[:K], P):
            #     raise ValueError("error", int_weights, shamir.decrypt(shares[:K], P))

            loaded_weights1[i][j] = int(shares[0])
            loaded_weights2[i][j] = int(shares[1])
            loaded_weights3[i][j] = int(shares[2])

    test_image = x_test[random_idx]

    # 予測を実行
    prediction0 = util.predict(test_image, loaded_weights, loaded_biases)
    prediction1 = util.predict(test_image, loaded_weights1, loaded_biases)
    prediction2 = util.predict(test_image, loaded_weights2, loaded_biases)
    prediction3 = util.predict(test_image, loaded_weights3, loaded_biases)

    # 予測を復元
    prediction = []
    for i in range(len(prediction0)):
        shares = [int(prediction1[i]), int(prediction2[i])]
        prediction.append(shamir.decrypt(shares, P))

    # prediction0の要素を全てAccuracy倍してintに変換する　
    prediction0 = [int(prediction0[i] * Accuracy_weight) for i in range(len(prediction0))]

    return prediction, prediction0


def main():
    (x_train, _), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)
    loaded_weights, loaded_biases = util.load_weights()

    # テストデータからランダムなインデックスを選択
    random_idx = np.random.randint(0, len(x_test))

    # 予測を実行
    prediction, prediction0 = recognition(random_idx, x_test, loaded_weights, loaded_biases)

    # 画像を表示
    plt.imshow(x_test[random_idx].reshape(28, 28), cmap="gray")
    plt.title(
        f"Before Shamir: {util.array_max(prediction0)}, After Shamir: {util.array_max(prediction)}, Actual: {np.argmax(y_test[random_idx])}")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
