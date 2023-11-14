# 画像認識 ver2
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

            # numpy.float64 型に戻す
            loaded_weights1[i][j] = np.float64(shares[0])
            loaded_weights2[i][j] = np.float64(shares[1])
            loaded_weights3[i][j] = np.float64(shares[2])

            if int_weights != shamir.decrypt(shares[:K], P):
                ValueError("error", int_weights, shamir.decrypt(shares[:K], P))

    # 検出用画像データを秘密分散する
    # 秘密分散は正の整数しか扱えないので、Accuracy_image倍してintに変換する
    test_image = []
    for i in range(len(x_test[random_idx])):
        test_image.append(int(x_test[random_idx][i] * Accuracy_image))

    # 画像を秘密分散
    test_image_shares1 = []
    test_image_shares2 = []
    test_image_shares3 = []
    for i in range(len(test_image)):
        shares = shamir.encrypt(test_image[i], K, N, P)
        test_image_shares1.append(shares[0])
        test_image_shares2.append(shares[1])
        test_image_shares3.append(shares[2])

        if test_image[i] != shamir.decrypt(shares[:K], P):
            ValueError("error", test_image[i], shamir.decrypt(shares[:K], P))

    prediction0 = util.predict(x_test[random_idx], loaded_weights, loaded_biases)
    prediction1 = util.predict(test_image_shares1, loaded_weights1, loaded_biases)
    prediction2 = util.predict(test_image_shares2, loaded_weights2, loaded_biases)
    prediction3 = util.predict(test_image_shares3, loaded_weights3, loaded_biases)

    # 予測を復元
    prediction = []
    for i in range(len(prediction0)):
        shares = [prediction1[i], prediction2[i], prediction3[i]]
        prediction.append(shamir.decrypt(shares, P))

    prediction0 = [int(prediction0[i] * Accuracy_weight * Accuracy_image) for i in range(len(prediction0))]

    return prediction, prediction0


def main():
    (x_train, _), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)
    loaded_weights, loaded_biases = util.load_weights()

    # テストデータからランダムなインデックスを選択
    random_idx = np.random.randint(0, len(x_test))

    # 予測を実行
    prediction, prediction0 = recognition(random_idx, x_test, loaded_weights, loaded_biases)
    print("秘密分散前:", prediction0)
    print("秘密分散前予測結果:", util.array_max(prediction0))
    print("秘密分散後:", prediction)
    print("秘密分散後予測結果:", util.array_max(prediction))

    # 画像を表示
    # plt.imshow(x_test[random_idx].reshape(28, 28), cmap="gray")
    # plt.title(
    #     f"Before Shamir: {util.array_max(prediction0)}, After Shamir: {util.array_max(prediction)}, Actual: {np.argmax(y_test[random_idx])}")
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    main()
