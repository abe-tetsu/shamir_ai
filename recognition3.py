# 画像認識 ver3
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


def recognition(random_idx, x_test, loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3):
    # for i in range(5):
    #     index = np.random.randint(0, len(loaded_weights))
    #     print("index:", index)
    #     dec = shamir.array_decrypt23(loaded_weights1[index], loaded_weights2[index], P)
    #     print("重み, 秘密分散前:", loaded_weights[index][0], loaded_weights[index][1], loaded_weights[index][2],
    #           loaded_weights[index][3], loaded_weights[index][4], loaded_weights[index][5], loaded_weights[index][6],
    #           loaded_weights[index][7], loaded_weights[index][8], loaded_weights[index][9])
    #     print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    #     print("----------------------------")


    # 検出用画像データを秘密分散する
    # 秘密分散は正の整数しか扱えないので、Accuracy_image倍してintに変換する
    test_image = x_test[random_idx]

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

    prediction0 = np.dot(test_image, loaded_weights)
    prediction1 = np.dot(test_image_shares1, loaded_weights1)
    prediction2 = np.dot(test_image_shares2, loaded_weights2)
    prediction3 = np.dot(test_image_shares3, loaded_weights3)
    # for i in range(10):
    #     zz = 0
    #     for j in range(784):
    #         zz += test_image[j] * loaded_weights[j][i]
    #     prediction0.append(zz)
    #
    # for i in range(10):
    #     zz = 0
    #     for j in range(784):
    #         zz += test_image_shares1[j] * loaded_weights1[j][i]
    #     prediction1.append(zz)
    #
    # for i in range(10):
    #     zz = 0
    #     for j in range(784):
    #         zz += test_image_shares2[j] * loaded_weights2[j][i]
    #     prediction2.append(zz)
    #
    # for i in range(10):
    #     zz = 0
    #     for j in range(784):
    #         zz += test_image_shares3[j] * loaded_weights3[j][i]
    #     prediction3.append(zz)

    # 予測を復元
    prediction = shamir.array_decrypt33(prediction1, prediction2, prediction3, P)
    # print("prediction0:", prediction0)
    # print("prediction :", prediction)

    return prediction, prediction0


    # prediction = []
    # for i in range(len(prediction0)):
    #     shares = [int(prediction1[i]), int(prediction2[i]), int(prediction3[i])]
    #     prediction.append(shamir.decrypt(shares, P))
    #     print("prediction:", prediction[i], "prediction0:", prediction0[i])
    #     time.sleep(2)
    #
    # prediction0 = [int(prediction0[i] * Accuracy_weight * Accuracy_image) for i in range(len(prediction0))]
    #
    # return prediction, prediction0


def main():
    print("P:", P, " P/2:", int(P/2))
    (x_train, _), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)
    loaded_weights, loaded_biases = util.load_weights()
    loaded_weights1, loaded_biases1 = util.load_encrypted_weight("weights1.pkl")
    loaded_weights2, loaded_biases2 = util.load_encrypted_weight("weights2.pkl")
    loaded_weights3, loaded_biases3 = util.load_encrypted_weight("weights3.pkl")

    # テストデータからランダムなインデックスを選択
    random_idx = np.random.randint(0, len(x_test))

    # 予測を実行
    prediction, prediction0 = recognition(random_idx, x_test, loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3)

    # 画像を表示
    plt.imshow(x_test[random_idx].reshape(28, 28), cmap="gray")
    plt.title(
        f"Before Shamir: {util.array_max(prediction0)}, After Shamir: {util.array_max(prediction)}, Actual: {np.argmax(y_test[random_idx])}")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
