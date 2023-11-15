import time
import numpy as np
import recognition1
import util

P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight
Accuracy_image = util.Accuracy_image
TestNum = 1000


def main():
    test_start = time.time()

    (_, _), (x_test, y_test) = util.load_data()
    loaded_weights, loaded_biases = util.load_weights()

    correct_count = 0
    correct_count_before_shamir = 0
    random_idx = np.random.randint(0, len(x_test) - TestNum)

    for i in range(TestNum):
        test_index = random_idx + i
        prediction_shamir, prediction = recognition1.recognition(test_index, x_test, loaded_weights, loaded_biases)
        if np.argmax(prediction_shamir) == np.argmax(y_test[test_index]):
            correct_count += 1
        if np.argmax(prediction) == np.argmax(prediction_shamir):
            correct_count_before_shamir += 1

    print(f"テストデータ数　　　　　　: {TestNum}")
    accuracy = correct_count / TestNum
    print(f"精度　　　　　　　　　　　: {accuracy * 100:.2f}%")

    accuracy_before_shamir = correct_count_before_shamir / TestNum
    print(f"秘密分散前の出力との一致率: {accuracy_before_shamir * 100:.2f}%")

    test_end = time.time()
    print(f"テスト時間　　　　　　　　: {test_end - test_start} seconds")


if __name__ == '__main__':
    main()
