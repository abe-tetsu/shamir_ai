import util
import shamir
import pickle

K = 2
N = 3
P = pow(2, 62) - 1


def save_weights(weights, biases, filename="weights.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases}, f)


# weightを秘密分散する
def main():
    # weightをロード
    weights, biases = util.load_weights()

    # weightを秘密分散するために、util.Accuracy_weight倍する
    newWeights = []
    for weight_row in weights:
        newWeights_row = []
        for weight in weight_row:
            newWeights_row.append(int(weight * util.Accuracy_weight))
        newWeights.append(newWeights_row)

    print(newWeights)
    print(len(newWeights))
    print(len(newWeights[0]))

    # weightを秘密分散
    weights_shares1 = []
    weights_shares2 = []
    weights_shares3 = []
    for i in range(len(weights)):
        shares = shamir.array_encrypt(newWeights[i], K, N, P)
        weights_shares1_row = []
        weights_shares2_row = []
        weights_shares3_row = []
        for j in range(len(shares)):
            weights_shares1_row.append(shares[j][0])
            weights_shares2_row.append(shares[j][1])
            weights_shares3_row.append(shares[j][2])
        weights_shares1.append(weights_shares1_row)
        weights_shares2.append(weights_shares2_row)
        weights_shares3.append(weights_shares3_row)

    print(len(newWeights), len(newWeights[0]))
    print(len(weights_shares1), len(weights_shares1[0]))

    # 秘密分散したweightを保存
    save_weights(weights_shares1, biases, "weights1.pkl")
    save_weights(weights_shares2, biases, "weights2.pkl")
    save_weights(weights_shares3, biases, "weights3.pkl")


if __name__ == '__main__':
    main()
