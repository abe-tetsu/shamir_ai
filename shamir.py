import random


def lagrange_interpreter(x_list, i):
    x_i = x_list[i]
    res = 1
    for cnt, x_atom in enumerate(x_list):
        if cnt != i:
            res *= (0 - x_atom) / (x_i - x_atom)
    return res


def lagrange(x_list, y_list):
    res = 0

    for n in range(len(x_list)):
        res += lagrange_interpreter(x_list, n) * y_list[n]
    return res


def decrypt(shares, p):
    # 整数じゃないときエラーを返す
    for share in shares:
        if int != type(share):
            print("share is not int type.")
            print("share:", share, "shares:", shares)
            print("type(share):", type(share))

    k = len(shares)
    x_list = [i + 1 for i in range(k)]
    y_list = shares
    f0 = lagrange(x_list, y_list)

    return int(f0 % p)


def encrypt(secret_int, k, n, p):
    # secret_int が int 型であることをチェックする
    # int型では無い場合、その値と、値の型をprintして、エラーを返す
    if type(secret_int) != int:
        print("secret_int is not int type.")
        print("secret_int:", secret_int)
        print("type(secret_int):", type(secret_int))
        raise ValueError("encrypt must be int type.")

    # 係数をランダムに決める
    a = [random.randint(0, 100) for _ in range(k - 1)]

    # n個のシェアを作成する
    shares = []
    for i in range(1, n + 1):
        share = 0
        for j in range(1, k):
            share += a[j - 1] * i ** j
        share += secret_int
        share %= p
        shares.append(share)

    # shares = [int(share) for share in shares]
    return shares
