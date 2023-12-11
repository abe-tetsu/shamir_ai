import random
import numpy


# def lagrange_interpreter(x_list, i):
#     x_i = x_list[i]
#     res = 1
#     for cnt, x_atom in enumerate(x_list):
#         if cnt != i:
#             res *= (0 - x_atom) / (x_i - x_atom)
#     return res


# def lagrange(x_list, y_list):
#     res = 0
#
#     for n in range(len(x_list)):
#         res += lagrange_interpreter(x_list, n) * y_list[n]
#     return res
#
#
# def decrypt(shares, p):
#     print("shares:",shares, "p:", p)
#     # 整数じゃないときエラーを返す
#     for share in shares:
#         if int != type(share) and numpy.int64 != type(share):
#             pass
#             # raise ValueError("decrypt function: share is not int type.", share, "type:", type(share))
#
#     k = len(shares)
#     x_list = [i + 1 for i in range(k)]
#     y_list = shares
#     f0 = lagrange(x_list, y_list)
#     f0_mod = int(f0 % p)
#
#     # 復元された値がpの半分より大きい場合、それを負の数として扱う
#     print("f0_mod:", f0_mod)
#     if f0_mod > p // 2:
#         print("f0_mod > p // 2")
#         print("f0_mod_before:", f0_mod)
#         f0_mod -= p
#         print("f0_mod_after:", f0_mod)
#
#     return f0_mod


def lagrange(x_list, y_list):
    res = 0
    for n in range(len(x_list)):
        res_n = y_list[n]
        for m in range(len(x_list)):
            if m != n:
                res_n *= (0 - x_list[m]) // (x_list[n] - x_list[m])
        res += res_n
    return res


def decrypt(shares, p):
    # print("shares:", shares)
    k = len(shares)
    x_list = [i + 1 for i in range(k)]
    y_list = shares
    f0 = lagrange(x_list, y_list)
    f0_mod = int(f0 % p)

    # 復元された値がpの半分より大きい場合、それを負の数として扱う
    if f0_mod > p // 2:
        # print("f0_mod > p // 2")
        f0_mod -= p

    return f0_mod


# (3, 3)閾値分散法にのみ対応
def array_decrypt33(shares1, shares2, shares3, p):
    res = []
    for share1, share2, share3 in zip(shares1, shares2, shares3):
        res.append(decrypt([int(share1), int(share2), int(share3)], p))

    return res


def array_decrypt23(shares1, shares2, p):
    res = []
    for share1, share2 in zip(shares1, shares2):
        res.append(decrypt([int(share1), int(share2)], p))

    return res


def encrypt(secret_int, k, n, p):
    # secret_int が int 型であることをチェックする
    # int型では無い場合、その値と、値の型をprintして、エラーを返す
    if type(secret_int) != int and type(secret_int) != numpy.int64:
        print("secret_int is not int type.")
        print("secret_int:", secret_int)
        print("type(secret_int):", type(secret_int))
        raise ValueError("encrypt must be int type.")

    # 係数をランダムに決める
    a = [random.randint(1000, 10000) for _ in range(k - 1)]

    # n個のシェアを作成する
    shares = []
    for i in range(1, n + 1):
        share = 0
        for j in range(1, k):
            share += a[j - 1] * i ** j
        share += secret_int
        share %= p
        shares.append(share)

        # print("secret_int:", secret_int)
        # print("係数:", a)
        # print("シェア:", shares)


        if share > p//2:
            print("超えたよ！", share, p)
            print("secret_int:", secret_int)
            print("係数:", a)
            print("シェア:", shares)

    # shares = [int(share) for share in shares]
    return shares


def array_encrypt(array1D, k, n, p):
    # 1次元配列の各要素を秘密分散する
    encrypted_array = [encrypt(int(element), k, n, p) for element in array1D]
    return encrypted_array


def convert_shamir(p1, p2, p3, k, n, p):
    p1_share = encrypt(int(p1), k, n, p)
    p2_share = encrypt(int(p2), k, n, p)
    p3_share = encrypt(int(p3), k, n, p)

    dec1 = decrypt([p1_share[0], p2_share[0], p3_share[0]], p)
    dec2 = decrypt([p1_share[1], p2_share[1], p3_share[1]], p)
    dec3 = decrypt([p1_share[2], p2_share[2], p3_share[2]], p)

    return dec1, dec2, dec3


def array_convert_shamir(p1_array, p2_array, p3_array, k, n, p):
    # 結果を格納するための配列を初期化
    dec1_array = []
    dec2_array = []
    dec3_array = []

    # 配列の長さが同じであることを確認
    if not (len(p1_array) == len(p2_array) == len(p3_array)):
        raise ValueError("All input arrays must have the same length.")

    # 各要素に対して convert_shamir 関数を適用
    for p1, p2, p3 in zip(p1_array, p2_array, p3_array):
        dec1, dec2, dec3 = convert_shamir(p1, p2, p3, k, n, p)
        dec1_array.append(dec1)
        dec2_array.append(dec2)
        dec3_array.append(dec3)

    return dec1_array, dec2_array, dec3_array


def array_convert_shamir_2d(p1_matrix, p2_matrix, p3_matrix, k, n, p):
    # 結果を格納するための配列を初期化
    dec1_matrix = []
    dec2_matrix = []
    dec3_matrix = []

    # 配列の長さが同じであることを確認
    if not (len(p1_matrix) == len(p2_matrix) == len(p3_matrix)):
        raise ValueError("All input matrices must have the same length.")

    # 各行に対して array_convert_shamir 関数を適用
    for p1_row, p2_row, p3_row in zip(p1_matrix, p2_matrix, p3_matrix):
        dec1_row, dec2_row, dec3_row = array_convert_shamir(p1_row, p2_row, p3_row, k, n, p)
        dec1_matrix.append(dec1_row)
        dec2_matrix.append(dec2_row)
        dec3_matrix.append(dec3_row)

    dec1_matrix = numpy.array(dec1_matrix)
    dec2_matrix = numpy.array(dec2_matrix)
    dec3_matrix = numpy.array(dec3_matrix)

    return dec1_matrix, dec2_matrix, dec3_matrix
