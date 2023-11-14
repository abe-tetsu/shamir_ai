import unittest
from deep_learning import shamir
import random


class TestShamir(unittest.TestCase):
    def test_shamir_encrypt(self):
        secret = 100
        k = 2
        n = 3
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

    def test_shamir_decrypt(self):
        secret = 100
        shares = [102, 104]
        p = 3359
        dec_secret = shamir.decrypt(shares, p)
        self.assertEqual(dec_secret, secret)

    def test_shamir23(self):
        secret = random.randint(1, 100)
        k = 2
        n = 3
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_shamir34(self):
        secret = random.randint(1, 100)
        k = 3
        n = 4
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_shamir45(self):
        secret = random.randint(1, 100)
        k = 4
        n = 5
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_shamir56(self):
        secret = random.randint(1, 100)
        k = 5
        n = 6
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_lagrange(self):
        x = [1, 2, 3]
        y = [105, 116, 133]
        f0 = shamir.lagrange(x, y)

        self.assertEqual(f0, 100)

    def test_add_sub(self):
        secret1 = 4
        secret2 = 2
        k = 3
        n = 4
        p = 3359
        shares1 = shamir.encrypt(secret1, k, n, p)
        shares2 = shamir.encrypt(secret2, k, n, p)

        shares_sub = [shares1[i] - shares2[i] for i in range(n)]
        dec_secret = shamir.decrypt(shares_sub[:k], p)
        self.assertEqual(dec_secret, secret1 - secret2)

        shares_add = [shares1[i] + shares2[i] for i in range(n)]
        dec_secret = shamir.decrypt(shares_add[:k], p)
        self.assertEqual(dec_secret, secret1 + secret2)

    def test_int(self):
        # int の時正常に動作する
        secret = 100
        k = 2
        n = 3
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

        # float の時エラーが帰る
        secret = 100.0
        k = 2
        n = 3
        p = 3359

        with self.assertRaises(AssertionError):
            shares = shamir.encrypt(secret, k, n, p)
            self.assertEqual(len(shares), n)

            dec_secret = shamir.decrypt(shares[:k], p)
            self.assertEqual(dec_secret, secret)

        # 配列の時エラーが帰る
        secret = [100, 200]
        k = 2
        n = 3
        p = 3359

        with self.assertRaises(AssertionError):
            shares = shamir.encrypt(secret, k, n, p)
            self.assertEqual(len(shares), n)

            dec_secret = shamir.decrypt(shares[:k], p)
            self.assertEqual(dec_secret, secret)

    def test_try_dec(self):
        shares = [7,7,7]
        P = 3359
        dec = shamir.decrypt(shares, P)
        print(dec)

    def test_try(self):
        P = 3359
        K = 2
        N = 3

        secret1 = 10
        shares1 = shamir.encrypt(secret1, K, N, P)
        print(shares1)

        secret2 = 5
        shares2 = shamir.encrypt(secret2, K, N, P)
        print(shares2)

        dec1 = shamir.decrypt(shares1[:K], P)
        print(dec1)

        dec2 = shamir.decrypt(shares2[:K], P)
        print(dec2)

        w1 = shares2[0] * shares1[0]
        w2 = shares2[1] * shares1[1]
        w3 = shares2[2] * shares1[2]
        print(w1, w2, w3)

        dec2 = shamir.decrypt([w1, w2, w3], P)
        print(dec2)

    def test_shamir_sum(self):
        P = 3359
        K = 2
        N = 3

        e = 3
        x = 10
        x_shares = shamir.encrypt(x, K, N, P)
        print(x_shares)

        r = 5
        r_shares = shamir.encrypt(r, K, N, P)
        print(r_shares)

        s1 = r_shares[0] + e * x_shares[0]
        s2 = r_shares[1] + e * x_shares[1]
        s3 = r_shares[2] + e * x_shares[2]
        print(s1, s2, s3)

        dec = shamir.decrypt([s1, s2], P)
        print(dec) # 35


    def test_shamir_multi(self):
        P = 3359
        K = 2
        N = 3

        node1 = 5
        node2 = 6
        w1 = 10
        w2 = 9

        node1_share = shamir.encrypt(node1, K, N, P)
        node2_share = shamir.encrypt(node2, K, N, P)
        w1_share = shamir.encrypt(w1, K, N, P)
        w2_share = shamir.encrypt(w2, K, N, P)

        output1 = node1_share[0] * w1_share[0] + node2_share[0] * w2_share[0]
        output2 = node1_share[1] * w1_share[1] + node2_share[1] * w2_share[1]
        output3 = node1_share[2] * w1_share[2] + node2_share[2] * w2_share[2]

        dec = shamir.decrypt([output1, output2, output3], P)
        print(dec)

if __name__ == '__main__':
    unittest.main()
