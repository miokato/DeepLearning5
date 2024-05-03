import numpy as np


def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return y


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)  # 行列式
    inv = np.linalg.inv(cov)  # 逆行列
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


def gmm(x, phis, mus, covs):
    """ 混合ガウスモデル """
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


def likelihood(xs, phis, mus, covs):
    """ 対数尤度 """
    eps = 1e-8
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N