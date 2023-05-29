import numpy as np


def power_iteration(X, max_iter = 10):
    m = X.shape[1]
    B = X.T @ X
    v = np.random.randn(m, 1)
    lam = 0
    for k in range(max_iter):
        y = B @ v
        lam = v.T @ y
        v = y / np.linalg.norm(y)

    sigma = np.sqrt(lam)
    u = X @ v / sigma
    return sigma, u, v


def power_iteration_SVD(X, max_iter = 10 , rank = None, trim = True):
    m, n = X.shape
    r = min(m, n)
    if rank is not None:
        if rank > r:
            raise Exception("The rank is maximally the smallest dimension of the matrix!")
        else:
            r = rank
    U = np.zeros((m, r))
    S = np.zeros((r, r))
    V = np.zeros((n, r))
    for i in range(r):
        sigma, u, v = power_iteration(X, max_iter)
        X = X - (u @ v.T) * sigma.item()
        U[:, i] = np.reshape(u, (m,))
        V[:, i] = np.reshape(v, (n,))
        S[i, i] = sigma
        if abs(sigma) < 1e-14:
            if rank is not None:
                print("The rank of the matrix is actually smaller than the target rank")
            if trim:
                return U[:, i], S[:i,:i], V[:, :i].T
            break

    return U, S, V.T



