# Implementation from Data Driven Science and Engineering, Brunton & Kutz

import numpy as np



def rSVD(X, r, q, p):
    """
    Computes the randomized SVD of a matrix X .

            Parameters:
                    X (…, M, N) array_like: matrix to be factorized
                    r (int): target rank
                    q (int): number of power iterations
                    p (int): oversampling factor

            Returns:
                    U, S, Vh
    """
    # Step 1 : Sample column space of X with P matrix
    ny = X.shape[1]  #number of columns of X
    P = np.random.randn(ny, r+p)  # projection matrix
    Z = X @ P
    # power iteration q steps
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode = 'reduced')
    # Step 2: Compute the SVD on projected Y = Q.T @ Y
    Y = Q.T @ X
    UY, S, Vh = np.linalg.svd(Y, full_matrices = False)
    U = Q @ UY
    return U, S, Vh



