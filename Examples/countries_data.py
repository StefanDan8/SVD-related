from Algorithms.power_iteration import *
import pandas as pd
import numpy as np
import os

np.set_printoptions(suppress = True)

# Exercise 5.2 Sheet 5 MA4800

df = pd.read_csv(os.path.join('resources', 'countries_data.txt'), skiprows = 2, delim_whitespace = True)
df = df.melt('[columns]').pivot('variable', '[columns]', 'value').reset_index().rename_axis(columns = None)
A = (df.loc[:, df.columns[1:]]).to_numpy()

# a) first singular value, first left and right singular vectors
sigma1, u1, v1 = power_iteration(A)

# b) Compute the best k-rank approximation to the matrix A for different
# values of k and analyze the error.
k = [1, 2, 3]
for i in k:
    U1, S1, Vh1 = power_iteration_SVD(A, rank = i)
    A1 = U1 @ S1 @ Vh1
    print("Rank: {r}, Error in Frobenius norm: {err}".format(
        r = i, err = np.linalg.norm(A - A1, 2)
    ))

# c)Compute the best r-rank approximation to
# the matrix A with r = rank (A) using your program from b). Compare the two obtained
# representations of A computing the norm of error matrices.
k = np.linalg.matrix_rank(A)
U, S, Vt = np.linalg.svd(A, full_matrices = False)
U1, S1, Vh1 = power_iteration_SVD(A, rank = k)
print(U.shape, np.diag(S).shape, Vt.shape)
print(np.linalg.norm(U @ np.diag(S) @ Vt - U1 @ S1 @ Vh1))
