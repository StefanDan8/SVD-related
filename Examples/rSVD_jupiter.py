from Algorithms.randomized_SVD import rSVD
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import time

A = imread(os.path.join('resources', 'jupiter.jpg'))  # 2260 x 3207 pixels
X = np.mean(A, axis = 2)  # grayscale

st = time.time()
U, S, Vh = np.linalg.svd(X, full_matrices = False)  # classic SVD
et = time.time()
elapsed_time_SVD = et-st

r = 20  # target rank
q = 1  # Power iterations
p = 5  # Oversampling parameter

st = time.time()
rU, rS, rVh = rSVD(X, r, q, p)
et = time.time()
elapsed_time_rSVD = et-st
# Reconstruction
XSVD = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
XrSVD = rU[:, :r] @ np.diag(rS[:r]) @ rVh[:r, :]
errXSVD = np.linalg.norm(X - XSVD, ord = 2) / np.linalg.norm(X, ord = 2)
errXrSVD = np.linalg.norm(X - XrSVD, ord = 2) / np.linalg.norm(X, ord = 2)

print("Normal SVD error in 2-norm: {err}".format(err = errXSVD))
print("Randomized SVD error in 2-norm: {err}".format(err = errXrSVD))

print("Elapsed time computing normal SVD: {et} seconds".format(et = elapsed_time_SVD))
print("Elapsed time computing randomized SVD: {et} seconds".format(et = elapsed_time_rSVD))

# Normal SVD error in 2-norm: 0.0003919737506944072
# Randomized SVD error in 2-norm: 0.0005036988811025861
# Elapsed time computing normal SVD: 11.160741806030273 seconds
# Elapsed time computing randomized SVD: 0.8241932392120361 seconds


# Plot
fig, axs = plt.subplots(1, 3)

plt.set_cmap('gray')
axs[0].imshow(X)
axs[0].set_title("Original")
axs[0].axis('off')
axs[1].imshow(XSVD)
axs[1].set_title("Normal SVD")
axs[1].axis('off')
axs[2].imshow(XrSVD)
axs[2].set_title("Randomized SVD")
axs[2].axis('off')
plt.show()
