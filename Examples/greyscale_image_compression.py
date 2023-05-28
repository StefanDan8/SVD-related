import matplotlib.pyplot as plt
import numpy as np
import os

img = plt.imread(os.path.join('resources', 'peppers.png'))  # returns (M,N,3) numpy array in [0,1]
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # extract the channels
img_grayscale = 0.299 * R + 0.587 * G + 0.114 * B  # turn to grayscale
U, S, Vh = np.linalg.svd(img_grayscale)  # compute SVD
# plt.scatter(range(1,len(S)+1), S) #plot the singular values
ranks = [10, 25, 100]
compr_imgs = []
for i in range(3):
    compr_imgs.append(U[:, :ranks[i]] @ np.diag(S[:ranks[i]]) @ Vh[:ranks[i], :])  # compute the rank k approx
    residuum = img_grayscale - compr_imgs[i]  # compute residuals
    # evaluate errors
    print("Rank = {r} | Frobenius distance = {fd} | Spectral Distance = {sd}".format(
        r = ranks[i], fd = np.linalg.norm(residuum, 'fro'), sd = np.linalg.norm(residuum, 2)
    ))
    print("quick check for correctness: {k_plus_1}th singular value = {sigma}".format(
        k_plus_1 = ranks[i] + 1, sigma = S[ranks[i]]))  # this should be equal to the spectral norm of the residuum

# plotting the compressed images and the initial image in grayscale
f, axarr = plt.subplots(2, 2)
f.tight_layout(pad = 2.5)
plt.set_cmap('gray')
axarr[0][0].imshow(compr_imgs[0])  # rank 10
axarr[0][0].set_title('rank 10', fontsize = 10)
axarr[0][0].axis('off')
axarr[0][1].imshow(compr_imgs[1])  # rank 25
axarr[0][1].set_title('rank 25', fontsize = 10)
axarr[0][1].axis('off')
axarr[1][0].imshow(compr_imgs[2])  # rank 100
axarr[1][0].set_title('rank 100', fontsize = 10)
axarr[1][0].axis('off')
axarr[1][1].imshow(img_grayscale)  # rank 384
axarr[1][1].set_title('full rank', fontsize = 10)
axarr[1][1].axis('off')
plt.show()

