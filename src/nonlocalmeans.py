# from numba import jit
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
# import cv2 as cv
from skimage.util import random_noise


# def load_images(path):
#     image = cv.imread(path, cv.IMREAD_GRAYSCALE)
#     image = np.divide(image, 255)
#     return image


def create_gaussian(shape=(7, 7), a=1.5):
    """
    This is used to create weights which is distributed as bivariate gaussian kernel
    """

    tmpx = (shape[0] - 1) / 2
    tmpy = (shape[1] - 1) / 2
    x_idx = np.arange(-tmpx, tmpx + 1)
    y_idx = np.arange(-tmpy, tmpy + 1)
    y_idx = y_idx.reshape(shape[1], 1)

    # bivariate gaussian
    gauss = np.exp(-(x_idx * x_idx + y_idx * y_idx) / (2 * a * a))

    # normalizing gaussian weights
    gauss = np.divide(gauss, np.sum(gauss))
    return gauss


def add_noise(image, mode="gaussian"):
    """
    This function is used to add noise of given mode
    """
    noisy_image = random_noise(image, mode)
    return noisy_image


# @jit()
def weighted_dist(i, j, w, var=0.1):
    """
    This function is used to calculate the weighted distance between 2 patchs
    i is one patch and j is another, w is the gaussian kernel
    """
    diff = i - j
    return (w * (diff * diff - 2 * var)).sum()


# @jit()
def nl_means(gt_image, patch_size, search_size, h, sigma, a, noise_mode="gaussian"):
    w = create_gaussian((patch_size, patch_size), a)

    # padding images so that corner pixels get proper justice-
    # zero padding would make it difficult for corner pixels to find similar patches
    to_pad = patch_size // 2
    padded_image = np.pad(gt_image, (to_pad, to_pad), mode="reflect")

    # initializing the output
    pred = np.zeros(gt_image.shape)

    # first 2 for loops to get the central pixel of the patch under consideration
    for i in range(to_pad, padded_image.shape[0] - to_pad):
        for j in range(to_pad, padded_image.shape[1] - to_pad):
            curr_patch = padded_image[i - to_pad:i + to_pad + 1, j - to_pad:j + to_pad + 1]

            # this is just to sum all the weights (as Z(i) does)
            total_sum = 0

            # going thorough search window
            for a in range(i - search_size // 2, i + (search_size // 2) + 1):

                # removing corner cases
                if a - to_pad < 0:
                    continue
                if a + to_pad >= padded_image.shape[0]:
                    break

                for b in range(j - search_size // 2, j + (search_size // 2) + 1):
                    if b - to_pad < 0 or (a == i and b == j):
                        continue
                    if b + to_pad >= padded_image.shape[1]:
                        break

                    search_patch = padded_image[a - to_pad:a + to_pad + 1, b - to_pad:b + to_pad + 1]

                    # finding weights for search window patch and current patch
                    weight = np.exp(-weighted_dist(curr_patch, search_patch, w, sigma ** 2) / h ** 2)
                    print(f'[x: {j}, y: {i}, a: {b}, a: {a}] weight={weight}')
                    total_sum += weight

                    # directly adding all the weighted patches. normalizing in end
                    pred[i - to_pad, j - to_pad] += weight * padded_image[a, b]

            # normalizing the pixel output
            pred[i - to_pad, j - to_pad] /= total_sum
    return pred


if __name__ == "__main__":
    noise_type = 's&p'  # this currently supports 'gaussian' and 's&p' noise modes
    patch_size = 7
    search_size = 21
    h = 0.1
    sigma = 0.1
    a = 1.5
    img_path = "images/bird.jpg"
    save_folder = "results"

    # image = load_images(img_path)
    # image = image[:150, :150]

    # Debugging config
    image = np.arange(20) * 10 / 255.0
    image = np.tile(image[np.newaxis, :], [20, 1])
    sigma = 0
    search_size = 3
    patch_size = 3
    pred = nl_means(image, patch_size, search_size, h, sigma, a, noise_type)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].imshow(pred, cmap='gray', vmin=0.0, vmax=1.0)
    plt.savefig('outputs/plots.pdf')
    plt.show()
