import numpy as np
import ot

def compute_wasserstein_barycenter(images, reg=1e-1):
    """
    Compute the Wasserstein barycenter of a list or array of 2D grayscale images.

    Parameters:
        images (list or np.ndarray): List or array of 2D numpy arrays (H x W).
        reg (float): Entropic regularization parameter for Sinkhorn algorithm.

    Returns:
        np.ndarray: 2D array (H x W) representing the Wasserstein barycenter.
    """
    if isinstance(images, list):
        images = np.stack(images, axis=0)

    n_images, h, w = images.shape
    n_pixels = h * w

    # Normalize each image to make it a probability distribution
    distributions = []
    for img in images:
        img = img / img.sum()
        distributions.append(img.flatten())

    distributions = np.array(distributions).T  # shape: (n_pixels, n_images)

    # Create cost matrix on the 2D pixel grid
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    C = ot.utils.dist(coords, coords)
    C /= C.max()

    # Compute the barycenter
    bary = ot.bregman.barycenter(distributions, C, reg)

    return bary.reshape((h, w))
