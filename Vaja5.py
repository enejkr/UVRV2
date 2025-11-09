import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage

import naloga_2_resitve as n2  # use the module to access functions and patch alias


def _draw_plus(size=21, thickness=3) -> np.ndarray:
    k = size // 2
    img = np.zeros((size, size), dtype=np.float32)
    img[k - thickness//2:k + (thickness+1)//2, :] = 1.0  # horizontal bar
    img[:, k - thickness//2:k + (thickness+1)//2] = 1.0  # vertical bar
    return img


def _draw_x(size=21, thickness=3) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    d1 = np.abs(yy - xx)
    d2 = np.abs(yy - (size - 1 - xx))
    mask = (d1 <= thickness//2) | (d2 <= thickness//2)
    img[mask] = 1.0
    return img


def _augment(patch: np.ndarray, noise=0.05) -> np.ndarray:
    p = patch.copy()
    if noise > 0:
        p = p + np.random.normal(0, noise, p.shape).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    # light blur to mimic real data
    p = cv2.GaussianBlur(p, (3, 3), 0)
    return p


def build_synthetic_patches(n=8, size=21, thickness=3):
    plus = _draw_plus(size, thickness)
    cross = _draw_x(size, thickness)
    plus_patches = [_augment(plus, noise=0.05) for _ in range(n)]
    x_patches = [_augment(cross, noise=0.05) for _ in range(n)]
    # convert to 3-channel for robustness with the filter prep (it handles both)
    plus_patches = [np.stack([p, p, p], axis=-1) for p in plus_patches]
    x_patches = [np.stack([p, p, p], axis=-1) for p in x_patches]
    return plus_patches, x_patches


def main():
    # Patch the alias bug in naloga_2_resitve: detekcija_plus uses `ndi.maximum_filter`
    # while the module imports `scipy.ndimage as ndimage`.
    n2.ndi = ndimage  # make sure maximum_filter is available

    # Build a synthetic filter for quick testing
    plus_patches, x_patches = build_synthetic_patches(n=12, size=23, thickness=3)
    w = n2.pripravi_filter_plus(plus_patches, x_patches)
    # Load image and drop alpha if present (plt.imread may return RGBA)
    slika = plt.imread('Primeri/primer_5.png')
    if slika.ndim == 3 and slika.shape[-1] == 4:
        slika = slika[..., :3]

    # Run detection with the prepared filter
    coords_yx = n2.detekcija_plus(slika, w)  # returns (N, 2) as (y, x)

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.title('Detekcija "+"')
    plt.imshow(slika)
    if coords_yx.size > 0:
        plt.plot(coords_yx[:, 1], coords_yx[:, 0], 'ro', markersize=4)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    main()
