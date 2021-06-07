import numpy as np

from skimage import data
from skimage.exposure import histogram

coins = data.coins()

hist, hist_centers = histogram(coins)

from skimage.feature import canny

edges = canny(coins / 255.)

from scipy import ndimage as ndi

fill_coins = ndi.binary_fill_holes(edges)
label_objects, nb_labels = ndi.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

from skimage.filters import sobel

elevation_map = sobel(coins)

from skimage.segmentation import watershed

segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
