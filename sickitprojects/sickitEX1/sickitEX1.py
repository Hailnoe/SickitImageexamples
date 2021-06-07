"""
Sick it Example 1 from Sickit-Image.org

Reformatted by Trey Noe

This file is based on the first example and runs a simple program that identifies coins through an elevation map, 
but fails to accomplish the same task using the "canny" map method of the same image. The elevation
map succeeds because it does not rely on one pixel of a different color or a "coin-shaped" ring of unbroken
pixels instead merely relying on smaller sections of said coin to pick out patterns that would indicate a coin being
present in the image. Though the canny method is effective it can fail if it misses a single pixel.
I also reformatted the example so that way all imported materials are at the top instead of continuously added
throughout the exercises. If you want to see the original go to 'sickitEX1/original.py'.
"""

import numpy as np

from skimage import data
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

coins = data.coins()
hist, hist_centers = histogram(coins)

edges = canny(coins / 255.)
fill_coins = ndi.binary_fill_holes(edges)
label_objects, nb_labels = ndi.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2
elevation_map = sobel(coins)
segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
