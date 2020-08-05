import numpy as np


npzfile = np.load('mask.npz')
mask_indices = npzfile["mask_as_start_and_end_indices"]
print(mask_indices)