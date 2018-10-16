import numpy as np
import scipy.io as sio

def overlay(img, future_traj, past_traj):
    overlay_img = img.copy()
    for p in future_traj:
        overlay_img[int(p[0]), int(p[1]), 0] = 255  # red
        overlay_img[int(p[0]), int(p[1]), 1] = 255  # green
        overlay_img[int(p[0]), int(p[1]), 2] = 255  # blue
    for p in past_traj:
        overlay_img[int(p[0]), int(p[1]), 0] = 255
        overlay_img[int(p[0]), int(p[1]), 1] = 0
        overlay_img[int(p[0]), int(p[1]), 2] = 0
    return overlay_img

def feat2rgb(feat):
    normalization = sio.loadmat('example_data/data_mean_std.mat')
    red = (feat[2] * normalization['red_std'] + normalization['red_mean']).astype(np.uint8)
    green = (feat[3] * normalization['green_std'] + normalization['green_mean']).astype(np.uint8)
    blue = (feat[4] * normalization['blue_std'] + normalization['blue_mean']).astype(np.uint8)
    color = np.stack([red, green, blue], axis=2)
    return color