import mdp.offroad_grid as offroad_grid
import loader.offroad_loader as offroad_loader
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

grid_size = 80
discount = 0.9

loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False)
loader.data_list.sort()
data_list = loader.data_list
loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=False)

data_normalization = sio.loadmat('/data/datasets/yanfu/irl_data/train-data-mean-std.mat')


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

root = 'paper_dataset_viz'
for step, (feat, past_traj, future_traj) in tqdm(enumerate(loader)):
    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)
    past_traj_sample = past_traj[0].numpy()  # choose one sample from the batch
    past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
    past_traj_sample = past_traj_sample.astype(np.int64)

    base_name = data_list[step].split('/')[-1].split('.')[0]
    height_max = feat[0, 0].numpy()
    plt.imsave('{}/{}-hmax.png'.format(root,base_name), height_max)
    img = imageio.imread('{}/{}-hmax.png'.format(root,base_name))
    overlay_img = overlay(img, future_traj_sample, past_traj_sample)
    imageio.imwrite('{}/{}-hmax.png'.format(root,base_name), overlay_img)

    height_var = feat[0, 1].numpy()
    plt.imsave('{}/{}-hvar.png'.format(root,base_name), height_var)
    img = imageio.imread('{}/{}-hvar.png'.format(root,base_name))
    overlay_img = overlay(img, future_traj_sample, past_traj_sample)
    imageio.imwrite('{}/{}-hvar.png'.format(root,base_name), overlay_img)

    red = (feat[0, 2].numpy() * data_normalization['red_std'] + data_normalization['red_mean']).astype(np.uint8)
    green = (feat[0, 3].numpy() * data_normalization['green_std'] + data_normalization['green_mean']).astype(np.uint8)
    blue = (feat[0, 4].numpy() * data_normalization['blue_std'] + data_normalization['blue_mean']).astype(np.uint8)
    color = np.stack([red, green, blue], axis=2)
    overlay_color = overlay(color, future_traj_sample, past_traj_sample)
    imageio.imwrite('{}/{}-rgb.png'.format(root,base_name), overlay_color)
