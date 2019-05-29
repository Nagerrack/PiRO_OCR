import numpy as np
from skimage.color import rgb2hsv


def remove_vert_surrounding(img, img_hsv):
    hues = img_hsv[:, :, 0]
    sats = img_hsv[:, :, 1]

    upper_cut = int(img_hsv.shape[0] * 0.2)
    lower_cut = int(img_hsv.shape[0] * 0.7)

    hues_short = hues[upper_cut:lower_cut]
    sats_short = sats[upper_cut:lower_cut]

    means_hues = [np.mean(hues_short[i]) for i in range(hues_short.shape[0] // 2)]
    global_mean_hues = np.mean(means_hues) * 0.85

    means_sats = [np.mean(sats_short[i]) for i in range(sats_short.shape[0] // 2)]
    global_mean_sats = np.mean(means_sats) * 2.4

    if global_mean_hues > 0.4 and global_mean_sats > 0.15:
        rows_to_delete = [i for i in range(hues.shape[0]) if np.mean(hues[i]) < global_mean_hues]
    else:
        rows_to_delete = [i for i in range(sats.shape[0]) if np.mean(sats[i]) > global_mean_sats]

    return np.delete(img, rows_to_delete, 0), rows_to_delete


def remove_horiz_surrounding(img, img_hsv):
    upper_cut = int(img_hsv.shape[0] * 0.1)
    lower_cut = int(img_hsv.shape[0] * 0.9)
    left_cut = int(img_hsv.shape[1] * 0.05)
    right_cut = int(img_hsv.shape[1] * 0.75)

    hues_t = img_hsv[:, :, 0].transpose()
    hues_short_t = hues_t[left_cut:right_cut]

    means_hues = [np.mean(hues_short_t[i]) for i in range(hues_short_t.shape[0] // 2)]
    global_mean_hues = np.mean(means_hues) * 0.75

    cols_to_delete = [i for i, hue in enumerate(hues_t) if np.mean(hue) < global_mean_hues]
    deleted_cols = cols_to_delete

    img = np.delete(img, cols_to_delete, 1)
    img_hsv = np.delete(img_hsv, cols_to_delete, 1)

    left_cut = int(img_hsv.shape[1] * 0.08)
    right_cut = int(img_hsv.shape[1] * 0.75)

    sats_t = img_hsv[:, :, 1].transpose()
    sats_t_short = sats_t[left_cut:right_cut]
    sats_t_left = sats_t[:left_cut]
    sats_t_right = sats_t[right_cut:]

    means_sats_left = [max(np.mean(col[:upper_cut]), np.mean(col[lower_cut:])) for col in sats_t_left]
    means_sats_right = [max(np.mean(col[:upper_cut]), np.mean(col[lower_cut:])) for col in sats_t_right]

    means_sats = [np.mean(sats_t_short[i]) for i in range(sats_t_short.shape[0] // 2)]
    global_mean_sats = np.mean(means_sats) * 7

    cols_to_delete = [i for i, mean in enumerate(means_sats_left) if mean > global_mean_sats]
    cols_to_delete.extend([i + right_cut - 1 for i, mean in enumerate(means_sats_right) if mean > global_mean_sats])
    deleted_cols.extend(cols_to_delete)

    return np.delete(img, cols_to_delete, 1), deleted_cols


def remove_surrounding(img):
    img_hsv = rgb2hsv(img)
    deleted = {'rows': 0, 'columns': 0}

    img, deleted['rows'] = remove_vert_surrounding(img, img_hsv)
    img, deleted['columns'] = remove_horiz_surrounding(img, img_hsv)

    return img, deleted
