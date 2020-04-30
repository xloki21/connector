import os
import numpy as np
from PIL import Image
import concurrent.futures as futures


def imwrite(array, filename):
    im = Image.fromarray(array)
    im.save(filename)


def imread_full(filename):
    im = Image.open(filename)
    im.load()
    return im


def imread_lazy(filename):
    return Image.open(filename)


def pil_to_nparray(pil_image):
    return np.array(pil_image)


def create_grid(im_shape, patch_shape, dx, dy, padding=True):
    w, h = im_shape
    ph, pw = patch_shape

    if w >= pw:
        i = np.arange(w)[:-pw + 1:dx]
        if i[-1] != w - pw:
            i = np.hstack([i, w - pw])
    else:
        i = np.array([0], dtype=np.int64)
        if not padding:
            pw = w

    if h >= ph:
        j = np.arange(h)[:-ph + 1:dy]
        if j[-1] != h - ph:
            j = np.hstack([j, h - ph])
    else:
        j = np.array([0], dtype=np.int64)
        if not padding:
            ph = h

    x, y = np.meshgrid(i, j)

    xmin = x.reshape(-1, )
    ymin = y.reshape(-1, )
    xmax = xmin + pw
    ymax = ymin + ph

    return xmin, ymin, xmax, ymax


def save_image_roi(image, roi, filename):
    # todo: test
    width, height = image.size
    xmin, ymin, xmax, ymax = roi

    # if image size lesser than roi size
    # xmax = np.clip(xmax, a_min=0, a_max=width)
    # ymax = np.clip(ymax, a_min=0, a_max=height)

    patch = image.crop((xmin, ymin, xmax, ymax))

    patch.save(filename)
    return patch


def crop_patches(image_filepath,
                 patch_shape,
                 dx,
                 dy,
                 patch_root_dir,
                 ext=None,
                 padding=True,
                 max_workers=None):
    image = imread_full(image_filepath)
    image_filename = os.path.basename(image_filepath)
    image_basename = os.path.splitext(image_filename)[0]

    xmin, ymin, xmax, ymax = create_grid(im_shape=image.size,
                                         patch_shape=patch_shape,
                                         dx=dx,
                                         dy=dy,
                                         padding=padding)
    if ext is None:
        ext = os.path.splitext(image_filepath)[1]
    elif ext[0] != '.':
        ext = '.' + ext

    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_patch = {
            executor.submit(save_image_roi,
                            image,
                            roi,
                            os.path.join(patch_root_dir,
                                         "{basename}_{ymin:04}_{xmin:04}{ext}".format(basename=image_basename,
                                                                                      xmin=roi[0],
                                                                                      ymin=roi[1],
                                                                                      ext=ext))): roi
            for roi in zip(xmin, ymin, xmax, ymax)}

        for future in futures.as_completed(future_to_patch):
            future.result()
    return xmin, ymin, xmax, ymax


def imread_and_mean_and_std(image_filepath):
    img = imread_full(image_filepath)
    img = pil_to_nparray(img).astype(np.float32)
    return np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))


def concurrent_imread_and_mean(image_filelist, sample_size):
    means = np.zeros((len(image_filelist), 3), dtype=np.float32)
    stds = np.zeros((len(image_filelist), 3), dtype=np.float32)
    with futures.ThreadPoolExecutor(max_workers=sample_size) as executor:
        future_to_stats = {
            executor.submit(imread_and_mean_and_std,
                            image_filepath): image_filepath
            for image_filepath in image_filelist}

    for idx, future in enumerate(futures.as_completed(future_to_stats)):
        mean, std = future.result()
        means[idx, :] = mean
        stds[idx, :] = std
    return means.mean(axis=0), stds.mean(axis=0)
