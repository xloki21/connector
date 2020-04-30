import concurrent.futures as futures
import io
import json
import os
from collections import OrderedDict
from typing import Any

import bs4
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection


def imwrite(array, filename):
    im = Image.fromarray(array)
    im.save(filename)


def imread(filename):
    im = Image.open(filename)
    return np.array(im)


def save_json_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def convert_pascal_voc_to_dict(filename):
    with open(filename, 'r') as file:
        parser = bs4.BeautifulSoup(markup=file,
                                   features='lxml')

        folder = parser.folder.text
        filename = parser.filename.text
        path = parser.path.text
        width = parser.size.width.text
        height = parser.size.height.text
        depth = parser.size.depth.text

        annotation = OrderedDict({'folder': folder,
                                  'filename': filename,
                                  'path': path,
                                  'image_shape': list(map(int, (height, width, depth))),
                                  'objects': []})

        objects = parser.find_all(name='object')

        for entry in objects:
            bndbox = [entry.bndbox.xmin.text,
                      entry.bndbox.ymin.text,
                      entry.bndbox.xmax.text,
                      entry.bndbox.ymax.text]

            bndbox = list(map(float, bndbox))
            xmin, ymin, xmax, ymax = bndbox
            xmin, ymin, xmax, ymax = xmin - 1, ymin - 1, xmax - 1, ymax - 1
            obj: OrderedDict[str, Any] = OrderedDict({
                'image': 'image',
                'x': [xmin, xmin, xmax, xmax],
                'y': [ymin, ymax, ymax, ymin],
                'label': entry.contents[1].text,
                'tag': 0})
            annotation['objects'].append(obj)
        return annotation


def save_axes(ax, dpi):
    ax.figure.canvas.draw()
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=dpi, transparent=True)
    ax.axis("off")
    buff.seek(0)
    img = (plt.imread(buff) * 255).astype(np.uint8)
    return img


def sliding_window(im_shape, patch_shape, dx, dy):
    # todo: <fixed> bug if im_shape < patch_shape (P2319.png DOTA)
    # return as is
    h, w = im_shape
    ph, pw = patch_shape

    if w >= pw:
        i = np.arange(w)[:-pw + 1:dx]
        if i[-1] != w - pw:
            i = np.hstack([i, w - pw])
    else:
        i = np.array([0], dtype=np.int64)
        pw = w

    if h >= ph:
        j = np.arange(h)[:-ph + 1:dy]
        if j[-1] != h - ph:
            j = np.hstack([j, h - ph])
    else:
        j = np.array([0], dtype=np.int64)
        ph = h

    x, y = np.meshgrid(i, j)

    xmin = x.reshape(-1, )
    ymin = y.reshape(-1, )
    xmax = xmin + pw
    ymax = ymin + ph

    return xmin, ymin, xmax, ymax


def save_image_roi(image, roi, filename):
    # todo: test
    height, width = image.shape[:2]
    xmin, ymin, xmax, ymax = roi

    # if image size lesser than roi size
    xmax = np.clip(xmax, a_min=0, a_max=width)
    ymax = np.clip(ymax, a_min=0, a_max=height)

    patch = image[ymin:ymax, xmin:xmax]
    if patch.ndim == 2:
        # convert from gray to RGB
        patch = np.expand_dims(patch, axis=2).repeat(3, axis=2)
    imwrite(patch, filename)
    return patch


def concurrent_crop_sliding_window_patches(image_filepath,
                                           patch_shape,
                                           pixel_shift,
                                           patch_root_dir,
                                           ext=None,
                                           max_workers=None):
    image = imread(image_filepath)
    image_filename = os.path.basename(image_filepath)
    image_basename = os.path.splitext(image_filename)[0]

    xmin, ymin, xmax, ymax = sliding_window(im_shape=image.shape[:2],
                                            patch_shape=patch_shape,
                                            dx=pixel_shift[1],
                                            dy=pixel_shift[0])
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


def draw_scoreboxes(image, bboxes,
                    labels,
                    scores=None,
                    fill=False,
                    dpi=100,
                    cmap='hsv',
                    score_as_bar=True):
    patch_collection = []
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / float(dpi), height / float(dpi)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)

    cmap = ScalarMappable(cmap=cmap).get_cmap()
    n_classes = len(np.unique(labels))

    if scores is not None:
        scores = np.clip(scores, 0, 1)

    colormap_arr = cmap(np.linspace(0, 255, n_classes).astype(int))
    colors = {label: colormap_arr[idx] for idx, label in enumerate(np.unique(labels))}
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        xmin, ymin, xmax, ymax = bbox

        if fill:
            p_roi = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, color=colors[label], fill=fill,
                                      alpha=0.3)
            patch_collection.append(p_roi)

        p_border = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, color=colors[label], fill=False,
                                     linewidth=2)
        patch_collection.append(p_border)

        height = ymax - ymin + 1
        width = xmax - xmin + 1

        if scores is not None:
            score = scores[i]
            if score_as_bar:
                p_bar_width = int(np.clip(width * 0.1, a_min=6, a_max=10))

                p_bar_border = patches.Rectangle((xmax + 5,
                                                  ymin + (1 - score) * height),
                                                 p_bar_width,
                                                 scores[i] * height - 1,
                                                 color=(1 - score, score, 0),
                                                 fill=False, linewidth=2)

                patch_collection.append(p_bar_border)

                p_bar_roi = patches.Rectangle((xmax + 5, ymin + (1 - score) * height),
                                              p_bar_width, score * height - 1,
                                              color=(1 - score, score, 0),
                                              fill=True)

                patch_collection.append(p_bar_roi)

                p_bar_border_split = patches.Rectangle((xmax + 5, ymin + (1 - score) * height),
                                                       p_bar_width,
                                                       score * height - 1,
                                                       color='black',
                                                       fill=False, linewidth=1)
                patch_collection.append(p_bar_border_split)
            else:
                ax.text(xmin + 3, ymin + 15, "{score:.02f}".format(score=score), color='black')
        ax.text(xmin + 3, ymax - 3, label, color='black')

    p = PatchCollection(patch_collection, match_original=True)
    ax.add_collection(p)
    result = save_axes(ax, dpi=dpi)
    plt.close()
    return result


def imread_and_mean_and_std(image_filepath):
    img = imread(image_filepath).astype(np.float32) / 255.0
    return np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))


def concurrent_imread_and_mean(image_filelist, max_workers):
    means = np.zeros((len(image_filelist), 3), dtype=np.float32)
    stds = np.zeros((len(image_filelist), 3), dtype=np.float32)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stats = {
            executor.submit(imread_and_mean_and_std,
                            image_filepath): image_filepath
            for image_filepath in image_filelist}

    for idx, future in enumerate(futures.as_completed(future_to_stats)):
        mean, std = future.result()
        means[idx, :] = mean
        stds[idx, :] = std
    return means.mean(axis=0), stds.mean(axis=0)
