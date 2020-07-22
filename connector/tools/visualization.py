import io
import numpy as np
from PIL.ImageDraw import ImageDraw

from connector.tools.imaging import imread_lazy
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_axes(ax, dpi):
    ax.figure.canvas.draw()
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=dpi, transparent=True)
    ax.axis("off")
    buff.seek(0)
    img = (plt.imread(buff) * 255).astype(np.uint8)
    return img


def draw_rect_with_attributes_and_landmarks(image,
                                            rect=None,
                                            color=None,
                                            attributes=None,
                                            landmarks=None,
                                            fill=False,
                                            dpi=100):
    # todo: ==> PIL
    patch_collection = []
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / float(dpi), height / float(dpi)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)

    if rect and all([coord is not None for coord in rect]):
        xmin, ymin, width, height = rect
        # rect
        if fill:
            p_roi = patches.Rectangle((xmin, ymin), width, height, color=color, fill=fill,
                                      alpha=0.3)
            patch_collection.append(p_roi)

        p_border = patches.Rectangle((xmin, ymin), width, height, color=color, fill=False,
                                     linewidth=2)
        patch_collection.append(p_border)

    if attributes:
        att_text = '\n'.join(attributes)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # attributes
        ax.text(0.05, 0.95, att_text, transform=ax.transAxes, fontsize=25,
                verticalalignment='top', bbox=props)

    if landmarks:
        ax.plot(landmarks[0::2], landmarks[1::2], 'go')

    p = PatchCollection(patch_collection, match_original=True)
    ax.add_collection(p)
    result = save_axes(ax, dpi=dpi)
    plt.close()
    return result


def create_custom_colordict(labels, cmap, alpha):
    cmap = ScalarMappable(cmap=cmap).get_cmap()
    n_classes = len(labels)
    colormap_arr = (cmap(range(0, 255, 255 // n_classes)) * 255).astype(int)
    colormap_arr[:, -1] = alpha
    colors = {label: (colormap_arr[idx]) for idx, label in enumerate(np.unique(labels))}
    return colors


def draw_rect_items(image_filename,
                    items,
                    labels,
                    color_dict,
                    scores=None,
                    filled=True,
                    scores_as_bar=False):
    img = imread_lazy(image_filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    border_width = 3
    painter = ImageDraw(im=img, mode='RGBA')

    if len(labels) > 0:
        if scores is not None:
            scores = np.clip(scores, 0, 1)
        else:
            scores = [None] * len(labels)

    for (index, item), label, score in zip(items.iterrows(), labels, scores):
        xmin, ymin, xmax, ymax = item
        width = xmax - xmin
        height = ymax - ymin
        painter.rectangle(xy=item,
                          outline=(*tuple(color_dict[label])[:3], 255),
                          fill=tuple(color_dict[label]) if filled else None,
                          width=border_width)
        t_width, t_height = painter.textsize(text="{label}".format(label=label))
        if (t_width < width) and (t_height < height):
            tcx, tcy = (xmin + xmax) // 2, (ymin + ymax) // 2,
            painter.text(xy=(tcx - t_width // 2, tcy - t_height // 2),
                         text="{label}".format(label=label),
                         fill=(0, 0, 0))

        if score:
            if scores_as_bar:
                p_bar_width = int(np.clip(width * 0.1, a_min=6, a_max=10))
                bxmin = xmax + 5
                bymin = int(ymin + (1 - score) * height)

                painter.rectangle(xy=(bxmin,
                                      bymin,
                                      bxmin + p_bar_width,
                                      int(bymin + score * height)),
                                  outline=None,
                                  fill=(int(255 * (1 - score)), int(255 * score), 0))
            t_width, t_height - painter.textsize(text="{score:0.2f".format(score=score))
            painter.text(xy=(xmin + border_width + 1, ymax - border_width - t_height + 2),
                         text="{score:0.2f".format(score=score),
                         fill=(0, 0, 0))
    return img


def draw_poly_items(image_filename,
                    items,
                    labels,
                    color_dict,
                    scores=None,
                    filled=True):
    img = imread_lazy(image_filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    painter = ImageDraw(im=img, mode='RGBA')

    if len(items) > 0:
        if scores is not None:
            scores = np.clip(scores, 0, 1)
        else:
            scores = [None] * len(items)
    for i, (item, label, score) in enumerate(zip(items, labels, scores)):
        painter.polygon(xy=item,
                        fill=tuple(color_dict[label]) if filled else None,
                        outline=(*tuple(color_dict[label])[:3], 255))

        t_width, t_height = painter.textsize(text="{label}".format(label=label))
        width, height = np.max(item, axis=0) - np.min(item, axis=0)
        if (t_width < width) and (t_height < height):
            tcx, tcy = np.sum(item, axis=0) / len(item)
            painter.text(xy=(tcx - t_width // 2, tcy - t_height // 2),
                         text="{label}".format(label=label),
                         fill=(0, 0, 0))

        x, y = item[0]
        painter.rectangle(xy=(x - 2, y - 2,
                              x + 2, y + 2), fill=(255, 255, 0, 255))
    return img
