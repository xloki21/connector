import io
import numpy as np
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


def draw_scoreboxes(image,
                    bboxes,
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
