from collections import OrderedDict
from typing import Any

import bs4


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