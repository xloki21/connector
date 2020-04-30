import os
import os.path as path
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tqdm


class OneraDatasetConnector:
    def __init__(self, data=None):
        self._data = data
        self._root_dir = None

    @classmethod
    def connect(cls, cache_file=None, root_dir='//f125.sils.local/doc/projects/ML/data/Onera'):
        cls._root_dir = root_dir

        if cache_file is None:
            cache_file = path.join(root_dir, 'onera.cache')

        if path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                print('log: файл с кешированными данными обнаружен и будет загружен.')
                data = pickle.load(file)
        else:
            print('log: После подключения датасета в корневой директории будет создан файл с кешированными данными.')
            content = os.listdir(root_dir)
            data = {}
            channels = ('B01', 'B02', 'B03', 'B04',
                        'B05', 'B06', 'B07', 'B08',
                        'B09', 'B10', 'B11', 'B12',
                        'B8A', 'visible')
            for entry in tqdm.tqdm(content):
                abs_entry_path = path.join(root_dir, entry)
                if os.path.isdir(abs_entry_path):
                    data[entry] = {'before': {}, 'after': {}, 'label': None}
                    img_1_data_path = path.join(abs_entry_path, 'imgs_1_rect')
                    img_2_data_path = path.join(abs_entry_path, 'imgs_2_rect')
                    cm_data_path = path.join(abs_entry_path, 'cm')
                    for channel in channels:
                        channel_img_1_filename = path.join(img_1_data_path, channel + '.tif')
                        channel_img_2_filename = path.join(img_2_data_path, channel + '.tif')

                        if os.path.exists(channel_img_1_filename):
                            data[entry]['before'][channel] = plt.imread(channel_img_1_filename)
                        elif channel == 'visible':
                            img_1_visible_path = path.join(abs_entry_path, 'pair/img1.png')
                            data[entry]['before'][channel] = plt.imread(img_1_visible_path)

                        if os.path.exists(channel_img_2_filename):
                            data[entry]['after'][channel] = plt.imread(channel_img_2_filename)
                        elif channel == 'visible':
                            img_2_visible_path = path.join(abs_entry_path, 'pair/img2.png')
                            data[entry]['after'][channel] = plt.imread(img_2_visible_path)
                    if path.exists(cm_data_path):
                        label_tif_file = glob(path.join(cm_data_path, '*.tif'))

                        data[entry]['label'] = plt.imread(label_tif_file[0])

            with open(path.join(root_dir, 'onera.cache'), 'wb') as file:
                pickle.dump(data, file=file)

        return cls(data=data)

    def __getattr__(self, item):
        if item in self.images:
            return self._data[item].copy()
        else:
            raise AttributeError("Данные не найдены")

    @property
    def train_set(self):
        return dict([(key, self._data[key]) for key in self._data.keys() if self._data[key]['label'] is not None])

    @property
    def test_set(self):
        return dict([(key, self._data[key]) for key in self._data.keys() if self._data[key]['label'] is None])

    @property
    def images(self):
        return list(self._data.keys())

    @property
    def raw_data(self):
        return self._data

    def show_multisensor_data(self, name, state, cmap=None):

        def imadjust(image):
            image -= image.min()
            return image / np.max(image)

        if name in self._data.keys():
            patches = [imadjust(self._data[name][state][key]) for key in self._data[name][state].keys()]
            row1 = np.hstack(patches[0:4])
            row2 = np.hstack(patches[4:8])
            row3 = np.hstack(patches[8:12])

            rgb_split = np.squeeze(np.hstack(np.split(patches[-1], 3, axis=2)))
            row4 = np.hstack([patches[12], rgb_split])
            row5 = np.hstack([patches[13]] * 4)
            panno = np.vstack([row1, row2, row3, row4])
            rgb_panno = np.stack([panno] * 3, axis=2)
            plt.imshow(np.vstack([rgb_panno, row5]), cmap=cmap)

    def compare_multisensor_data(self, name, cmap=None):
        plt.figure()
        plt.axis('off')
        self.show_multisensor_data(name, state='before', cmap=cmap)

        plt.figure()
        plt.axis('off')
        self.show_multisensor_data(name, state='after', cmap=cmap)
        plt.show()


conn = OneraDatasetConnector().connect(cache_file='d:/onera.cache')
print(conn.train_set)
k = 9
