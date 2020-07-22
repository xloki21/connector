import os
import pandas as pd
import numpy as np
from skimage.transform import resize
from PIL import Image

from multiprocessing.pool import ThreadPool
from connector.tools.imaging import imread_full, pil_to_nparray
from connector.generic import ChangeDetectionDatasetConnector


class CDDatasetConnector(ChangeDetectionDatasetConnector):

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self._name = 'cd2014'
        # По умолчанию лучше сделать эти коэфф-ты не действующими, т.к. некоторые сети в архитектуре
        # явно используют BatchNormalization-операцию в начале графа
        self._mean_image0 = None
        self._mean_image1 = None
        self._std_image0 = None
        self._std_image1 = None
        self._advanced_label = False
        self._shuffle_pair = False
        self._thread = 10
        self._worker_pool = ThreadPool(self._thread)
        self.out_shape = (512, 512, 3)

    @classmethod
    def init(cls, dataframe=None):
        return cls(dataframe=dataframe)

    @classmethod
    def connect(cls, root_dir, filename):
        _df = pd.read_csv(filename, delimiter=' ',
                          names=['image0', 'image1', 'mask'])

        def split_path(path):
            result = []
            while True:
                tail, head = os.path.split(path)
                if head == '':
                    return result
                result.append(head)
                path = tail

        _df['cluster'] = _df.image0.apply(lambda x: split_path(x)[-1])
        _df['group'] = _df.image0.apply(lambda x: split_path(x)[-2])

        # _df[["image0", "image1", "mask"]] = os.path.abspath(root_dir) + "/" + _df[["image0", "image1", "mask"]]
        _df["image0"] = _df["image0"].apply(lambda x: os.path.join(os.path.abspath(root_dir), x))
        _df["image1"] = _df["image1"].apply(lambda x: os.path.join(os.path.abspath(root_dir), x))
        _df["mask"] = _df["mask"].apply(lambda x: os.path.join(os.path.abspath(root_dir), x))
        return cls(dataframe=_df)

    def advanced(self, advanced_label=True, shuffle_pair=True):
        self._advanced_label = advanced_label
        self._shuffle_pair = shuffle_pair

    def collater_fn(self, idx):
        batch_size = len(idx)

        # Массив оригинальных данных исключительно или в формате серых (C=1) или цветных изображений (C=3)
        assert self.out_shape[2] in (1, 3)

        if self._mean_image0 is None:
            self._mean_image0 = np.zeros(self.out_shape[2])

        if self._std_image0 is None:
            self._std_image0 = np.ones(self.out_shape[2])

        if self._mean_image1 is None:
            self._mean_image1 = np.zeros(self.out_shape[2])

        if self._std_image1 is None:
            self._std_image1 = np.ones(self.out_shape[2])

        assert len(self._mean_image0) == self.out_shape[2]
        assert len(self._mean_image1) == self.out_shape[2]
        assert len(self._std_image0) == self.out_shape[2]
        assert len(self._std_image1) == self.out_shape[2]

        # Оригинальные данные
        t0_images_orig = np.zeros((batch_size, *self.out_shape), dtype=np.float32)
        t1_images_orig = np.zeros((batch_size, *self.out_shape), dtype=np.float32)

        # Данные с предобработкой
        t0_images = np.zeros((batch_size, *self.out_shape), dtype=np.float32)
        t1_images = np.zeros((batch_size, *self.out_shape), dtype=np.float32)

        labels = np.zeros((batch_size, *self.out_shape[:2], 4 if self._advanced_label else 2), dtype=np.float32)

        data = self.df.iloc[idx]

        for i, (_, pair) in enumerate(data.iterrows()):
            # Читаем оригинальные данные: Может прочитать uint8[HxW],
            # принудительно конвертируем в float32 [0, 1]
            # input0 = pil_to_nparray(imread_full(pair.image0)).astype(np.float32) / 255.0
            input0 = pil_to_nparray(imread_full(pair.image0).resize((self.out_shape[0], self.out_shape[1]))).astype(np.float32) / 255.0
            input1 = pil_to_nparray(imread_full(pair.image1).resize((self.out_shape[0], self.out_shape[1]))).astype(np.float32) / 255.0
            gt = pil_to_nparray(imread_full(pair["mask"]).resize((self.out_shape[0], self.out_shape[1]))).astype(bool)

            if input0.ndim == 2 and self.out_shape[2] == 3:
                # Конвертация серого изображения в цветное
                input0 = np.stack((input0, input0, input0), axis=2)

            if input1.ndim == 2 and self.out_shape[2] == 3:
                # Конвертация серого изображения в цветное
                input1 = np.stack((input1, input1, input1), axis=2)

            # Читаем и конвертируем маски
            nochange = (~gt).astype(np.float32)
            change = gt.astype(np.float32)
            label = np.stack([change, nochange], axis=-1).astype(np.float32)

            # Сохраняем оригинальные ненормированные данные
            # atleast3d([512x512]).shape == (512, 512, 1)
            t0_images_orig[i] = np.atleast_3d(input0)
            t1_images_orig[i] = np.atleast_3d(input1)

            t0_images[i] = (t0_images_orig[i] - self._mean_image0) / self._std_image0
            t1_images[i] = (t1_images_orig[i] - self._mean_image1) / self._std_image1

            labels[i] = label
            i += 1
        return t0_images, t1_images, labels, t0_images_orig, t1_images_orig
