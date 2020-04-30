import os
import pandas as pd
import numpy as np

from multiprocessing.pool import ThreadPool
from connector.tools.imaging import imread_full, pil_to_nparray
from connector.generic import ChangeDetectionDatasetConnector


class CDDatasetConnector(ChangeDetectionDatasetConnector):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self._name = 'cd2014'
        self._mean_image0 = np.array([118, 117, 110])
        self._mean_image1 = np.array([121, 119, 112])
        self._std_image0 = np.array([48, 47, 48])
        self._std_image1 = np.array([51, 50, 51])
        self._advanced_label = False
        self._shuffle_pair = False

    @classmethod
    def init(cls, dataframe=None):
        return cls(dataframe=dataframe)

    @classmethod
    def connect(cls, root_dir, filename):
        _df = pd.read_csv(filename, delimiter=' ',
                          names=['image0', 'image1', 'mask'])

        _df['cluster'] = _df.image0.apply(lambda x: x.split("/")[0])
        _df['group'] = _df.image0.apply(lambda x: x.split("/")[1])

        _df[["image0", "image1", "mask"]] = os.path.abspath(root_dir) + "/" + _df[["image0", "image1", "mask"]]

        return cls(dataframe=_df)

    def advanced(self, advanced_label=True, shuffle_pair=True):
        self._advanced_label = advanced_label
        self._shuffle_pair = shuffle_pair

    def collate_fn(self, idx):
        imgs1 = []
        imgs2 = []
        imgs3 = []
        data = self.df.iloc[idx]
        for i, pair in data.iterrows():
            img1 = pil_to_nparray(imread_full(pair.image0))
            img2 = pil_to_nparray(imread_full(pair.image1))
            img3 = pil_to_nparray(imread_full(pair["mask"]))

            imgs1.append(img1)
            imgs2.append(img2)
            imgs3.append(img3)
        return [imgs1, imgs2, imgs3]
