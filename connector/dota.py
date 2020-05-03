import os
import glob
from collections import OrderedDict

import tqdm
import pandas as pd

from connector.generic import LocalisationDatasetConnector
from connector.tools.imaging import pil_to_nparray, imread_full


class DOTADatasetConnector(LocalisationDatasetConnector):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.name = 'DOTA'

    @classmethod
    def connect(cls, imagedir, labeldir, rebuild_index=False):
        """
        Создать коннектор для датасета.
        :param imagedir: Директория, в которой хранятся изображения
        :param labeldir: Директория, в которой хранится файл с аннотированными данными.
        :param rebuild_index: Принудительная перестройка индекса.
        :return: Объект-коннектор для формирования подвыборок данных.
        """
        # todo: Обработать случай, когда размер шарда меньше разметки одного изображения

        pandas_sharded_dataframe = glob.glob(labeldir + '/*.shard', recursive=True)

        if len(pandas_sharded_dataframe) == 0 or rebuild_index:
            pandas_sharded_dataframe = os.path.join(labeldir, 'dataframe.{index:06}.shard')
            print('log: Набор Файлов с индексом в формате Pandas не найден. Индекс будет перестроен.')
            imfile_list = []
            for mask in ["txt"]:
                imfile_list += glob.glob(labeldir + '/*.' + mask, recursive=True)

            shard_size = 30000
            shard_index = 0
            empty_data = [OrderedDict({'image': 'path_to_image',
                                       'x': [0, 0, 0, 0],
                                       'y': [0, 0, 0, 0],
                                       'label': 'none',
                                       'tag': 0})] * shard_size

            shard = pd.DataFrame(data=empty_data)
            last_index = 0
            shard_completed = False
            for file in tqdm.tqdm(imfile_list):
                name = os.path.splitext(os.path.split(file)[1])[0]

                data = pd.read_csv(file, delimiter=' ',
                                   header=1,  # Этот ключ необходим. Zero-based индекс хедера для данных.
                                   names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'label', 'tag'])
                data["image"] = os.path.join(imagedir, name + '.png')
                data["x"] = data[['x1', 'x2', 'x3', 'x4']].values.tolist()
                data["y"] = data[['y1', 'y2', 'y3', 'y4']].values.tolist()
                data.drop(columns=['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4'], inplace=True)
                n_entries = len(data)
                if (last_index + n_entries) >= shard_size:

                    shard.iloc[:last_index].to_csv(pandas_sharded_dataframe.format(index=shard_index),
                                                   index=False,
                                                   sep=' ')
                    shard_index = shard_index + 1
                    shard.iloc[:n_entries] = data[['image', 'x', 'y', 'label', 'tag']].values
                    last_index = n_entries
                    shard_completed = True
                else:
                    shard_completed = False
                    shard.iloc[last_index:last_index + n_entries] = data[['image', 'x', 'y', 'label', 'tag']].values
                    last_index += n_entries

            if not shard_completed:
                shard.iloc[:last_index].to_csv(pandas_sharded_dataframe.format(index=shard_index), index=False, sep=' ')

        return cls.load(labeldir)

    @classmethod
    def convert_to_original_format(cls, connector, labeldir):
        basename = os.path.basename(connector.images[0])
        basename_wo_ext = os.path.splitext(basename)[0]
        label_file = os.path.join(labeldir, basename_wo_ext + '.txt')  # DOTA feature
        with open(label_file, "w") as a_f:
            a_f.write("#\n")
            a_f.write("#\n")

        dota_df = connector.df.copy()
        dota_df['x1'] = dota_df.x.apply(lambda x: x[0])
        dota_df['x2'] = dota_df.x.apply(lambda x: x[1])
        dota_df['x3'] = dota_df.x.apply(lambda x: x[2])
        dota_df['x4'] = dota_df.x.apply(lambda x: x[3])

        dota_df['y1'] = dota_df.y.apply(lambda x: x[0])
        dota_df['y2'] = dota_df.y.apply(lambda x: x[1])
        dota_df['y3'] = dota_df.y.apply(lambda x: x[2])
        dota_df['y4'] = dota_df.y.apply(lambda x: x[3])

        dota_df.to_csv(label_file,
                       mode="a", sep=' ',
                       index=False,
                       header=None,
                       columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'label', 'tag'])

    def collate_fn(self, idx):
        filenames = self.images[idx]
        imgs = []
        annotations = []
        for filename in filenames:
            img = pil_to_nparray(imread_full(filename))

            entry = self.select_images(image_idx=filenames).df
            xmin = entry.x.apply(lambda x: min(x))
            ymin = entry.y.apply(lambda x: min(x))
            width = (entry.x.apply(lambda x: max(x)) - xmin + 1)
            height = (entry.y.apply(lambda x: max(x)) - ymin + 1)
            label = entry.label.tolist()

            annotation = list(zip(xmin.tolist(), ymin.tolist(),
                                  width.tolist(), height.tolist(), label))

            imgs.append(img)
            annotations.append(annotation)

        return imgs, annotations
