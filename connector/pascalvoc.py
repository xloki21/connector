import glob
import os.path as path

import pandas as pd
import tqdm

from connector.generic import LocalisationDatasetConnector
from connector.tools.xml import convert_pascal_voc_to_dict


class PASCALVOCDatasetConnector(LocalisationDatasetConnector):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.name = 'PASCALVOC'

    @classmethod
    def init(cls, dataframe=None):
        return cls(dataframe=dataframe)

    @classmethod
    def connect(cls, imagedir, labeldir, rebuild_index=False):
        """
        Создать коннектор для датасета.
        :param imagedir: Директория, в которой хранятся изображения
        :param labeldir: Директория, в которой хранится файл с аннотированными данными.

        :return: Объект-коннектор для формирования подвыборок данных.
        """
        pandas_sharded_dataframe = glob.glob(labeldir + '/*.shard', recursive=True)
        if len(pandas_sharded_dataframe) == 0 or rebuild_index:
            print('log: Файл с индексом в формате Pandas не найден. Индекс будет перестроен.')
            imfile_list = []
            for mask in ["xml"]:
                imfile_list += glob.glob(labeldir + '/*.' + mask, recursive=True)

            df = None
            for file in tqdm.tqdm(imfile_list, desc='loading annotation original data'):

                annotation = convert_pascal_voc_to_dict(file)
                data = pd.DataFrame(data=annotation['objects'],
                                    columns=['image', 'x', 'y', 'label', 'tag'])
                data["image"] = path.join(imagedir, annotation['filename'])
                if df is None:
                    df = data.copy()
                else:
                    df = df.append(data.copy())
            conn = cls(dataframe=df)
            conn.save(labeldir, shard_size=5000)
        else:
            conn = cls.load(labeldir)
        return conn

    def convert_to_original_format(self, connector, labeldir):
        pass
