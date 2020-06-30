import glob
import os.path as path

import pandas as pd
import tqdm

from connector.generic import LocalisationDatasetConnector
from connector.tools.visualization import create_custom_colordict, draw_rect_items
from connector.tools.xml import convert_pascal_voc_to_dict


class PASCALVOCDatasetConnector(LocalisationDatasetConnector):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.name = 'PASCALVOC'
        self.default_label_set = self.labels

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

    def collater_fn(self, data):
        pass

    def draw_image_annotation(self, image_file, color_dict=None):
        # original data annotated as hbboxes.
        subset = self.select_images(image_idx=image_file)
        if color_dict is None:
            color_dict = create_custom_colordict(self.default_label_set, cmap='hsv', alpha=120)

        annotated_image = draw_rect_items(image_filename=image_file,
                                          items=subset.hbbox,
                                          labels=subset.df.label,
                                          color_dict=color_dict,
                                          scores=None,
                                          filled=True)
        return annotated_image
