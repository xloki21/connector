import os
import pandas as pd

from connector.generic import DataFrameCommonOpsMixin
from connector.tools.imaging import imread_full, pil_to_nparray
from connector.tools.visualization import draw_rect_with_attributes_and_landmarks
import matplotlib.pyplot as plt


class CelebADatasetConnector(DataFrameCommonOpsMixin):
    def __init__(self, dataframe=None):
        super().__init__(dataframe=dataframe)
        self.name = 'CelebA'

    def __str__(self):
        labels_info = "Dataset with {n} images(s)".format(n=len(self.images))
        images_info = "Unique persons: {n}".format(n=len(self.persons))

        return "\n".join([labels_info, images_info])

    @property
    def attributes(self):
        return self.df.columns[len(self.df.columns) - 40:].tolist()

    @property
    def images(self):
        return self.df.image_id.unique()

    @property
    def persons(self):
        return self.df.person_id.unique()

    @classmethod
    def connect(cls, folder, aligned=False):
        # identity
        an_identity = os.path.join(folder, 'Anno', 'identity_CelebA.txt')
        df_identity = pd.read_csv(an_identity, delimiter=' ', header=None, names=['person_id'])
        df_identity.index.name = 'image_id'

        # bboxes
        an_list_bbx = os.path.join(folder, 'Anno', 'list_bbox_celeba.txt')
        df_list_bbx = pd.read_csv(an_list_bbx, header=1, delimiter=r'\s+',
                                  names=['xmin', 'ymin', 'width', 'height'])
        df_list_bbx.index.name = 'image_id'

        # %%
        if aligned:  # no bboxes available
            an_list_lmk = os.path.join(folder, 'Anno', 'list_landmarks_align_celeba.txt')
            df_list_bbx[['xmin', 'ymin', 'width', 'height']] = None
        else:
            an_list_lmk = os.path.join(folder, 'Anno', 'list_landmarks_celeba.txt')

        # landmarks
        df_list_lmk = pd.read_csv(an_list_lmk, header=1, delimiter=r'\s+', names=['lex', 'ley',
                                                                                  'rex', 'rey',
                                                                                  'nx', 'ny',
                                                                                  'lmx', 'lmy',
                                                                                  'rmx', 'rmy'])
        df_list_lmk.index.name = 'image_id'

        # todo: evaluate "tight bbox" from landmarks
        # attributes
        an_list_att = os.path.join(folder, 'Anno', 'list_attr_celeba.txt')
        df_list_att = pd.read_csv(an_list_att, header=1, delimiter=r'\s+')
        df_list_att.index.names = ['image_id']
        df_list_att = df_list_att > 0

        _df = df_identity.join(df_list_bbx).join(df_list_lmk).join(df_list_att)
        _df.reset_index(inplace=True)

        # save full image path
        subdir = 'img_align_celeba' if aligned else 'img_celeba'
        _df['image_id'] = _df.image_id.apply(lambda x: os.path.join(folder, subdir, x))
        return cls(dataframe=_df)

    def select_person(self, idx: list):
        selected = self.df.person_id.isin(idx)
        return self.init(dataframe=self._df.loc[selected])

    def select_image(self, idx: list):
        selected = self.df.image_id.isin(idx)
        return self.init(dataframe=self.df.loc[selected])

    def select_attributes(self, attributes: list, all_true=True):
        selected_attrs = self.df[attributes]
        idx = selected_attrs.all(axis=1) if all_true else selected_attrs.any(axis=1)
        df = self.df.loc[idx]
        return self.init(dataframe=df)

    def show(self, show_attributes=False):
        for index, entry in self.df.iterrows():
            img = pil_to_nparray(imread_full(entry.image_id))
            active_attributes = [attr for attr in self.attributes if entry[attr]]
            xmin, ymin, width, height = entry.xmin, entry.ymin, entry.width, entry.height
            print(entry.person_id)
            landmarks = entry[['lex', 'ley', 'rex', 'rey',  # eyes
                               'nx', 'ny',  # nose
                               'lmx', 'lmy', 'rmx', 'rmy']]  # mouth
            result = draw_rect_with_attributes_and_landmarks(image=img,
                                                             rect=(xmin, ymin, width, height),
                                                             color='r',
                                                             attributes=active_attributes if show_attributes else None,
                                                             landmarks=landmarks.tolist(),
                                                             fill=True)
            print(entry)
            plt.imshow(result)
            plt.show()
