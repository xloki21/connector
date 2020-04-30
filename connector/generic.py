import os
import glob
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import abstractmethod, ABC
from multiprocessing.pool import ThreadPool

from connector.tools.json import save_json_data
from connector.tools.visualization import draw_scoreboxes
from connector.tools.multiprocessing import creating_task_for_pool
from connector.tools.imaging import imread_full, imread_lazy, crop_patches
from connector.tools.imaging import imread_and_mean_and_std, pil_to_nparray


class BaseDatasetConnector(ABC):
    def __init__(self):
        self._thread = os.cpu_count() * 2
        self._prefetch = self._thread * 2
        self._worker_pool = ThreadPool(self._thread)
        # TODO: Надо написать еще какие-нибудь методы
        # self._worker_pool.close()


class LocalisationDatasetConnector(BaseDatasetConnector):
    def __init__(self, dataframe):
        super().__init__()
        self._df = dataframe

    def __str__(self):
        labels_info = "Dataset with {n} label(s): {labels}".format(n=len(self.labels), labels=self.labels)
        images_info = "Number of images: {n}".format(n=len(self.images))
        return "\n".join([labels_info, images_info])

    def __add__(self, connector):
        _df = self._df.append(connector.df.copy())

        return self.init(dataframe=_df)

    def __len__(self):
        return len(self._df)

    @property
    def df(self):
        return self._df

    @property
    def size(self):
        return len(self._df)

    @property
    def images(self):
        return self._df.image.unique()

    @property
    def labels(self):
        return self._df.label.unique()

    def create_coco_format_annotation(self, jsonfile=None, labeltoid: dict = None):
        annotations = []

        if labeltoid is None:
            labeltoid = dict((value, key + 1) for key, value in enumerate(self.labels))

        categories = [dict((("id", labeltoid[entry]), ("name", entry))) for entry in labeltoid]
        imagetoid = dict((image, idx) for idx, image in enumerate(self.images))
        bbox_id = 0
        image = {"file_name": "",
                 "height": 0,
                 "width": 0,
                 "id": 0}
        images = [image] * len(self.images)
        for i, filename in tqdm.tqdm(enumerate(self.images), desc='loading image information', total=len(self.images)):
            imshape = imread_lazy(filename).size
            image["file_name"] = filename
            image["height"] = imshape[0]
            image["width"] = imshape[1]
            image["id"] = imagetoid[filename]
            images[i] = image.copy()

        for i, entry in tqdm.tqdm(self.df.iterrows(), total=self.size, desc='converting annotation to COCO format'):
            xmin = min(entry.x)
            ymin = min(entry.y)
            width = max(entry.x) - xmin + 1
            height = max(entry.y) - ymin + 1
            annotation = {
                "id": bbox_id,
                "image_id": imagetoid[entry.image],
                "bbox": [xmin, ymin, width, height],
                "iscrowd": 0,
                "category_id": labeltoid[entry.label]
            }
            annotations.append(annotation)
            bbox_id += 1

        result = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        if jsonfile:
            print('dumping data to {filename}'.format(filename=os.path.abspath(jsonfile)))
            save_json_data(result, filename=jsonfile)

        return result

    @classmethod
    @abstractmethod
    def init(cls, dataframe=None):
        return cls(dataframe=dataframe)

    @classmethod
    @abstractmethod
    def connect(cls, image_dir, label_dir):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def collate_fn(self, data):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def convert_to_original_format(self, connector, labeldir):
        raise NotImplementedError('Not implemented')

    def select_images(self, image_idx):
        if isinstance(image_idx, list):
            idx = self.df.image.isin(image_idx)
        elif isinstance(image_idx, np.ndarray):
            im_list = image_idx.tolist()
            idx = self.df.image.isin(im_list)
        elif isinstance(image_idx, str):
            idx = self.df.image.isin([image_idx])
        else:
            raise TypeError("image_idx: должен быть или str или list<str> или numpy.ndarray(dtype=str)")
        return self.init(dataframe=self.df.loc[idx])

    def select_objects(self, width=None, height=None):
        df = self.df.copy()  # ?

        if width:
            w = df.x.apply(lambda x: max(x)) - df.x.apply(lambda x: min(x))
            idx = w.between(*width)
            df = df.loc[idx]

        if height:
            h = df.y.apply(lambda x: max(x)) - df.y.apply(lambda x: min(x))
            idx = h.between(*height)
            df = df.loc[idx]

        return self.init(dataframe=df)

    def select_labels(self, labels):
        if isinstance(labels, list):
            idx = self.df.label.isin(labels)
        elif isinstance(labels, str):
            idx = self.df.label.isin([labels])
        else:
            raise TypeError("labels: должен быть или str или list<str>")

        return self.init(dataframe=self.df.loc[idx])

    def select_random_samples(self, nsamples):
        idx = self._df.index.values.copy()
        np.random.shuffle(idx)
        if nsamples < len(idx):
            idx = idx[:nsamples]
        return self.init(dataframe=self.df.iloc[idx])

    def select_roi(self, roi, thresh_area=0.5):
        x1, y1, x2, y2 = roi
        local_df = self.df.copy()

        xmin = local_df.x.apply(lambda x: min(x))
        ymin = local_df.y.apply(lambda x: min(x))
        xmax = local_df.x.apply(lambda x: max(x))
        ymax = local_df.y.apply(lambda x: max(x))

        area = (xmax - xmin) * (ymax - ymin)

        xnew = local_df.x.apply(lambda x: np.clip(x, a_min=x1, a_max=x2 - 1).tolist())
        ynew = local_df.y.apply(lambda x: np.clip(x, a_min=y1, a_max=y2 - 1).tolist())

        xmin_new = xnew.apply(lambda x: min(x))
        ymin_new = ynew.apply(lambda x: min(x))

        xmax_new = xnew.apply(lambda x: max(x))
        ymax_new = ynew.apply(lambda x: max(x))

        area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)
        temp = area_new / area
        idx = temp >= thresh_area

        local_df.loc[idx, 'x'] = xnew.loc[idx]
        local_df.loc[idx, 'y'] = ynew.loc[idx]

        return self.init(dataframe=local_df.loc[idx])

    def convert(self, roi_shape, imagedir, labeldir, rel_shift=1.0, padding=True):
        if not os.path.exists(labeldir):
            os.makedirs(labeldir, exist_ok=True)

        if not os.path.exists(imagedir):
            os.makedirs(imagedir, exist_ok=True)

        images = self.images

        df = None
        dy = int(roi_shape[0] * rel_shift)
        dx = int(roi_shape[1] * rel_shift)
        for imgname in tqdm.tqdm(images, desc='Converting'):
            img_basename = os.path.basename(imgname)
            img_basename_wo_ext, ext = os.path.splitext(img_basename)

            xmin, ymin, xmax, ymax = crop_patches(image_filepath=imgname,
                                                  patch_shape=roi_shape,
                                                  dx=dx,
                                                  dy=dy,
                                                  patch_root_dir=imagedir,
                                                  ext=ext,
                                                  padding=padding,
                                                  max_workers=8)
            image_annotation = self.select_images(imgname)
            for roi in zip(xmin, ymin, xmax, ymax):
                roi_annotation = image_annotation.select_roi(roi=roi)
                if roi_annotation.size > 0:
                    roi_annotation.df.x = roi_annotation.df.x.apply(
                        lambda x: [x[i] - float(roi[0]) for i in range(len(x))])
                    roi_annotation.df.y = roi_annotation.df.y.apply(
                        lambda x: [x[i] - float(roi[1]) for i in range(len(x))])
                    patch_filename = "{basename}_{ymin:04}_{xmin:04}{ext}".format(basename=img_basename_wo_ext,
                                                                                  xmin=roi[0],
                                                                                  ymin=roi[1],
                                                                                  ext=ext)

                    roi_annotation.df['image'] = os.path.join(imagedir, patch_filename)

                    self.convert_to_original_format(connector=roi_annotation,
                                                    labeldir=labeldir)

                    if df is None:
                        df = roi_annotation.df.copy()
                    else:
                        df = df.append(roi_annotation.df.copy())
        conn = self.init(dataframe=df)
        conn.save(folder=labeldir, shard_size=10000)
        return conn

    def draw_image_annotation(self, image_file, cmap='autumn'):
        raw_img = imread_full(image_file)
        raw_img = pil_to_nparray(raw_img)
        # Получаем глобальный список меток для среза.
        image_info = self.select_images(image_idx=image_file)

        x = image_info.df.x
        y = image_info.df.y
        xmin = x.apply(lambda var: min(var))
        xmax = x.apply(lambda var: max(var))
        ymin = y.apply(lambda var: min(var))
        ymax = y.apply(lambda var: max(var))
        bboxes = zip(xmin, ymin, xmax, ymax)
        return draw_scoreboxes(raw_img,
                               bboxes=bboxes,
                               labels=image_info.df['label'].values,
                               scores=None,
                               fill=True,
                               cmap=cmap,
                               score_as_bar=False)

    def _calculate_statistics(self):
        index = pd.MultiIndex.from_product([self.labels, ['width', 'height']],
                                           names=['label', 'feature'])
        info = pd.DataFrame(index=index,
                            columns=['count', 'mean', 'min', '25%', '50%', '75%', 'max'])

        for label in self.df.label.unique():
            idx = self.df.label == label
            label_slice = self.df.loc[idx].copy()
            w = label_slice.x.apply(lambda x: max(x)) - label_slice.x.apply(lambda x: min(x))
            h = label_slice.y.apply(lambda x: max(x)) - label_slice.y.apply(lambda x: min(x))
            w_stats = w.describe().drop(['std']).astype(int)
            h_stats = h.describe().drop(['std']).astype(int)
            info.loc[label, 'width'] = w_stats
            info.loc[label, 'height'] = h_stats
        return info

    def describe(self, filename: str = None):
        info = self._calculate_statistics()

        if filename:
            info.to_csv(filename, sep='\t', float_format='%.2f')
        else:
            print(info)
            print('Выбрано: {total} объектов'.format(total=self.size))
        return info

    def calculate_stat_coeffs(self, n_bootsrap=10, sample_size=20, filename=None):
        # todo: only rgb-images supported. Need to generalize
        def worker_fn(filenames):
            means = np.zeros((len(filenames), 3), dtype=np.float32)
            stds = np.zeros((len(filenames), 3), dtype=np.float32)
            for idx, _filename in enumerate(filenames):
                _mean, _std = imread_and_mean_and_std(_filename)
                means[idx, :] = _mean
                stds[idx, :] = _std
            return means.mean(axis=0), stds.mean(axis=0)

        bootstrap_means = np.zeros(shape=(n_bootsrap, 3))
        bootstrap_stds = np.zeros(shape=(n_bootsrap, 3))

        stat_iter = creating_task_for_pool(sampler=np.random.choice(self.images, size=sample_size * n_bootsrap),
                                           batch_size=sample_size, worker_pool=self._worker_pool,
                                           worker_fn=worker_fn, prefetch=self._prefetch)

        for idx_stat, stat in tqdm.tqdm(enumerate(stat_iter),
                                        desc='Вычисление статистик',
                                        total=len(stat_iter)):
            mean, std = stat
            bootstrap_means[idx_stat, :] = mean
            bootstrap_stds[idx_stat, :] = std

        mv, sv = bootstrap_means.mean(axis=0), bootstrap_stds.mean(axis=0)
        if filename:
            with open(filename, 'w') as file:
                file.write("mean value: {value}\n".format(value=mv))
                file.write("std value: {value}\n".format(value=sv))
        return mv, sv

    def show(self):
        for image in self.images:
            print(image)
            result = self.draw_image_annotation(image_file=image)
            plt.imshow(result)
            plt.show()

    def save(self, folder, shard_size):

        os.makedirs(folder, exist_ok=True)
        n_entries = self.size

        if n_entries % shard_size == 0:
            n_shards = n_entries // shard_size
        else:
            n_shards = n_entries // shard_size + 1

        from_i, to_i = 0, 0
        for i in range(n_shards - 1):
            from_i = i * shard_size
            to_i = from_i + shard_size
            self.df.iloc[from_i:to_i].to_csv(os.path.join(folder,
                                                          "dataset.{index:06}.shard".format(index=i)),
                                             index=False,
                                             sep=' ')

        self.df.iloc[to_i:].to_csv(os.path.join(folder,
                                                "dataset.{index:06}.shard".format(index=n_shards - 1)),
                                   index=False,
                                   sep=' ')

    @classmethod
    def load(cls, folder):
        pandas_sharded_dataframe = glob.glob(folder + '/*.shard', recursive=True)
        _df = None
        for shard in tqdm.tqdm(pandas_sharded_dataframe, desc='Загрузка датафрейма'):
            _sdf = pd.read_csv(shard,
                               delimiter=' ',
                               header=0,  # Этот ключ необходим, чтобы указать, что 0-строка - это хедер.
                               names=['image', 'x', 'y', 'label', 'tag'])
            if _df is None:
                _df = _sdf.copy()
            else:
                _df = _df.append(_sdf.copy())

        def map_str_to_list(strlist):
            return list(map(float, strlist.strip('[]').split(', ')))

        _df.x = _df.x.apply(map_str_to_list)
        _df.y = _df.y.apply(map_str_to_list)

        return cls(dataframe=_df)

    def torch_interface(self):
        try:
            from connector.ctorch import TorchConnector
        except:
            raise ModuleNotFoundError('pytorch not found')
        return TorchConnector(dataframe=self.df, transforms_compose=None)


class ChangeDetectionDatasetConnector(BaseDatasetConnector):
    def __init__(self, dataframe):
        super().__init__()
        self._df = dataframe

    def __len__(self):
        return len(self._df)

    def __add__(self, connector):
        _df = self._df.append(connector.df.copy())

        return self.init(dataframe=_df)

    @property
    def df(self):
        return self._df

    @property
    def size(self):
        return len(self._df)

    @property
    def image0(self):
        return self._df.image0

    @property
    def image1(self):
        return self._df.image1

    @property
    def cluster(self):
        return self._df.cluster.unique()

    @property
    def group(self):
        return self._df.group.unique()

    @classmethod
    @abstractmethod
    def connect(cls, root_dir, filename):
        raise NotImplementedError('Not implemented')

    @classmethod
    @abstractmethod
    def init(cls, dataframe=None):
        return cls(dataframe=dataframe)

    @abstractmethod
    def collate_fn(self, data):
        raise NotImplementedError('Not implemented')

    def select_group(self, groups):
        if isinstance(groups, list):
            idx = self.df.group.isin(groups)
        elif isinstance(groups, str):
            idx = self.df.group.isin([groups])
        else:
            raise TypeError("groups: должен быть или str или list<str>")

        return self.init(dataframe=self.df.loc[idx])

    def select_cluster(self, clusters):
        if isinstance(clusters, list):
            idx = self.df.cluster.isin(clusters)
        elif isinstance(clusters, str):
            idx = self.df.cluster.isin([clusters])
        else:
            raise TypeError("clusters: должен быть или str или list<str>")

        return self.init(dataframe=self.df.loc[idx])

    def calculate_stat_coeffs(self, n_bootsrap=10, sample_size=20, filename=None):
        def worker_fn(filenames):
            means = np.zeros((len(filenames), 3), dtype=np.float32)
            stds = np.zeros((len(filenames), 3), dtype=np.float32)
            for idx, _filename in enumerate(filenames):
                _mean, _std = imread_and_mean_and_std(_filename)
                means[idx, :] = _mean
                stds[idx, :] = _std
            return means.mean(axis=0), stds.mean(axis=0)

        bootstrap_means = np.zeros(shape=(2, n_bootsrap, 3))
        bootstrap_stds = np.zeros(shape=(2, n_bootsrap, 3))

        for i, image in enumerate([self.image0, self.image1]):

            stat_iter = creating_task_for_pool(sampler=np.random.choice(image, size=sample_size * n_bootsrap),
                                               batch_size=sample_size, worker_pool=self._worker_pool,
                                               worker_fn=worker_fn, prefetch=self._prefetch)

            for idx_stat, stat in tqdm.tqdm(enumerate(stat_iter),
                                            desc='Изображение {}'.format(i),
                                            total=len(stat_iter)):
                mean, std = stat
                bootstrap_means[i, idx_stat, :] = mean
                bootstrap_stds[i, idx_stat, :] = std

        mv = bootstrap_means.mean(axis=1)
        sv = bootstrap_stds.mean(axis=1)

        if filename:
            with open(filename, 'w') as file:
                file.write("image0 mean value: {value}\n".format(value=mv[0]))
                file.write("image1 mean value: {value}\n".format(value=mv[1]))
                file.write("image0 std value: {value}\n".format(value=sv[0]))
                file.write("image1 std value: {value}\n".format(value=sv[1]))

        return mv, sv

    def _calculate_statistics(self):
        def worker_fn(filenames):
            a = np.zeros(len(filenames), dtype=np.bool)
            for _idx, filename in enumerate(filenames):
                a[_idx] = pil_to_nparray(imread_full(filename)).max() == 1
            return a.sum()

        index = pd.MultiIndex.from_product([self.cluster, self.group],
                                           names=['cluster', 'group'])
        info = pd.DataFrame(index=index,
                            columns=['count', 'changed'])

        for group in tqdm.tqdm(self.df.group.unique(), desc="Проверка GT"):
            idx = self.df.group == group
            series = self.df.loc[idx].copy()

            worker_iter = creating_task_for_pool(sampler=np.random.choice(series["mask"], size=len(series)),
                                                 batch_size=1, worker_pool=self._worker_pool,
                                                 worker_fn=worker_fn, prefetch=self._prefetch)

            changed = np.zeros(len(series))
            for i, res in enumerate(worker_iter):
                changed[i] = res

            info.loc[series.iloc[0]["cluster"], group] = [len(series), np.sum(changed)]

        return info.dropna()

    def describe(self, filename: str = None):
        info = self._calculate_statistics()

        if filename:
            info.to_csv(filename, sep='\t', float_format='%.2f')
        else:
            print(info)
            print('Выбрано: {total} объектов'.format(total=self.size))
        return info
