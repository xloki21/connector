import os
import glob
from abc import abstractmethod
from multiprocessing.pool import ThreadPool
from typing import Iterable

import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from PIL.ImageStat import Stat

from connector.tools.sampler import *
from connector.tools.visualization import draw_poly_items, create_custom_colordict
from connector.tools.imaging import imread_full, imread_lazy, crop_patches, pil_to_nparray
from connector.tools.imaging import imread_and_mean_and_std
from connector.tools.json import save_json_data
from connector.tools.dataloader import _MultiWorkerIter


class DataFrameCommonOpsMixin:
    """Mixin: Common ops and properties.
    """

    def __init__(self, dataframe=None):
        self._df = dataframe

    def __add__(self, connector):
        _df = self._df.append(connector.df.copy())
        return self.init(dataframe=_df)

    @property
    def df(self):
        return self._df

    @property
    def size(self):
        return len(self._df)

    @classmethod
    def init(cls, dataframe=None):
        return cls(dataframe=dataframe)


class LocalisationDatasetConnector(DataFrameCommonOpsMixin):
    def __init__(self, dataframe):
        super().__init__(dataframe=dataframe)
        self._thread = os.cpu_count() - 2
        self._worker_pool = None
        self.default_label_set = None

    def __str__(self):
        labels_info = "Dataset with {n} label(s): {labels}".format(n=len(self.labels), labels=self.labels)
        objects_info = "Number of objects: {n}".format(n=self.size)
        images_info = "Number of unique images: {n}".format(n=len(self.images))
        return "\n".join([labels_info, objects_info, images_info])

    def __len__(self):
        return len(self.images)

    @property
    def images(self):
        return self._df.image.unique()

    @property
    def labels(self):
        return self._df.label.unique()

    @property
    def width(self):
        return self.df.x.apply(lambda x: max(x)) - self.df.x.apply(lambda x: min(x))

    @property
    def height(self):
        return self.df.y.apply(lambda x: max(x)) - self.df.y.apply(lambda x: min(x))

    @property
    def hbbox(self):
        xmin = self.df.x.apply(lambda x: min(x))
        xmin.name = 'xmin'
        ymin = self.df.y.apply(lambda x: min(x))
        ymin.name = 'ymin'
        xmax = self.df.x.apply(lambda x: max(x))
        xmax.name = 'xmax'
        ymax = self.df.y.apply(lambda x: max(x))
        ymax.name = 'ymax'

        data = list(zip(zip(xmin, ymin),
                        zip(xmin, ymax),
                        zip(xmax, ymax),
                        zip(xmax, ymin)))

        return pd.Series(data=data, index=xmin.index, name='hbbox')

    @property
    def obbox(self):
        result = (self.df.x + self.df.y).apply(lambda x: [(x[i], x[i + 4]) for i in range(len(x) // 2)])
        return result

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
        for i, filename in tqdm.tqdm(enumerate(self.images),
                                     desc='loading image information',
                                     total=len(self.images),
                                     ascii=True):
            imw, imh = imread_lazy(filename).size
            image["file_name"] = filename
            image["height"] = imh
            image["width"] = imw
            image["id"] = imagetoid[filename]
            images[i] = image.copy()

        extended = self.df.copy()
        extended['hbbox'] = self.hbbox

        for i, entry in tqdm.tqdm(extended.iterrows(),
                                  total=self.size,
                                  desc='converting annotation to COCO format',
                                  ascii=True):
            xmin = entry['hbbox'][0][0]
            ymin = entry['hbbox'][0][1]
            xmax = entry['hbbox'][2][0]
            ymax = entry['hbbox'][2][1]
            width = xmax - xmin + 1
            height = ymax - ymin + 1
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

    def create_csv_format_annotation(self, annotationfile, labelmappingfile):
        extended = self.df.copy()
        hbbox = self.hbbox

        extended['xmin'] = hbbox.apply(lambda x: int(x[0][0]))
        extended['ymin'] = hbbox.apply(lambda x: int(x[0][1]))
        extended['xmax'] = hbbox.apply(lambda x: int(x[2][0]))
        extended['ymax'] = hbbox.apply(lambda x: int(x[2][1]))

        extended.to_csv(annotationfile,
                        header=None, index=None,
                        columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])

        with open(labelmappingfile, 'w') as lf:
            lf.write("\n".join(["{label},{id}".format(label=label, id=id) for id, label in enumerate(self.labels)]))

    @classmethod
    @abstractmethod
    def connect(cls, image_dir, label_dir):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def collater_fn(self, data):
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

    def convert(self, patch_shape, imagedir, labeldir, rel_shift=1.0, padding=True, max_workers=None):
        if not os.path.exists(labeldir):
            os.makedirs(labeldir, exist_ok=True)

        if not os.path.exists(imagedir):
            os.makedirs(imagedir, exist_ok=True)

        if max_workers is None:
            max_workers = os.cpu_count()

        images = self.images

        df = None
        dy = int(patch_shape[0] * rel_shift)
        dx = int(patch_shape[1] * rel_shift)
        for imgname in tqdm.tqdm(images, desc='Converting', ascii=True):
            img_basename = os.path.basename(imgname)
            img_basename_wo_ext, ext = os.path.splitext(img_basename)

            xmin, ymin, xmax, ymax = crop_patches(image_filepath=imgname,
                                                  patch_shape=patch_shape,
                                                  dx=dx,
                                                  dy=dy,
                                                  patch_root_dir=imagedir,
                                                  ext=ext,
                                                  padding=padding,
                                                  max_workers=max_workers)
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

    def draw_image_annotation(self, image_file, color_dict=None, mode='hbb'):
        assert mode in ('hbb', 'obb')
        # default drawing method
        subset = self.select_images(image_idx=image_file)
        if color_dict is None:
            color_dict = create_custom_colordict(self.default_label_set, cmap='hsv', alpha=120)
        annotated_image = draw_poly_items(image_filename=image_file,
                                          items=subset.hbbox if mode == 'hbb' else subset.obbox,
                                          labels=subset.df.label,
                                          scores=None,
                                          filled=True,
                                          color_dict=color_dict)
        return annotated_image

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
        self._worker_pool = ThreadPool(self._thread)

        def worker_fn(filenames):
            means = np.zeros((len(filenames), 3), dtype=np.float32)
            stds = np.zeros((len(filenames), 3), dtype=np.float32)
            for idx, _filename in enumerate(filenames):
                imstat = Stat(imread_lazy(_filename))
                means[idx, :] = imstat.mean
                stds[idx, :] = imstat.stddev
            return means.mean(axis=0), stds.mean(axis=0)

        bootstrap_means = np.zeros(shape=(n_bootsrap, 3))
        bootstrap_stds = np.zeros(shape=(n_bootsrap, 3))

        batch_sampler_image = BatchSampler(np.random.choice(self.images, size=sample_size * n_bootsrap),
                                           batch_size=sample_size)

        worker_iter = _MultiWorkerIter(worker_pool=self._worker_pool,
                                       batch_sampler=batch_sampler_image,
                                       worker_fn=worker_fn,
                                       prefetch=self._thread * 2)

        for idx_stat, stat in tqdm.tqdm(enumerate(worker_iter),
                                        desc='Calculating stat coeffs',
                                        ascii=True,
                                        total=n_bootsrap):
            mean, std = stat
            bootstrap_means[idx_stat, :] = mean
            bootstrap_stds[idx_stat, :] = std

        mv, sv = bootstrap_means.mean(axis=0), bootstrap_stds.mean(axis=0)
        if filename:
            with open(filename, 'w') as file:
                file.write("mean value: {value}\n".format(value=mv))
                file.write("std value: {value}\n".format(value=sv))
        return mv, sv

    def show(self, cmap='hsv', mode='hbb'):
        assert mode in ('hbb', 'obb')
        color_dict = create_custom_colordict(self.default_label_set, cmap=cmap, alpha=120)
        for image in self.images:
            print(image)
            result = self.draw_image_annotation(image_file=image, color_dict=color_dict, mode=mode)
            plt.imshow(pil_to_nparray(result))
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
        if len(pandas_sharded_dataframe) > 0:
            for shard in tqdm.tqdm(pandas_sharded_dataframe, desc='Load dataframe', ascii=True):
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
        else:
            return None

    def split(self, param, seed: int = None):
        split_sizes = []
        if isinstance(param, int):
            split_sizes = [self.size // param] * param
        elif isinstance(param, Iterable):
            if sum(param) <= 1:
                split_sizes = [int(self.size * value) for value in param]
            else:
                split_sizes = param
        if seed:
            ind = np.arange(0, self.size, dtype=np.int64)
            np.random.seed(seed=seed)
            np.random.shuffle(ind)
            self.df.index = pd.Int64Index(ind)
        result = []
        index_to = 0
        for split_size in split_sizes:
            index_from = index_to
            index_to += split_size
            result.append(self.init(dataframe=self.df.iloc[index_from:index_to]))
        return result


class ChangeDetectionDatasetConnector(DataFrameCommonOpsMixin):
    def __init__(self, dataframe):
        super().__init__(dataframe=dataframe)
        self._threads = os.cpu_count() - 2
        self._worker_pool = None

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

    @abstractmethod
    def collater_fn(self, data):
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
        self._worker_pool = ThreadPool(self._threads)

        def worker_fn(filenames):
            means = np.zeros((len(filenames), 3), dtype=np.float32)
            stds = np.zeros((len(filenames), 3), dtype=np.float32)
            for idx, _filename in enumerate(filenames):
                imstat = Stat(imread_lazy(_filename))
                _mean, _std = imread_and_mean_and_std(_filename)
                means[idx, :] = _mean
                stds[idx, :] = _std
            return means.mean(axis=0), stds.mean(axis=0)

        bootstrap_means = np.zeros(shape=(2, n_bootsrap, 3))
        bootstrap_stds = np.zeros(shape=(2, n_bootsrap, 3))

        for i, image in enumerate([self.image0, self.image1]):
            batch_sampler_image1 = BatchSampler(np.random.choice(image, size=sample_size * n_bootsrap),
                                                batch_size=sample_size)

            stat_iter = _MultiWorkerIter(worker_pool=self._worker_pool,
                                         batch_sampler=batch_sampler_image1,
                                         worker_fn=worker_fn,
                                         prefetch=self._threads * 2)

            for idx_stat, stat in tqdm.tqdm(enumerate(stat_iter),
                                            desc='Изображение {}'.format(i),
                                            total=n_bootsrap):
                mean, std = stat
                bootstrap_means[i, idx_stat, :] = mean
                bootstrap_stds[i, idx_stat, :] = std

        mv = bootstrap_means.mean(axis=(1, 2))
        sv = bootstrap_stds.mean(axis=(1, 2))

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

            batch_sampler = BatchSampler(np.random.choice(series["mask"], size=len(series)),
                                         batch_size=1)

            worker_iter = _MultiWorkerIter(worker_pool=self._worker_pool,
                                           batch_sampler=batch_sampler,
                                           worker_fn=worker_fn,
                                           prefetch=self._threads * 2)

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

    def convert(self, patch_shape, root_dir, rel_shift=1.0, padding=True):

        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        for cluster in self.cluster:
            cluster_data = self.select_cluster(clusters=cluster)
            images0 = cluster_data.image0
            images1 = cluster_data.image1
            gts = cluster_data.df['mask']
            dy = int(patch_shape[0] * rel_shift)
            dx = int(patch_shape[1] * rel_shift)

            cluster_dir = os.path.join(root_dir, cluster)
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir, exist_ok=True)

            imagedir_t0 = os.path.join(cluster_dir, 't0')
            imagedir_t1 = os.path.join(cluster_dir, 't1')
            imagedir_gt = os.path.join(cluster_dir, 'gt')

            if not os.path.exists(imagedir_t0):
                os.makedirs(imagedir_t0, exist_ok=True)

            if not os.path.exists(imagedir_t1):
                os.makedirs(imagedir_t1, exist_ok=True)

            if not os.path.exists(imagedir_gt):
                os.makedirs(imagedir_gt, exist_ok=True)

            with open(os.path.join(root_dir, 'data.txt'), 'w') as file:
                for imgname1, imgname2, imgnamegt in tqdm.tqdm(zip(images0, images1, gts),
                                                               desc='Converting timeline data'):
                    img1_basename = os.path.basename(imgname1)
                    img1_basename_wo_ext, ext1 = os.path.splitext(img1_basename)

                    img2_basename = os.path.basename(imgname2)
                    img2_basename_wo_ext, ext2 = os.path.splitext(img2_basename)

                    imggt_basename = os.path.basename(imgnamegt)
                    imggt_basename_wo_ext, ext3 = os.path.splitext(imggt_basename)

                    # crop t0 data
                    xmin1, ymin1, xmax1, ymax1 = crop_patches(image_filepath=imgname1,
                                                              patch_shape=patch_shape,
                                                              dx=dx,
                                                              dy=dy,
                                                              patch_root_dir=imagedir_t0,
                                                              ext=ext1,
                                                              padding=padding,
                                                              max_workers=8)
                    cropnames1 = [os.path.join(cluster, 't0',
                                               "{basename}_{ymin:04}_{xmin:04}{ext}".format(
                                                   basename=img1_basename_wo_ext,
                                                   ymin=y, xmin=x, ext=ext1)) for
                                  (x, y) in zip(xmin1, ymin1)]
                    # crop t1 data
                    xmin2, ymin2, xmax2, ymax2 = crop_patches(image_filepath=imgname2,
                                                              patch_shape=patch_shape,
                                                              dx=dx,
                                                              dy=dy,
                                                              patch_root_dir=imagedir_t1,
                                                              ext=ext2,
                                                              padding=padding,
                                                              max_workers=8)
                    cropnames2 = [os.path.join(cluster, 't1',
                                               "{basename}_{ymin:04}_{xmin:04}{ext}".format(
                                                   basename=img2_basename_wo_ext,
                                                   ymin=y, xmin=x, ext=ext2)) for
                                  (x, y) in zip(xmin2, ymin2)]
                    # crop gt data
                    xmingt, ymingt, xmaxgt, ymaxgt = crop_patches(image_filepath=imgnamegt,
                                                                  patch_shape=patch_shape,
                                                                  dx=dx,
                                                                  dy=dy,
                                                                  patch_root_dir=imagedir_gt,
                                                                  ext=ext3,
                                                                  padding=padding,
                                                                  max_workers=8)
                    cropnames3 = [os.path.join(cluster, 'gt',
                                               "{basename}_{ymin:04}_{xmin:04}{ext}".format(
                                                   basename=imggt_basename_wo_ext,
                                                   ymin=y, xmin=x, ext=ext3)) for
                                  (x, y) in zip(xmingt, ymingt)]

                    for cn1, cn2, cn3 in zip(cropnames1, cropnames2, cropnames3):
                        print("{value1} {value2} {value3}".format(value1=cn1,
                                                                  value2=cn2,
                                                                  value3=cn3), file=file)
