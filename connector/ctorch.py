from torch.utils.data import Dataset

from connector.generic import LocalisationDatasetConnector
from connector.tools.imaging import imread_lazy


class TorchConnector(Dataset):

    def __init__(self, dataframe, transforms_compose=None):
        super().__init__()
        self.connector = LocalisationDatasetConnector(dataframe=dataframe)
        self.transforms_compose = transforms_compose

    def __len__(self):
        return len(self.connector.images)

    def __getitem__(self, idx):
        filename = self.connector.images[idx]
        annotation = self.connector.select_images(image_idx=filename).create_coco_format_annotation()

        return imread_lazy(filename), annotation
