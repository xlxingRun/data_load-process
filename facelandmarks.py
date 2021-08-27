from torch.utils.data import Dataset
import os
import pandas as pd
from skimage import io
import numpy as np


class FaceLandmarkDataset(Dataset):
    """面部标记数据集."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file: string, 带注释的csv文件的路径
        :param root_dir: string, 包含所有图像的目录
        :param transform: callable, optional, 一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        s = {'image': image, 'landmarks': landmarks}

        if self.transform:
            s = self.transform(s)

        return s
