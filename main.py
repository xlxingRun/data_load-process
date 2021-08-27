from __future__ import print_function, division

from torch.utils.data import DataLoader
from torchvision import transforms

from batch import show_landmarks_batch
from facelandmarks import FaceLandmarkDataset
from read import show_landmarks
import matplotlib.pyplot as plt
from transform import Rescale, RandomCrop, ToTensor
# 忽略警告
import warnings
warnings.filterwarnings('ignore')
# 打开交互模式
# plt.ion()

if __name__ == '__main__':
    # 使用FaceLandmarkDataset类读取数据到face_dataset
    face_dataset = FaceLandmarkDataset(csv_file='data/faces/face_landmarks.csv',
                                       root_dir='data/faces')
    fig = plt.figure()

    # 为什么调用plt.pause()，会直接画图，n个子图在n张figure上
    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(sample['image'], sample['landmarks'], plt)
        # show_landmarks(**sample)
        if i == 3:
            plt.show()
            break

    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])

    # 在样本上应用上述的每个变换
    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(transformed_sample['image'], transformed_sample['landmarks'], plt)

    plt.show()

    transformed_dataset = FaceLandmarkDataset(csv_file='data/faces/face_landmarks.csv',
                                              root_dir='data/faces/',
                                              transform=transforms.Compose([
                                                  Rescale(256),
                                                  RandomCrop(224),
                                                  ToTensor()
                                              ])
                                              )

    for i in range(len(transformed_dataset)):
        s = transformed_dataset[i]
        print(i, s['image'].size(), s['landmarks'].size())
        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())

        # 观察第4批次并停止
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.show()
            break
