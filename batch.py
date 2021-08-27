import matplotlib.pyplot as plt
from torchvision import utils


# 辅助功能：显示批次
def show_landmarks_batch(sample_batch):
    """Show image with landmarks for a batch of samples."""
    image_batch, landmarks_batch = sample_batch['image'], sample_batch['landmarks']
    batch_size = len(image_batch)
    im_size = image_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(image_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for idx in range(batch_size):
        plt.scatter(landmarks_batch[idx, :, 0].numpy() + idx * im_size + (idx + 1) * grid_border_size,
                    landmarks_batch[idx, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')
