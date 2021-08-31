import torch
from torch.utils.data import Dataset
import os
import cv2


class Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.dataset = os.listdir(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_name = self.dataset[item]
        img_root = self.root + "/" + img_name
        img = cv2.imread(img_root)
        # img = img[..., ::-1]
        img = img.swapaxes(0, 2)
        img = (img / 255 - 0.5) * 2

        return torch.tensor(img, dtype = torch.float32)


# if __name__ == '__main__':
#     data = Dataset("D:/Data_set/Cartoon_faces")
#     print(data[0].shape)
