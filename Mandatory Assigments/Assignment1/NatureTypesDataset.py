from torch.utils.data import Dataset
import pandas as pd
import torch
from skimage import io


# the Dataset and DataLoader classes are used to load data into the model
# Implementation taken from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class NatureTypesDataset(Dataset):
    # Nature Types Dataset

    class_dict = {
        "buildings": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5,
    }

    # Map class numbers to class labels
    inv_class_dict = dict(map(reversed, class_dict.items()))

    def __init__(self, csv_file, root_dir, transform=None, train=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx, 0]  # we have the whole path in the csv file
        image = io.imread(img_path)
        label = self.data.iloc[idx, 1]

        # convert label to tensor
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor

        if self.transform:
            image = self.transform(image)

        else:
            image = torch.tensor(image, dtype=torch.float32).permute(
                2, 0, 1
            )  # Ensure image is a tensor, change HWC to CHW format

        return image, label
