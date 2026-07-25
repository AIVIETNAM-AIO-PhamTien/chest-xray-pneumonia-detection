from torch.utils.data import Dataset
from PIL import Image

class CXRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label