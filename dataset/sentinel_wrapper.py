import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange


class Sentinel_Subset(Dataset):

    def __init__(self, X, Y=None, transform=None):
        self.X = rearrange(X, 'b g h w c -> (b g) h w c')

        if Y is not None:
            self.Y = rearrange(Y, 'b g n d -> (b g) n d')
        else:
            self.Y = None

        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]                   # [H, W, C]
        x = rearrange(x, 'h w c -> c h w')  # [C, H, W]

        if self.transform is not None:
            xi = self.transform(x)
            xj = self.transform(x)
        else:
            xi = x
            xj = x

        if self.Y is None:
            return xi, xj

        y = self.Y[index]
        return xi, xj, y


def get_simclr_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def get_data_loader(X, Y=None, batch_size=32, shuffle=True, num_workers=0):
    transform = get_simclr_transform()
    dataset = Sentinel_Subset(X, Y, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )


if __name__ == "__main__":
    X = torch.randn(1, 4, 224, 224, 3)
    Y = torch.randn(1, 4, 4, 10)
    loader = get_data_loader(X, Y, batch_size=4)

    for xi, xj, y in loader:
        print("xi:", xi.shape)
        print("xj:", xj.shape)
        print("y :", y.shape)
        break