import json
import h5py
import torch
from torch.utils.data import Dataset


class Sentinel_Dataset(Dataset):
    def __init__(self, root_dir, data_file):
        self.root_dir = root_dir
        self.data_file = data_file

        with open(self.data_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.samples = sorted(self.samples, key=lambda s: int(s["timeframe_index"]))

        if not self.samples:
            raise ValueError(f"No Sentinel samples found in {self.data_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        file_path = sample["file_path"]
        group_name = sample["group"]
        sample_id = sample.get("id", group_name)

        with h5py.File(file_path, "r") as hf:
            arr = hf[group_name]["image"][:]   # [4, H, W, 3]

        x = torch.tensor(arr, dtype=torch.float32)
        x = x.unsqueeze(0)  # [1, 4, H, W, 3]

        return x, sample_id


class Sentinel_SequenceDataset(Dataset):
    def __init__(self, root_dir, data_file):
        self.base = Sentinel_Dataset(root_dir, data_file)

        if len(self.base) != 24:
            raise ValueError(
                f"Expected 24 timeframe examples for sequence grouping, got {len(self.base)}"
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        xs = []
        ids = []

        for i in range(24):
            x, sample_id = self.base[i]
            xs.append(x.squeeze(0))
            ids.append(sample_id)

        x_all = torch.stack(xs, dim=0)

        x_seq = x_all.reshape(6, 4, *x_all.shape[1:]).mean(dim=1)
        x_seq = x_seq.unsqueeze(0)

        return x_seq, "SEQ_001"


if __name__ == "__main__":
    dataset = Sentinel_Dataset(
        root_dir=".",
        data_file="/vol/home/s3881946/Downloads/JSON_Loader_Input/sentinel_manifest.json"
    )
    print(f"Independent dataset length: {len(dataset)}")
    x, sample_id = dataset[0]
    print(sample_id, x.shape)

    seq_dataset = Sentinel_SequenceDataset(
        root_dir=".",
        data_file="/vol/home/s3881946/Downloads/JSON_Loader_Input/sentinel_manifest.json"
    )
    x_seq, seq_id = seq_dataset[0]
    print(seq_id, x_seq.shape)