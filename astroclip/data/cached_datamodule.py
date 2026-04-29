import h5py
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from bisect import bisect_right
from pathlib import Path

class CachedTokenDataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        self.shards = self._find_shards()
        self.lengths = [self._read_length(shard) for shard in self.shards]
        self.starts = []

        total = 0
        for length in self.lengths:
            self.starts.append(total)
            total += length

        self.total = total
        self._files = {}

    def _find_shards(self):
        if self.path.is_file():
            shards = [self.path]
        else:
            shards = sorted(self.path.glob("*.h5"))
        return shards

    @staticmethod
    def _read_length(shard):
        with h5py.File(shard, "r") as file:
            return int(file.attrs.get("written", file["targetid"].shape[0]))

    def __len__(self):
        return self.total

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _file(self, shard_index):
        if shard_index not in self._files:
            self._files[shard_index] = h5py.File(self.shards[shard_index], "r")
        return self._files[shard_index]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_index = bisect_right(self.starts, index) - 1
        local_index = index - self.starts[shard_index]
        file = self._file(shard_index)
        return {
            "image": torch.from_numpy(file["image_tokens"][local_index]),
            "spectrum": torch.from_numpy(file["spectrum_tokens"][local_index]),
            "targetid": torch.as_tensor(file["targetid"][local_index]),
        }


    def close(self):
        for file in getattr(self, "_files", {}).values():
            file.close()
        self._files = {}
    def __del__(self):
        self.close()


class CachedAstroClipDataloader(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int = 2048,
        num_workers: int = 2,
        drop_last: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.train_dataset = CachedTokenDataset(self.hparams.train_path)
        self.val_dataset = CachedTokenDataset(self.hparams.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=self.hparams.drop_last,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=self.hparams.drop_last,
            pin_memory=self.hparams.pin_memory,
        )
