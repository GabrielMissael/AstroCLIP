import h5py
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from bisect import bisect_right
from pathlib import Path

class CachedTokenDataset(Dataset):
    def __init__(self, path: str, start: int, stop: int):
        self.path = Path(path)
        self.shards = sorted(self.path.glob("*.h5"))
        self.lengths = [self._read_length(shard) for shard in self.shards]
        self.starts = []

        self.full_total = 0
        for length in self.lengths:
            self.starts.append(self.full_total)
            self.full_total += length

        self.start = start
        self.stop = stop
        self.total = self.stop - self.start
        self._files = {}

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
        index += self.start
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
            try:
                file.close()
            except Exception:
                pass
        self._files = {}
    def __del__(self):
        self.close()


class CachedTokenBatchDataset(Dataset):
    def __init__(
        self,
        path: str,
        start: int,
        stop: int,
        batch_size: int,
        drop_last: bool = True,
    ):
        self.path = Path(path)
        self.shards = sorted(self.path.glob("*.h5"))
        self.lengths = [self._read_length(shard) for shard in self.shards]
        self.starts = []

        self.full_total = 0
        for length in self.lengths:
            self.starts.append(self.full_total)
            self.full_total += length

        self.start = start
        self.stop = stop
        self.total = self.stop - self.start
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._files = {}

        if self.total < 0:
            raise ValueError(f"stop must be >= start, got {start=} and {stop=}")
        if self.start < 0 or self.stop > self.full_total:
            raise ValueError(
                f"requested range [{self.start}, {self.stop}) outside "
                f"cached range [0, {self.full_total})"
            )

    @staticmethod
    def _read_length(shard):
        with h5py.File(shard, "r") as file:
            return int(file.attrs.get("written", file["targetid"].shape[0]))

    def __len__(self):
        if self.drop_last:
            return self.total // self.batch_size
        return (self.total + self.batch_size - 1) // self.batch_size

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _file(self, shard_index):
        if shard_index not in self._files:
            self._files[shard_index] = h5py.File(self.shards[shard_index], "r")
        return self._files[shard_index]

    def __getitem__(self, batch_index):
        if torch.is_tensor(batch_index):
            batch_index = batch_index.item()
        if batch_index < 0:
            batch_index += len(self)
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)

        batch_start = self.start + batch_index * self.batch_size
        batch_stop = min(batch_start + self.batch_size, self.stop)

        images = []
        spectra = []
        targetids = []
        index = batch_start

        while index < batch_stop:
            shard_index = bisect_right(self.starts, index) - 1
            shard_start = self.starts[shard_index]
            shard_stop = shard_start + self.lengths[shard_index]
            take_stop = min(batch_stop, shard_stop)
            local_start = index - shard_start
            local_stop = take_stop - shard_start

            file = self._file(shard_index)
            images.append(torch.from_numpy(file["image_tokens"][local_start:local_stop]))
            spectra.append(torch.from_numpy(file["spectrum_tokens"][local_start:local_stop]))
            targetids.append(torch.from_numpy(file["targetid"][local_start:local_stop]))

            index = take_stop

        if len(images) == 1:
            image = images[0]
            spectrum = spectra[0]
            targetid = targetids[0]
        else:
            image = torch.cat(images, dim=0)
            spectrum = torch.cat(spectra, dim=0)
            targetid = torch.cat(targetids, dim=0)

        return {
            "image": image,
            "spectrum": spectrum,
            "targetid": targetid,
        }

    def close(self):
        for file in getattr(self, "_files", {}).values():
            try:
                file.close()
            except Exception:
                pass
        self._files = {}

    def __del__(self):
        self.close()


class CachedAstroClipDataloader(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_stop: int,
        val_start: int,
        val_stop: int,
        batch_size: int = 2048,
        num_workers: int = 2,
        drop_last: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.train_dataset = CachedTokenDataset(
            self.hparams.data_path,
            start=self.hparams.train_start,
            stop=self.hparams.train_stop,
        )
        self.val_dataset = CachedTokenDataset(
            self.hparams.data_path,
            start=self.hparams.val_start,
            stop=self.hparams.val_stop,
        )

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


class CachedAstroClipBatchDataloader(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_stop: int,
        val_start: int,
        val_stop: int,
        batch_size: int = 256,
        num_workers: int = 0,
        drop_last: bool = True,
        pin_memory: bool = False,
        shuffle: bool = True,
        prefetch_factor: int = 1,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_start = train_start
        self.train_stop = train_stop
        self.val_start = val_start
        self.val_stop = val_stop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def setup(self, stage: str) -> None:
        self.train_dataset = CachedTokenBatchDataset(
            self.data_path,
            start=self.train_start,
            stop=self.train_stop,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        self.val_dataset = CachedTokenBatchDataset(
            self.data_path,
            start=self.val_start,
            stop=self.val_stop,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def _loader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=self.shuffle,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            **self._loader_kwargs(),
        )
