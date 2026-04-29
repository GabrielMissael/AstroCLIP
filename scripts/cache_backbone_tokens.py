import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Offline workaround (for this trusted local HF dataset only!!!)
import pyarrow_hotfix
pyarrow_hotfix.uninstall()
import pyarrow as pa
pa.PyExtensionType.set_auto_load(True)

import datasets
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from astroclip.data.datamodule import AstroClipCollator
from astroclip.models.astroclip import ImageHead, SpectrumHead

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temporary-directory", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--image-config", required=True)
    parser.add_argument("--image-weights", required=True)
    parser.add_argument("--image-save-dir", required=True)
    parser.add_argument("--spectrum-weights", required=True)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = datasets.load_dataset(args.dataset, split=args.split, streaming=True)
    dataset = dataset.with_format("torch")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        collate_fn=AstroClipCollator(),
    )

    print("Loading AstroDINO model...")
    notebook_stdout = sys.stdout
    try:
        image_head = ImageHead(
            config=args.image_config,
            model_weights=args.image_weights,
            save_directory=args.image_save_dir,
            freeze_backbone=True,
        ).cuda()
    finally:
        sys.stdout = notebook_stdout

    print("Loading SpecFormer model...")
    spectrum_head = SpectrumHead(
        model_path=args.spectrum_weights,
        freeze_backbone=True,
    ).cuda()

    image_head.backbone.eval()
    spectrum_head.backbone.eval()

    out_dir = Path(args.out_dir)
    temporary_directory = Path(args.temporary_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary_directory.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    written_total = 0
    written_shard = 0
    file = None

    def open_shard(index):
        temporary_file = temporary_directory / f"{args.split}_{index:05d}.h5.temporary"
        h5_file = h5py.File(temporary_file, "w")
        h5_file.attrs["split"] = args.split
        h5_file.attrs["shard_index"] = index

        image_dataset = h5_file.create_dataset(
            "image_tokens",
            shape=(args.shard_size, 144, 1024),
            dtype="float32",
            chunks=(1, 144, 1024), # Shape from AstroDino
        )
        spectrum_dataset = h5_file.create_dataset(
            "spectrum_tokens",
            shape=(args.shard_size, 778, 768),
            dtype="float32",
            chunks=(1, 778, 768), # Shape from SpecFormer
        )
        targetid_dataset = h5_file.create_dataset(
            "targetid", shape=(args.shard_size,), dtype="int64"
        )
        return temporary_file, h5_file, image_dataset, spectrum_dataset, targetid_dataset

    def close_shard(temporary_file, h5_file, count, index):
        h5_file.attrs["written"] = count
        h5_file.close()

        final_file = out_dir / f"{args.split}_{index:05d}.h5"
        shutil.move(temporary_file, final_file)

        manifest = {
            "dataset": args.dataset,
            "split": args.split,
            "shard_index": index,
            "written": count,
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
            "image_config": args.image_config,
            "image_weights": args.image_weights,
            "spectrum_weights": args.spectrum_weights,
        }
        final_file.with_suffix(".h5.json").write_text(json.dumps(manifest, indent=2))

    temporary_file, file, image_dataset, spectrum_dataset, targetid_dataset = open_shard(
        shard_index
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching tokens"):
            image = batch["image"].cuda(non_blocking=True)
            spectrum = batch["spectrum"].cuda(non_blocking=True)

            image_tokens = image_head.backbone.patch_embed(image)
            for block in image_head.backbone.blocks:
                image_tokens = block(image_tokens)
            image_tokens = image_head.backbone.norm(image_tokens).cpu()

            spectrum_tokens = spectrum_head.backbone(spectrum)["embedding"].cpu()

            offset = 0
            batch_size = image_tokens.shape[0]

            while offset < batch_size:
                remaining_shard = args.shard_size - written_shard
                remaining_batch = batch_size - offset
                take = min(remaining_shard, remaining_batch)

                shard_slice = slice(written_shard, written_shard + take)
                batch_slice = slice(offset, offset + take)

                image_dataset[shard_slice] = image_tokens[batch_slice].numpy()
                spectrum_dataset[shard_slice] = spectrum_tokens[batch_slice].numpy()
                targetid_dataset[shard_slice] = batch["targetid"][batch_slice].numpy()

                written_shard += take
                written_total += take
                offset += take

                if written_shard == args.shard_size:
                    close_shard(temporary_file, file, written_shard, shard_index)
                    shard_index += 1
                    written_shard = 0
                    (
                        temporary_file,
                        file,
                        image_dataset,
                        spectrum_dataset,
                        targetid_dataset,
                    ) = open_shard(shard_index)

    if written_shard > 0:
        close_shard(temporary_file, file, written_shard, shard_index)
    else:
        file.close()
        os.remove(temporary_file)

    print(f"Done. Wrote {written_total} samples to {out_dir}")


if __name__ == "__main__":
    main()
