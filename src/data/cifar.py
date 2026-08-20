
"""
Dataset wrapper CIFAR-10 untuk eksperimen label noise.

Setiap sample mengembalikan:
    image, label, original_index

Original index dipertahankan karena pada tahap Cleanlab,
noise mask, OOF prediction, dan label detection, identitas
sample harus tetap konsisten.
"""

from typing import Optional, Sequence

import numpy as np

from PIL import Image

from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10

import torchvision.transforms as transforms


# ------------------------------------------------------------
# Statistik normalisasi yang digunakan oleh implementasi
# resmi CIFAR-N.
# ------------------------------------------------------------

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2023,
    0.1994,
    0.2010,
)


def buat_transformasi_cifar10(
    mode: str
):
    """
    Membuat transformasi CIFAR-10.

    mode='train':
        RandomCrop + RandomHorizontalFlip + normalisasi.

    mode='eval':
        Hanya tensor conversion + normalisasi.
    """

    if mode == "train":

        return transforms.Compose(
            [
                transforms.RandomCrop(
                    32,
                    padding=4
                ),

                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),

                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD
                ),
            ]
        )

    if mode == "eval":

        return transforms.Compose(
            [
                transforms.ToTensor(),

                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD
                ),
            ]
        )

    raise ValueError(
        "mode harus 'train' atau 'eval'."
    )


class CIFAR10Berindeks(Dataset):
    """
    Dataset CIFAR-10 yang mempertahankan original sample ID.

    labels_override dapat digunakan di fase berikutnya
    untuk synthetic atau human noisy labels tanpa mengubah
    citra maupun original index.
    """

    def __init__(
        self,
        root: str,
        indeks: Sequence[int],
        transform,
        labels_override: Optional[Sequence[int]] = None,
        download: bool = False,
    ):

        self.dataset_asli = CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=None
        )

        self.indeks = np.asarray(
            indeks,
            dtype=np.int64
        )

        if self.indeks.ndim != 1:

            raise ValueError(
                "indeks harus berupa array satu dimensi."
            )

        if (
            self.indeks.min() < 0
            or
            self.indeks.max()
            >= len(self.dataset_asli)
        ):

            raise ValueError(
                "Ditemukan indeks di luar rentang CIFAR-10."
            )

        self.transform = transform

        if labels_override is None:

            self.labels = np.asarray(
                self.dataset_asli.targets,
                dtype=np.int64
            )

        else:

            self.labels = np.asarray(
                labels_override,
                dtype=np.int64
            )

            if len(self.labels) != len(
                self.dataset_asli
            ):

                raise ValueError(
                    "labels_override harus memiliki "
                    "panjang 50.000 agar tetap sejajar "
                    "dengan original CIFAR index."
                )


    def __len__(
        self
    ):

        return len(
            self.indeks
        )


    def __getitem__(
        self,
        posisi
    ):

        original_index = int(
            self.indeks[
                posisi
            ]
        )

        array_gambar = (
            self.dataset_asli.data[
                original_index
            ]
        )

        gambar = Image.fromarray(
            array_gambar
        )

        if self.transform is not None:

            gambar = self.transform(
                gambar
            )

        label = int(
            self.labels[
                original_index
            ]
        )

        return (
            gambar,
            label,
            original_index
        )
