
"""
Definisi ResNet yang disesuaikan untuk citra CIFAR 32x32.

Model dibangun dari implementasi resmi torchvision:
- ResNet18
- ResNet34
- ResNet50

Perubahan yang dilakukan:
1. Tidak menggunakan pretrained weights.
2. Convolution awal diubah menjadi kernel 3x3, stride 1.
3. Initial max-pooling dihilangkan.
4. Fully-connected output disesuaikan dengan jumlah kelas.

Tujuannya adalah menggunakan stem input yang sama pada seluruh
model sehingga perbandingan kompleksitas tidak tercampur dengan
perbedaan preprocessing awal.
"""

from typing import Literal

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
)


NamaModel = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
]


def bangun_resnet_cifar(
    nama_model: NamaModel,
    jumlah_kelas: int = 10,
) -> nn.Module:
    """
    Membuat CIFAR-compatible ResNet.

    Parameter
    ---------
    nama_model:
        Salah satu dari:
        resnet18, resnet34, atau resnet50.

    jumlah_kelas:
        Jumlah kelas output.

    Return
    ------
    model:
        Model ResNet dengan stem yang telah disesuaikan
        untuk input CIFAR 32x32.
    """

    # --------------------------------------------------------
    # Validasi jumlah kelas.
    # --------------------------------------------------------

    if jumlah_kelas < 2:
        raise ValueError(
            "jumlah_kelas minimal harus 2."
        )


    # --------------------------------------------------------
    # Daftar builder torchvision.
    # --------------------------------------------------------

    pembuat_model = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }


    if nama_model not in pembuat_model:
        raise ValueError(
            f"Model '{nama_model}' tidak didukung. "
            "Gunakan resnet18, resnet34, atau resnet50."
        )


    # --------------------------------------------------------
    # weights=None memastikan tidak menggunakan
    # pretrained ImageNet weights.
    # --------------------------------------------------------

    model = pembuat_model[
        nama_model
    ](
        weights=None
    )


    # --------------------------------------------------------
    # Stem CIFAR:
    #
    # Input CIFAR hanya 32x32 sehingga convolution awal
    # dibuat 3x3, stride 1 dan padding 1.
    #
    # Semua model memakai stem IDENTIK.
    # --------------------------------------------------------

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )


    # --------------------------------------------------------
    # Max-pooling awal ImageNet dihilangkan agar feature map
    # 32x32 tidak mengalami downsampling terlalu dini.
    # --------------------------------------------------------

    model.maxpool = nn.Identity()


    # --------------------------------------------------------
    # Ubah classifier terakhir sesuai jumlah kelas dataset.
    # CIFAR-10 menggunakan 10 kelas.
    # --------------------------------------------------------

    jumlah_fitur = model.fc.in_features

    model.fc = nn.Linear(
        jumlah_fitur,
        jumlah_kelas
    )


    return model


def hitung_parameter_model(
    model: nn.Module
) -> dict:
    """
    Menghitung jumlah parameter model.

    Return
    ------
    dictionary dengan:
    - total_parameter
    - parameter_trainable
    - parameter_non_trainable
    """

    total_parameter = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    parameter_trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    parameter_non_trainable = (
        total_parameter
        -
        parameter_trainable
    )

    return {
        "total_parameter":
            int(total_parameter),

        "parameter_trainable":
            int(parameter_trainable),

        "parameter_non_trainable":
            int(parameter_non_trainable),
    }
