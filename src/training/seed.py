
"""
Utilitas reproducibility untuk eksperimen.

Modul ini mengatur random seed Python, NumPy, PyTorch,
CUDA, serta DataLoader worker.

Reproducibility bit-for-bit lintas hardware atau versi
library tidak selalu dijamin, tetapi kontrol ini mengurangi
sumber variasi yang dapat dikendalikan.
"""

import os
import random

import numpy as np
import torch


def atur_seed(
    seed: int,
    deterministik: bool = True,
):
    """
    Mengatur seluruh random seed utama.

    Parameter
    ---------
    seed:
        Random seed eksperimen.

    deterministik:
        Jika True, PyTorch diarahkan menggunakan
        algoritma deterministic bila tersedia.
    """

    # Konfigurasi cuBLAS yang digunakan PyTorch ketika
    # deterministic algorithms diaktifkan pada CUDA.
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = ":4096:8"

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Jangan melakukan benchmark algoritma cuDNN secara
    # dinamis karena pilihan algoritma dapat berbeda antar-run.
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = (
        deterministik
    )

    torch.use_deterministic_algorithms(
        deterministik,
        warn_only=False
    )


def seed_worker(
    worker_id: int
):
    """
    Menyetarakan seed NumPy dan Python random
    pada setiap DataLoader worker.
    """

    worker_seed = (
        torch.initial_seed()
        %
        (2 ** 32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


def buat_generator(
    seed: int
) -> torch.Generator:
    """
    Membuat torch.Generator dengan seed tertentu.
    """

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return generator
