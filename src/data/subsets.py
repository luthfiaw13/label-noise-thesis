
"""
Utilitas pembuatan subset eksperimen.

Subset dibuat secara stratified agar proporsi setiap kelas
tetap seimbang.
"""

import numpy as np

from sklearn.model_selection import (
    train_test_split
)


def buat_split_smoke(
    labels,
    jumlah_subset: int = 2000,
    proporsi_validasi: float = 0.20,
    seed: int = 314159,
):
    """
    Membuat subset smoke test secara stratified.

    Tahapan:
        50.000 CIFAR train
              ↓
        2.000 smoke subset
              ↓
        train + validation

    Official CIFAR test set tidak digunakan.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    semua_indeks = np.arange(
        len(labels)
    )


    # --------------------------------------------------------
    # Ambil subset 2.000 secara stratified.
    # --------------------------------------------------------

    indeks_subset, _ = (
        train_test_split(
            semua_indeks,
            train_size=jumlah_subset,
            stratify=labels,
            random_state=seed,
        )
    )


    label_subset = labels[
        indeks_subset
    ]


    # --------------------------------------------------------
    # Bagi subset menjadi training dan validation.
    # --------------------------------------------------------

    indeks_train, indeks_val = (
        train_test_split(
            indeks_subset,
            test_size=proporsi_validasi,
            stratify=label_subset,
            random_state=seed,
        )
    )


    # Sorting bukan syarat statistik, tetapi membuat artifact
    # indeks lebih mudah diperiksa dan dibandingkan.
    indeks_subset = np.sort(
        indeks_subset
    )

    indeks_train = np.sort(
        indeks_train
    )

    indeks_val = np.sort(
        indeks_val
    )


    return (
        indeks_subset,
        indeks_train,
        indeks_val
    )
