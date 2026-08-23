"""
Symmetric label noise generator.

Digunakan untuk eksperimen:
Label Noise Cleaning Thesis.

Input:
    clean labels

Output:
    noisy labels
    noise mask

Karakteristik:
    - reproducible
    - seed controlled
    - image tidak berubah
    - original index tetap aman
"""


from typing import Tuple

import numpy as np


def generate_symmetric_noise(
    labels,
    noise_rate: float,
    num_classes: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Membuat symmetric label noise.

    Parameters
    ----------
    labels :
        Array label asli.

    noise_rate :
        Proporsi label yang dirusak.
        Contoh:
            0.2 = 20%

    num_classes :
        Jumlah kelas.

    seed :
        Random seed.

    Returns
    -------
    noisy_labels :
        Label setelah noise.

    noise_mask :
        Boolean array.
        True jika label berubah.
    """


    labels = np.asarray(
        labels,
        dtype=np.int64
    )


    rng = np.random.default_rng(
        seed
    )


    noisy_labels = labels.copy()


    noise_mask = np.zeros(
        len(labels),
        dtype=bool
    )


    jumlah_noise = int(
        len(labels) * noise_rate
    )


    indeks_noise = rng.choice(
        len(labels),
        size=jumlah_noise,
        replace=False
    )


    for idx in indeks_noise:

        label_asli = labels[idx]


        kandidat = np.delete(
            np.arange(num_classes),
            label_asli
        )


        label_baru = rng.choice(
            kandidat
        )


        noisy_labels[idx] = label_baru


        noise_mask[idx] = True


    return (
        noisy_labels,
        noise_mask
    )