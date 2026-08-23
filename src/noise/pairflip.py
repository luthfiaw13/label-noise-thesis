"""
Pair flip label noise generator.

CIFAR-10 class transition:

airplane  <-> automobile
bird      <-> deer
cat       <-> dog
frog      <-> horse
ship      <-> truck

"""

from typing import Tuple

import numpy as np



CIFAR10_PAIR_MAPPING = {

    0: 1,   # airplane -> automobile
    1: 0,

    2: 4,   # bird -> deer
    4: 2,

    3: 5,   # cat -> dog
    5: 3,

    6: 7,   # frog -> horse
    7: 6,

    8: 9,   # ship -> truck
    9: 8,
}



def generate_pairflip_noise(
    labels,
    noise_rate: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:


    """
    Generate pair flip noise.

    Returns:
        noisy_labels,
        noise_mask
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


        if label_asli in CIFAR10_PAIR_MAPPING:


            noisy_labels[idx] = (
                CIFAR10_PAIR_MAPPING[
                    label_asli
                ]
            )


            noise_mask[idx] = True



    return (
        noisy_labels,
        noise_mask
    )