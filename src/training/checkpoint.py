
"""
Checkpoint training untuk Google Colab.

Checkpoint menyimpan:
- model
- optimizer
- scheduler
- AMP scaler
- optimizer step
- random states
- DataLoader generator state
- konfigurasi run
"""

from pathlib import Path

import os
import random

import numpy as np
import torch


def simpan_checkpoint(
    lokasi,
    model,
    optimizer,
    scheduler,
    scaler,
    global_step,
    generator_train,
    config,
):
    """
    Menyimpan checkpoint secara atomic.
    """

    lokasi = Path(
        lokasi
    )

    lokasi.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

        "scaler_state_dict":
            (
                scaler.state_dict()
                if scaler is not None
                else None
            ),

        "global_step":
            int(global_step),

        "config":
            config,

        # ----------------------------------------------------
        # RNG states
        # ----------------------------------------------------

        "python_rng_state":
            random.getstate(),

        "numpy_rng_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),

        "generator_train_state":
            (
                generator_train.get_state()
                if generator_train is not None
                else None
            ),
    }


    lokasi_temp = lokasi.with_suffix(
        lokasi.suffix + ".tmp"
    )


    torch.save(
        checkpoint,
        lokasi_temp
    )


    os.replace(
        lokasi_temp,
        lokasi
    )


def muat_checkpoint(
    lokasi,
    model,
    optimizer,
    scheduler,
    scaler,
    generator_train,
    map_location,
):
    """
    Memuat checkpoint dan mengembalikan progress training.

    weights_only=False digunakan karena checkpoint ini
    dibuat sendiri dan berisi state Python/NumPy selain tensor.
    """

    checkpoint = torch.load(
        lokasi,
        map_location=map_location,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    if (
        scheduler is not None
        and
        checkpoint[
            "scheduler_state_dict"
        ]
        is not None
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )


    if (
        scaler is not None
        and
        checkpoint[
            "scaler_state_dict"
        ]
        is not None
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )


    # --------------------------------------------------------
    # Restore random states.
    # --------------------------------------------------------

    random.setstate(
        checkpoint[
            "python_rng_state"
        ]
    )


    np.random.set_state(
        checkpoint[
            "numpy_rng_state"
        ]
    )


    torch.set_rng_state(
        checkpoint[
            "torch_rng_state"
        ]
    )


    if (
        torch.cuda.is_available()
        and
        checkpoint[
            "cuda_rng_state"
        ]
        is not None
    ):

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_rng_state"
            ]
        )


    if (
        generator_train is not None
        and
        checkpoint[
            "generator_train_state"
        ]
        is not None
    ):

        generator_train.set_state(
            checkpoint[
                "generator_train_state"
            ]
        )


    return {

        "global_step":
            int(
                checkpoint[
                    "global_step"
                ]
            ),

        "config":
            checkpoint[
                "config"
            ],
    }
