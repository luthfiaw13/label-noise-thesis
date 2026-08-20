
"""
Sistem checkpoint training untuk eksperimen skripsi.

Checkpoint menyimpan:
- model state;
- optimizer state;
- scheduler state;
- AMP GradScaler state;
- global optimizer step;
- konfigurasi run;
- RNG Python;
- RNG NumPy;
- RNG PyTorch CPU;
- RNG CUDA;
- DataLoader generator state.

Checkpoint selalu dimuat terlebih dahulu ke CPU agar state RNG
tidak ikut dipindahkan ke CUDA secara tidak sengaja.
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

    File terlebih dahulu ditulis sebagai file sementara,
    kemudian dipindahkan ke nama final.
    """

    lokasi = Path(lokasi)

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
        # RNG state
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


def _pindahkan_optimizer_ke_device(
    optimizer,
    device,
):
    """
    Memastikan seluruh tensor internal optimizer berada
    pada device yang sama dengan model.
    """

    for state in optimizer.state.values():

        for key, value in state.items():

            if isinstance(
                value,
                torch.Tensor
            ):
                state[key] = value.to(
                    device
                )


def muat_checkpoint(
    lokasi,
    model,
    optimizer,
    scheduler,
    scaler,
    generator_train,
    device,
):
    """
    Memuat checkpoint training secara aman.

    Strategi:
    1. Checkpoint selalu dibaca ke CPU.
    2. Model state dimuat ke model yang sudah berada di device.
    3. Optimizer state dimuat kemudian dipindahkan ke device.
    4. RNG state tetap dipulihkan sebagai CPU ByteTensor.
    5. CUDA RNG dipulihkan secara eksplisit jika tersedia.

    weights_only=False digunakan karena checkpoint milik sendiri
    berisi state Python dan NumPy selain tensor.
    """

    lokasi = Path(lokasi)

    if not lokasi.exists():

        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {lokasi}"
        )

    # --------------------------------------------------------
    # PENTING:
    # Selalu load ke CPU terlebih dahulu.
    # Jangan map seluruh checkpoint langsung ke CUDA karena
    # RNG/DataLoader generator state harus tetap CPU tensor.
    # --------------------------------------------------------

    checkpoint = torch.load(
        lokasi,
        map_location="cpu",
        weights_only=False
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "global_step",
        "config",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "generator_train_state",
    }

    missing_keys = (
        required_keys
        -
        set(checkpoint.keys())
    )

    if missing_keys:

        raise RuntimeError(
            "Checkpoint tidak lengkap. Key hilang: "
            f"{sorted(missing_keys)}"
        )

    # --------------------------------------------------------
    # Restore model.
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Restore optimizer.
    # --------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    _pindahkan_optimizer_ke_device(
        optimizer,
        device
    )

    # --------------------------------------------------------
    # Restore scheduler.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Restore AMP scaler.
    # --------------------------------------------------------

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
    # Restore RNG Python.
    # --------------------------------------------------------

    random.setstate(
        checkpoint[
            "python_rng_state"
        ]
    )

    # --------------------------------------------------------
    # Restore RNG NumPy.
    # --------------------------------------------------------

    np.random.set_state(
        checkpoint[
            "numpy_rng_state"
        ]
    )

    # --------------------------------------------------------
    # Restore RNG PyTorch CPU.
    # torch.set_rng_state membutuhkan CPU ByteTensor.
    # --------------------------------------------------------

    torch_rng_state = checkpoint[
        "torch_rng_state"
    ]

    if not isinstance(
        torch_rng_state,
        torch.Tensor
    ):

        torch_rng_state = torch.tensor(
            torch_rng_state,
            dtype=torch.uint8
        )

    torch_rng_state = (
        torch_rng_state
        .detach()
        .cpu()
        .to(dtype=torch.uint8)
        .contiguous()
    )

    torch.set_rng_state(
        torch_rng_state
    )

    # --------------------------------------------------------
    # Restore CUDA RNG.
    # CUDA RNG state juga direpresentasikan sebagai ByteTensor.
    # --------------------------------------------------------

    cuda_rng_state = checkpoint.get(
        "cuda_rng_state"
    )

    if (
        torch.cuda.is_available()
        and
        cuda_rng_state is not None
    ):

        jumlah_gpu = torch.cuda.device_count()

        if len(cuda_rng_state) != jumlah_gpu:

            raise RuntimeError(
                "Jumlah CUDA RNG state pada checkpoint "
                f"({len(cuda_rng_state)}) berbeda dengan "
                f"jumlah GPU runtime ({jumlah_gpu})."
            )

        cuda_states_final = []

        for state in cuda_rng_state:

            if not isinstance(
                state,
                torch.Tensor
            ):

                state = torch.tensor(
                    state,
                    dtype=torch.uint8
                )

            state = (
                state
                .detach()
                .cpu()
                .to(dtype=torch.uint8)
                .contiguous()
            )

            cuda_states_final.append(
                state
            )

        torch.cuda.set_rng_state_all(
            cuda_states_final
        )

    # --------------------------------------------------------
    # Restore DataLoader generator.
    # torch.Generator CPU membutuhkan CPU ByteTensor.
    # --------------------------------------------------------

    generator_state = checkpoint.get(
        "generator_train_state"
    )

    if (
        generator_train is not None
        and
        generator_state is not None
    ):

        if not isinstance(
            generator_state,
            torch.Tensor
        ):

            generator_state = torch.tensor(
                generator_state,
                dtype=torch.uint8
            )

        generator_state = (
            generator_state
            .detach()
            .cpu()
            .to(dtype=torch.uint8)
            .contiguous()
        )

        generator_train.set_state(
            generator_state
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
