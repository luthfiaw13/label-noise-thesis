
"""
Training engine dengan fixed optimizer-step budget.

Primary experiment penelitian menggunakan fixed optimizer
steps agar perbandingan tidak tercampur dengan perbedaan
jumlah update ketika ukuran dataset berubah setelah cleaning.
"""

import math
import time

import torch

from torch.optim.lr_scheduler import (
    LambdaLR
)

from src.evaluation.classification import (
    evaluasi_klasifikasi
)


def buat_scheduler_warmup_cosine(
    optimizer,
    total_steps: int,
    warmup_steps: int,
):
    """
    Linear warmup dilanjutkan cosine decay.
    """

    if total_steps <= 0:

        raise ValueError(
            "total_steps harus > 0."
        )

    if (
        warmup_steps < 0
        or
        warmup_steps >= total_steps
    ):

        raise ValueError(
            "warmup_steps harus >=0 dan < total_steps."
        )


    def faktor_lr(
        current_step
    ):

        # ----------------------------------------------------
        # Linear warmup.
        # ----------------------------------------------------

        if (
            warmup_steps > 0
            and
            current_step < warmup_steps
        ):

            return (
                current_step + 1
            ) / warmup_steps


        # ----------------------------------------------------
        # Cosine decay.
        # ----------------------------------------------------

        progress = (
            current_step
            -
            warmup_steps
        ) / max(
            1,
            total_steps
            -
            warmup_steps
        )

        progress = min(
            max(
                progress,
                0.0
            ),
            1.0
        )


        return (
            0.5
            *
            (
                1.0
                +
                math.cos(
                    math.pi
                    *
                    progress
                )
            )
        )


    return LambdaLR(
        optimizer,
        lr_lambda=faktor_lr
    )


def latih_hingga_step(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler,
    global_step_awal: int,
    global_step_target: int,
    gunakan_amp: bool = True,
):
    """
    Melatih model sampai jumlah optimizer step tertentu.

    Evaluasi validation dilakukan setelah setiap satu
    putaran penuh DataLoader.
    """

    global_step = int(
        global_step_awal
    )

    if global_step_target <= global_step:

        raise ValueError(
            "global_step_target harus lebih besar "
            "dari global_step_awal."
        )


    riwayat = []

    amp_aktif = (
        gunakan_amp
        and
        device.type == "cuda"
    )


    nomor_siklus = 0


    while (
        global_step
        <
        global_step_target
    ):

        nomor_siklus += 1

        model.train()


        total_loss = 0.0

        total_benar = 0

        total_sample = 0

        waktu_mulai = (
            time.perf_counter()
        )


        for (
            gambar,
            label,
            _
        ) in train_loader:

            if (
                global_step
                >=
                global_step_target
            ):

                break


            gambar = gambar.to(
                device,
                non_blocking=True
            )

            label = label.to(
                device,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
                enabled=amp_aktif,
            ):

                logits = model(
                    gambar
                )

                loss = criterion(
                    logits,
                    label
                )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Loss menjadi NaN/Inf."
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            global_step += 1


            if scheduler is not None:

                scheduler.step()


            batch_size = (
                label.size(0)
            )


            total_loss += (
                loss.item()
                *
                batch_size
            )


            prediksi = logits.argmax(
                dim=1
            )


            total_benar += int(
                (
                    prediksi
                    ==
                    label
                )
                .sum()
                .item()
            )


            total_sample += (
                batch_size
            )


        waktu_siklus = (
            time.perf_counter()
            -
            waktu_mulai
        )


        # ----------------------------------------------------
        # Validation.
        # ----------------------------------------------------

        val_metric, _, _ = (
            evaluasi_klasifikasi(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                gunakan_amp=gunakan_amp,
            )
        )


        train_loss = (
            total_loss
            /
            max(
                total_sample,
                1
            )
        )


        train_accuracy = (
            total_benar
            /
            max(
                total_sample,
                1
            )
        )


        learning_rate = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )


        record = {

            "global_step":
                global_step,

            "siklus":
                nomor_siklus,

            "train_loss":
                float(
                    train_loss
                ),

            "train_accuracy":
                float(
                    train_accuracy
                ),

            "val_loss":
                float(
                    val_metric[
                        "loss"
                    ]
                ),

            "val_accuracy":
                float(
                    val_metric[
                        "accuracy"
                    ]
                ),

            "val_macro_f1":
                float(
                    val_metric[
                        "macro_f1"
                    ]
                ),

            "learning_rate":
                float(
                    learning_rate
                ),

            "runtime_siklus_detik":
                float(
                    waktu_siklus
                ),
        }


        riwayat.append(
            record
        )


        print(
            f"Step {global_step:>3}/{global_step_target} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {100 * train_accuracy:.2f}% | "
            f"Val Loss: {val_metric['loss']:.4f} | "
            f"Val Acc: {100 * val_metric['accuracy']:.2f}% | "
            f"LR: {learning_rate:.6f}"
        )


    return (
        global_step,
        riwayat
    )
