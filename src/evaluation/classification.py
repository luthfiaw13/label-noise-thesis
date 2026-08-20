
"""
Evaluasi klasifikasi untuk validation/test dataset.
"""

import numpy as np

import torch

from sklearn.metrics import (
    f1_score
)


def evaluasi_klasifikasi(
    model,
    dataloader,
    criterion,
    device,
    gunakan_amp: bool = True,
):
    """
    Menghitung validation loss, accuracy, dan Macro-F1.

    Return
    ------
    metrics:
        dictionary metric.

    y_true:
        ground-truth labels.

    y_pred:
        predicted labels.
    """

    model.eval()

    total_loss = 0.0

    total_sample = 0

    total_benar = 0

    seluruh_target = []

    seluruh_prediksi = []


    amp_aktif = (
        gunakan_amp
        and
        device.type == "cuda"
    )


    with torch.inference_mode():

        for (
            gambar,
            label,
            _
        ) in dataloader:

            gambar = gambar.to(
                device,
                non_blocking=True
            )

            label = label.to(
                device,
                non_blocking=True
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


            batch_size = label.size(
                0
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


            seluruh_target.extend(
                label
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            seluruh_prediksi.extend(
                prediksi
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


    y_true = np.asarray(
        seluruh_target,
        dtype=np.int64
    )

    y_pred = np.asarray(
        seluruh_prediksi,
        dtype=np.int64
    )


    metrics = {

        "loss":
            (
                total_loss
                /
                total_sample
            ),

        "accuracy":
            (
                total_benar
                /
                total_sample
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
    }


    return (
        metrics,
        y_true,
        y_pred
    )
