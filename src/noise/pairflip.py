import numpy as np

def generate_pairflip_class_balanced_noise(labels, transition_rule, noise_rate, seed=17):
    rng = np.random.default_rng(seed)
    noisy_labels = labels.copy()
    noise_mask = np.zeros(len(labels), dtype=bool)

    for source, target in transition_rule.items():
        indices = np.where(labels == source)[0]
        jumlah_noise = int(len(indices) * noise_rate)
        selected = rng.choice(indices, jumlah_noise, replace=False)
        noisy_labels[selected] = target
        noise_mask[selected] = True
    return noisy_labels, noise_mask
