import numpy as np

def generate_symmetric_class_balanced_noise(labels, noise_rate, num_classes, seed=17):
    rng = np.random.default_rng(seed)
    noisy_labels = labels.copy()
    noise_mask = np.zeros(len(labels), dtype=bool)
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        jumlah_noise = int(len(class_indices) * noise_rate)
        selected_indices = rng.choice(class_indices, size=jumlah_noise, replace=False)
        kandidat_label = np.delete(np.arange(num_classes), class_id)
        noisy_labels[selected_indices] = rng.choice(kandidat_label, size=jumlah_noise)
        noise_mask[selected_indices] = True
    return noisy_labels, noise_mask
