# Label Noise Thesis

Repository eksperimen skripsi Informatika mengenai pengaruh jenis dan tingkat
label noise terhadap efektivitas label cleaning pada deep learning dengan
kompleksitas model yang berbeda.

## Research Design

Primary dataset:
- CIFAR-10

Human-noise validation:
- CIFAR-10N Aggregate
- CIFAR-10N Worst

External confirmation:
- CIFAR-100N Fine

Synthetic noise:
- Symmetric
- Asymmetric Pair-Flip
- Noise rates: 0%, 10%, 20%, 30%, 40%

Models:
- ResNet18
- ResNet34
- ResNet50

Primary cleaner:
- Cleanlab / Confident Learning

Secondary cleaner:
- AUM

Controls:
- No Cleaning
- Matched Random Removal
- Oracle-Matched

Fixed label auditor:
- ResNet18

Primary endpoint:
- Clean-test Accuracy

Compute environment:
- Google Colab

## Important Methodological Rules

- Cleanlab must use out-of-fold predicted probabilities.
- Clean ground-truth labels must not be used by the cleaner.
- Official CIFAR test data must not be used for hyperparameter tuning.
- Models are trained from scratch.
- Paired conditions must use the same noise masks and seeds.
- Primary comparison uses a fixed optimizer-step budget.
- Large experiment artifacts are stored outside this repository.

## Status

Current stage:
Environment and reproducibility setup.
