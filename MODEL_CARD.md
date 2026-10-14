# Model Card - FER V2 Final Candidate

## Summary

- Candidate ID: `ensemble_r01_resnet18_ce__r02_resnet18_smoothing__r03_resnet18_augment__tta`
- Architecture: torchvision ResNet-18 member models with a dropout + 7-class linear head
- Combination: equal mean of aligned softmax probabilities after per-member horizontal-flip TTA
- Members: `r01_resnet18_ce`, `r02_resnet18_smoothing`, `r03_resnet18_augment`
- Input: 48x48 grayscale face crop, resized to 96x96 and replicated to 3 channels
- Output labels: angry, disgust, fear, happy, neutral, sad, surprise

## Intended use

Hackathon benchmarking, technical evaluation, education, and reproducible FER experimentation.

## Out-of-scope use

Do not use for hiring, medical or psychological diagnosis, surveillance, policing, access control, student assessment, or claims about a person's true internal state.

## Evaluation

- Locked grouped-holdout accuracy: **67.86%**
- Macro F1: **0.6440**
- Holdout rows: **5,741**
- Split hash: `43616ca5590b0eadc3b973ba002062dc49d10f5e2a83691e37270f5ed689d61a`
- Selection: accuracy descending, then Macro F1 descending

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.650 | 0.562 | 0.603 | 799 |
| disgust | 0.895 | 0.391 | 0.544 | 87 |
| fear | 0.566 | 0.470 | 0.514 | 819 |
| happy | 0.831 | 0.882 | 0.856 | 1443 |
| neutral | 0.621 | 0.681 | 0.650 | 993 |
| sad | 0.557 | 0.622 | 0.588 | 966 |
| surprise | 0.754 | 0.754 | 0.754 | 634 |

## Training and data

The source data contains 28,709 labeled FER images. Exact decoded-pixel duplicate clusters are kept within a single side of the split; 39 conflicting-label groups were excluded from validation. The hidden test set was not used for training, checkpoint selection, label lookup, or pseudo-labeling. Every member used seed 42 and the same split hash.

## Limitations

One holdout can underestimate uncertainty. Exact hashes do not cover all visually similar images. The smallest class has limited support. Dataset labels and apparent expressions can be ambiguous or culturally contextual. Grad-CAM is qualitative rather than causal. Probability calibration is imperfect (10-bin ECE 0.039). Hidden-test performance is unknown.

## Provenance

- Final CSV: `outputs/submissions/ensemble_r01_resnet18_ce__r02_resnet18_smoothing__r03_resnet18_augment__tta.csv`
- CSV SHA-256: `20a59955fbfa4185f9e9f0b3c90f044b88ba9ae7a8ee9151a7a14a020956ce99`
- Exact test ID order: verified
- Repeat inference maximum absolute difference: `0.0`
