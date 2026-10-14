# FER Challenge V2 - Short Report

## Approach

I built an accuracy-first FER pipeline around an ImageNet-pretrained torchvision ResNet-18. The 48x48 grayscale image is resized to 96x96 and replicated to three channels. AdamW uses separate learning rates for the pretrained backbone and new seven-class head, with cosine decay. Checkpoints are selected by validation categorical accuracy; Macro F1 is tracked for imbalance diagnostics and used only to break an exact accuracy tie.

## Validation and preprocessing

- Audited **28,709** labeled and **7,178** test images; all were readable 48x48 grayscale JPEGs.
- Grouped images by SHA-256 of decoded pixels and created a seed-42, class-stratified holdout of **22,968 train / 5,741 validation** images.
- Verified **zero exact duplicate-hash overlap** between development train and validation.
- Found **1,516** duplicate groups and **39** conflicting-label groups; conflicting groups remained in training and labels were never edited.
- Computed training-partition-only grayscale mean/std (0.507683/0.255111) and compared them with ImageNet normalization for the pretrained backbone.

## Experiments

| Run | Change | Accuracy | Macro F1 | Best epoch |
|---|---|---:|---:|---:|
| `r01_resnet18_ce` | No augmentation; ImageNet normalization; accuracy-first checkpoint selection. | 64.31% | 0.6102 | 8 |
| `r02_resnet18_smoothing` | Single-variable comparison against r01: label smoothing 0.05. | 64.01% | 0.6079 | 8 |
| `r03_resnet18_augment` | Single-variable comparison against r01: conservative augmentation only; smoothing excluded after r02 underperformed. | 65.60% | 0.6201 | 8 |
| `r04_resnet18_graynorm` | Single-variable comparison against r01: training-only grayscale mean/std instead of ImageNet normalization. | 63.14% | 0.6045 | 8 |

Label smoothing did not beat the plain-CE baseline. Conservative flip, affine, brightness, and contrast augmentation improved the single-model result. The final candidate is `ensemble_r01_resnet18_ce__r02_resnet18_smoothing__r03_resnet18_augment__tta`: equal softmax averaging across `r01_resnet18_ce`, `r02_resnet18_smoothing`, `r03_resnet18_augment` with horizontal-flip test-time augmentation. All members use the same untouched grouped holdout, so the ensemble score is comparable rather than an average of incompatible folds.

The optional three-fold training stage was cut after the CPU-only ladder to preserve time for the required notebook PDF, report, and video workflow.

## Final validation result

- Categorical accuracy: **67.86%**
- Macro F1: **0.6440**
- Accuracy gain over the plain baseline: **+3.55 percentage points**
- Major confusion pairs: sad -> neutral (163), fear -> sad (150), neutral -> sad (146)
- Ten-bin ECE: **0.039**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.650 | 0.562 | 0.603 | 799 |
| disgust | 0.895 | 0.391 | 0.544 | 87 |
| fear | 0.566 | 0.470 | 0.514 | 819 |
| happy | 0.831 | 0.882 | 0.856 | 1443 |
| neutral | 0.621 | 0.681 | 0.650 | 993 |
| sad | 0.557 | 0.622 | 0.588 | 966 |
| surprise | 0.754 | 0.754 | 0.754 | 634 |

## Submission and limits

The final CSV contains exactly **7,178** rows in organizer `test.csv` order with only the seven allowed labels. Two complete inference passes matched with maximum probability difference **0.0**. SHA-256: `20a59955fbfa4185f9e9f0b3c90f044b88ba9ae7a8ee9151a7a14a020956ce99`.

This evidence comes from one locked holdout, not repeated nested cross-validation. Exact hashing misses non-identical near-duplicates. FER labels are subjective, and a face crop cannot establish internal emotional state. Hidden-test accuracy remains unknown until organizer evaluation; the model is not suitable for hiring, diagnosis, surveillance, policing, or other high-stakes decisions.

With more time I would repeat grouped cross-validation and calibration analysis. Pseudo-labeling is only a future research idea here and was **not implemented** because the challenge guardrails exclude test-set information from training.
