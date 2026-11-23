# 06 · Helmet Detection: CNN and VGG-16 Transfer Learning

**Question.** Can an image classifier reliably tell whether a worker is wearing a safety helmet, to support automated safety monitoring?
**Data.** 631 RGB images (200×200), 311 *with helmet* and 320 *without*. Stratified split: 441 train / 95 validation / 95 test. Course-provided; not redistributed.
**Tools.** TensorFlow/Keras, OpenCV, scikit-learn.

## Approach
Four models of increasing capacity:
1. A simple CNN trained from scratch.
2. VGG-16 (ImageNet weights, frozen) with a minimal classification head.
3. VGG-16 with a fully connected head.
4. VGG-16 with a fully connected head plus data augmentation (selected).

## Results (from notebook outputs)

| Model | Val accuracy | Val F1 |
|---|---|---|
| Simple CNN | 0.989 | 0.989 |
| VGG-16 (base) | 1.000 | 1.000 |
| VGG-16 + FFNN | 1.000 | 1.000 |
| VGG-16 + FFNN + augmentation | 1.000 | 1.000 |

Final model on the held-out test set: **accuracy, precision, recall, and F1 all 1.000 (95 images).**

## 2026 re-audit
- **A perfect score on 95 images is a ceiling, not a measurement.** With zero errors in 95 trials, the "rule of three" gives a 95% upper bound on the error rate of about 3/95 ≈ 3.2%. The honest claim is "≥ 96.8% accuracy with 95% confidence on this distribution."
- **I checked for duplicate leakage.** Hashing all 631 image arrays found no exact duplicates, so the result is not explained by identical images across splits. Near-duplicates (same scene, slightly different frame) were not checked.
- **Shortcut learning is the main open risk.** When classes are this separable, a model may key on background, framing, or photo source rather than the helmet. Next steps:
  - Saliency maps (Grad-CAM) on test images.
  - An external test set from different sites and cameras.
  - Stress tests (occlusion, other headwear, low light).
- **Minor:** the augmented training generator used `shuffle=False`, which feeds batches in a fixed order and can bias training. Worth correcting in a re-run.
- **Deployment framing:** for safety monitoring, the costly error is a missed no-helmet case. Report recall on that class separately and choose a threshold accordingly.

## Files
- [`helmet_detection_cnn.ipynb`](helmet_detection_cnn.ipynb): cleaned notebook (course instructions removed; code and outputs unchanged).
