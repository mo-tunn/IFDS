# IFDS Model Metrics

Saved from Kaggle training logs shared during development.

## Xception CNN

Selected run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7189646064 |
| AUC | 0.7783824026 |
| F1 | 0.7124324324 |
| Precision | 0.6096207216 |
| Recall | 0.8569570871 |
| Threshold | 0.3700000000 |

Earlier alternate run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7369255151 |
| AUC | 0.7915054677 |
| F1 | 0.7128027682 |
| Precision | 0.6404145078 |
| Recall | 0.8036410923 |

## EfficientNet CNN

Selected run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6946645536 |
| AUC | 0.7638079680 |
| F1 | 0.6932059448 |
| Precision | 0.5856502242 |
| Recall | 0.8491547464 |
| Threshold | 0.2600000000 |

Training details:

| Field | Value |
| --- | --- |
| Backbone | EfficientNetB0 |
| ImageNet pretrained | true |
| Mask-guided crops | true |
| Train mask matches | 3486 |
| Dataset | Kaggle `divg07/casia-20-image-tampering-detection-dataset` |
