# SSL-Lanes: Self-Supervised Learning for Motion Forecasting in Autonomous Driving
By [Prarthana Bhattacharyya](https://scholar.google.com/citations?user=v6pGkNQAAAAJ&hl=en), [Chengjie Huang](https://scholar.google.com/citations?user=O6gvGZgAAAAJ&hl=en) and [Krzysztof Czarnecki](https://scholar.google.com/citations?hl=en&user=ZzCpumQAAAAJ).

We provide code support and configuration files to reproduce the results in the paper: SSL-Lanes: Self-Supervised Learning for Motion Forecasting in Autonomous Driving.
<br/> Our code is based on [Lane-GCN](https://github.com/uber-research/LaneGCN), which is a clean open-sourced project for motion forecasting methods. 

## Overview

![](assets/methods_overview.png)

In this study, we report the first systematic exploration and assessment of incorporating self-supervision into motion forecasting. We first propose to investigate four novel self-supervised learning tasks for motion forecasting with theoretical rationale and quantitative and qualitative comparisons on the challenging large-scale Argoverse dataset. Secondly, we point out that our auxiliary SSL-based learning setup not only outperforms forecasting methods which use transformers, complicated fusion mechanisms and sophisticated online dense goal candidate optimization algorithms in terms of performance accuracy, but also has low inference time and architectural complexity. Lastly, we conduct several experiments to understand why SSL improves motion forecasting. 

## Results
### Quantitative Results

For this repository, the expected performance on Argoverse 1 validation set is:

| Models | minADE | minFDE | MR |
| :--- | :---: | :---: | :---: |
| Baseline | 0.73 | 1.12 | 11.07 |
| Lane-Masking | 0.70 | 1.02 | 8.82 |
| Distance to Intersection | 0.71 | 1.04 | 8.93 |
| Maneuver Classification | 0.72 | 1.05 | 9.36 |
| Success/Failure Classification | 0.70 | 1.01 | 8.59 |

### Qualitative Results
![](assets/teaser.png)

## Pretrained Models

We provide the pretrained checkpoints for the proposed above-mentioned models in [checkpoints/](https://drive.google.com/drive/folders/1zSznQ0Jzi2fzxLX7xeQpUJppezU7J1v3?usp=sharing). 

## Citation
If you find this project useful in your research, please consider starring the repository and citing:
```bibtex
@misc{bhattacharyya2022ssllanes,
      title={SSL-Lanes: Self-Supervised Learning for Motion Forecasting in Autonomous Driving}, 
      author={Prarthana Bhattacharyya, Chengjie Huang and Krzysztof Czarnecki},
      year={2022},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
* [LaneGCN](https://github.com/uber-research/LaneGCN)
* [SelfTask-GNN](https://github.com/ChandlerBang/SelfTask-GNN)

