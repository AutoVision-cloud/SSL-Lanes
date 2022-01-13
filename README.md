# Game-Theoretic Interface

## How to download the data and where to place it?
Download the motion forecasting data from https://www.argoverse.org/data.html.

Place it in `SSL-Lanes/dataset/train` and `SSL-Lanes/dataset/val`.

## How to setup Argoverse-api and Lane-GCN?
Install argoverse package and its dependencies from https://github.com/argoai/argoverse-api.

Install dependencies for Lane-GCN from https://github.com/uber-research/LaneGCN.

Check the scripts, and download the processed Lane-GCN data instead of running it for hours.
Place it in `SSL-Lanes/dataset/preprocess`.

Download the checkpoint from the link given in *testing* section. Place it in `SSL-Lanes/LaneGCN/results/lanegcn/ckpt`.

## Which links to change?
Change the `root_dir`, `weight` and `sys-dir` links to your local links.

## What is the purpose of the game-theoretic interface notebook?
Motivation: Use a game-theoretic planner to check feasibility/lack of other possible trajectories.
1. Provide past 2 sec history of every vehicle
2. Provide 3 sec predictions of every vehicle
3. Provide lane centerline of every vehicle
4. Provide say a 1xn matrix of vehicle of interest and its intersections in the 3 sec future with other vehicles
5. Along path distance for vehicles whose futures intersect
