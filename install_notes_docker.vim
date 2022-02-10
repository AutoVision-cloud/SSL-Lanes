User notes for how to install docker for Lane-GCN and how to install horovod:

1. docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
2. docker run --gpus '"device=4,5,6,7"' -it --name  SSL-Lanes -v ~/../../scratch/:/pt -w /pt -p 0.0.1.0:8818:8828 pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
3. Check install notes from install_notes.vim
4. pip install -e 'path to argoverse_api'
5. pip install scikit-image IPython tqdm ipdb
6. HOROVOD_CUDA_HOME='../usr/local/cuda-10.2' HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.5
7. add 2> /dev/null to the end of your command. it will disable all stderr messages.

-----------------------------------------------------------------------------------------
Helpful debugging:
- https://github.com/horovod/horovod/issues/910
- https://hub.docker.com/r/pytorch/pytorch/tags?page=1&name=devel
- https://github.com/horovod/horovod/issues/1029
