1. apt-get update && apt-get install -y git
2. apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

3. docker run --ipc=host --gpus '"device=0,1,2,3"' -it --name ssl -v /home/p6bhatta:/pt -w /pt -p 127.127.0.0:8851:1821 pytorch/pytorch:latest

4.  apt-get update  &&  apt install libopenmpi-dev  ##for mpi4py

5. apt-get update ##[edited]
apt-get install ffmpeg libsm6 libxext6  -y

