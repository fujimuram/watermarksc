# If this is the first time you are running it, do the following in the project root.
# docker build -t <image_name> ./
# docker run -it --gpus all -p <port>:8888 -v ${pwd}:/tf/<folder_name> --name <container_name> <image_name>
# (Windows: ${pwd}, Ubuntu: `pwd`)

# When you use 'start' command to wake up the container, run jupyter with
# source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root

FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

LABEL maintainer="KawaSwitch <bb52120306@ms.nagasaki-u.ac.jp"
LABEL description="Image for the study of DCGAN with tensorflow 2.5.0"
LABEL version="1.0"

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-dev 

RUN pip install opencv-python opencv-contrib-python
