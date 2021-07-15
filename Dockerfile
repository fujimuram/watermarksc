FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

LABEL maintainer="KawaSwitch <bb52120306@ms.nagasaki-u.ac.jp"
LABEL description="Image for the study of DCGAN with tensorflow 2.5.0"
LABEL version="1.0"

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-dev 

RUN pip install opencv-python opencv-contrib-python

# WORKDIR /app
# ADD ./requirements.txt /app

# pip install -r
# opencv-python opencv-contrib-python

#CMD [ "executable" ]


# Run with
# docker run -it --gpus all -p <port>:8888 -v ${pwd}:/tf/<folder_name> --name tensor_origin tensorflow/tensorflow:2.5.0-gpu-jupyter