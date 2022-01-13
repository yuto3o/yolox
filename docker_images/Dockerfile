FROM tensorflow/tensorflow:2.7.0

ENV DEBIAN_FRONTEND=noninteractive

# Install essentials
RUN apt update &&\
    apt install -y build-essential &&\
    rm -rf /var/lib/apt/lists/*

# OpenCV
RUN apt update &&\
    apt install --no-install-recommends -y python3-opencv &&\
    pip3 install -U pip &&\
    pip3 install opencv-python &&\
    pip3 install pyyaml &&\
    rm -rf /var/lib/apt/lists/*

RUN ldconfig