FROM tensorflow/tensorflow:2.7.0

ENV DEBIAN_FRONTEND=noninteractive

# Install a tool to find a fast, up-to-date Ubuntu Archive Mirror.
RUN apt update && apt install --no-install-recommends -y python3-pip &&\
    pip3 install -U pip &&\
    pip3 install apt-mirror-updater &&\
    apt-mirror-updater -a -q &&\
    rm -rf /var/lib/apt/lists/*

# Install essentials
RUN apt-mirror-updater -a -q &&\
    apt install -y build-essential &&\
    rm -rf /var/lib/apt/lists/*

# OpenCV
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y python3-opencv &&\
    pip3 install -U pip &&\
    pip3 install opencv-python &&\
    pip3 install pyyaml &&\
    rm -rf /var/lib/apt/lists/*

# LabelImg
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y pyqt5-dev-tools &&\
    pip3 install -U pip &&\
    pip3 install labelImg

# Model conversion
RUN pip3 install tf2onnx

RUN ldconfig
