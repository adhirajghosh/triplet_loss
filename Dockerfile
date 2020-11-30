FROM  tensorflow/tensorflow:latest-gpu-py3-jupyter 

MAINTAINER goodBoy

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade pip
RUN pip3 install opencv-python==3.4.2.16
RUN pip3 install opencv-contrib-python==3.4.2.16
RUN pip3 install -U scikit-learn
RUN pip3 install -U Keras
RUN pip3 install glob3
RUN pip3 install pandas
RUN pip3 install Pillow==2.2.2
RUN pip3 install cmake
RUN pip3 install dlib
RUN pip3 install face-recognition
RUN pip3 install matplotlib
RUN pip3 install requests==2.20.0
RUN pip3 install torch
RUN pip3 install torchvision==0.1.8
RUN pip3 install argparse
RUN pip3 install future
RUN pip3 uninstall tensorflow
RUN pip3 install tensorflow-gpu
RUN pip3 install tyepguard
RUN pip3 install -q tensorflow_datasets
RUN pip3 install -q --no-deps tensorflow-addons~=0.7

