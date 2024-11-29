FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

EXPOSE 17799

WORKDIR /app

RUN apt update && apt install git -y
RUN apt update \
    && apt install python3.8 python3.8-dev -y \
    && ln -s /usr/bin/python3.8 /usr/bin/python 
RUN apt install wget python3-distutils -y \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && python -m pip install --upgrade pip

RUN git clone https://github.com/PD-Mera/Real-ESRGAN 
WORKDIR /app/Real-ESRGAN 
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
# RUN git clone https://github.com/XPixelGroup/BasicSR
# RUN pip install ./BasicSR
RUN pip install basicsr
RUN pip install numpy opencv-python Pillow tqdm
RUN pip install .
RUN pip install -r requirements_additional.txt
RUN mkdir -p weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights

RUN pip uninstall opencv-python -y
RUN pip install opencv-python-headless

ENTRYPOINT ["python", "api_realesrgan.py"]