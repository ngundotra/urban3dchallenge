FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

RUN echo "hello!"
# RUN apt-get update --fix-missing && apt-get install -y wget libglib2.0 libsm-dev libxrender-dev libXext-dev libjpeg-dev vim
RUN apt-get update --fix-missing && apt-get install -y wget libglib2.0 libsm-dev libxrender-dev libjpeg-dev vim
ENV PATH "/miniconda/bin:$PATH"
ENV VERSION 4.2.12
RUN wget https://repo.continuum.io/miniconda/Miniconda3-${VERSION}-Linux-x86_64.sh
RUN chmod +x Miniconda3-${VERSION}-Linux-x86_64.sh
RUN ./Miniconda3-${VERSION}-Linux-x86_64.sh -b -f -p /miniconda
RUN conda install GDAL -y
ADD requirements.txt /opt
RUN pip install -r /opt/requirements.txt
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
RUN pip install torchvision
RUN mkdir -p /root/.torch/models
RUN wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -P /root/.torch/models

#ADD weights /results/weights/

ADD ["resnet34_1x1080_retrain.json", "resnet34_4x1080_pretrained.json", "/opt/app/"]
ADD ["train.sh", "test.sh", "test_retrain.sh", "/"]
ADD src /opt/app/src/
