ARG PYTORCH="1.3"

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install pip==19.3.1

RUN pip3 install torch==1.4.0 torchvision==0.5.0
# Install SOLO
#RUN conda install cython -y && conda clean --all
#RUN git clone https://github.com:WXinlong/SOLO.git /SOLO
ADD ./ /SOLO
WORKDIR /SOLO
#RUN pip install -r requirements/build.txt
#RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
#RUN pip3 install --no-cache-dir -e .
