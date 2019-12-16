FROM python:3.7.5
MAINTAINER Satyalab, satya-group@lists.andrew.cmu.edu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing \
    && apt-get upgrade -y \
    && apt-get install -y \
    --no-install-recommends \
    apt-utils \
    software-properties-common

RUN apt-get install -y \
    build-essential \
    libopencv-dev \
    python3 \
    python3-dev \
    python3-pip \
    libprotobuf-dev \
    protobuf-compiler \
    cmake

RUN pip3 install --upgrade pip

COPY . /opt/gabriel-lego
WORKDIR /opt/gabriel-lego
RUN pip3 install -Ur requirements.txt

EXPOSE 9099
WORKDIR /opt/gabriel-lego
ENTRYPOINT python ./main.py
