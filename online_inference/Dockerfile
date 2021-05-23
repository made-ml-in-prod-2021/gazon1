FROM python:3.9-slim-buster

# set working directory
RUN mkdir -p /usr/inference_hw2/
WORKDIR /usr/inference_hw2/

# install python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip \
  && pip install -r requirements.txt --no-cache-dir

# copy the current directory
COPY . .