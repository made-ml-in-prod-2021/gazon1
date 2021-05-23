FROM python:3.9-slim

# set working directory
RUN mkdir -p /usr/inference_hw2/
WORKDIR /usr/inference_hw2/

# install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy the current directory
COPY . .