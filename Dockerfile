FROM python:3.6-stretch

LABEL maintainer="noahluna@amazon.com"

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# Set the working directory to /app and copy current dir
WORKDIR /opt/ml/

# Copy all of our files into app
COPY . /opt/ml/

# Installing python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run script
CMD ["python", "/opt/ml/src/train.py"]



# You folder structure (that docker will use to make image)
#- app-name
#     |-- src
#          |-- main.py
#          |-- other_module.py
#     |-- requirements.txt
#     |-- Dockerfile

# How Docker container will look like
#- opt
#	|--ml
#   	|-- src
#          |-- main.py
#          |-- other_module.py
#     	|-- requirements.txt
#     	|-- Dockerfile


# /opt/ml/model/ This directory is expected to contain a list of model artifacts created by the training job.
# AWS SageMaker will automatically harvest the files in this folder at the end of the training run, tar them,
# and upload them to S3.
# RUN ["mkdir", "/opt/ml/model/"]

# If the training job fails, AWS SageMaker recommends writing the reason why to this file, however this is completely optional
# Make /opt/ml/output/failure
# RUN ["mkdir", "/opt/ml/output/failure"]
