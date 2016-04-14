# Base image
FROM ubuntu:14.04

# Install base packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopenblas-dev \
    python-dev \
    python-pip \
    python-nose \
    python-numpy \
    python-scipy \
 && rm -rf /var/lib/apt/lists/*

# app directory
COPY . /app
WORKDIR /app

# Install scikit-learn from github
RUN pip install cython
RUN pip install git+https://github.com/scikit-learn/scikit-learn.git@0.17.X

# Install packages from requirements.txt with pip
RUN pip install -r /app/requirements.txt

# Installing smart_alerts_intelligence_utils
# RUN git clone git@github.com:encorealerts/smart_alerts_intelligence_utils.git
RUN git clone https://95478f431e018aff65923a13ec0f46b868557d50@github.com/encorealerts/smart_alerts_intelligence_utils.git

# smart_alerts_intelligence_utils setup 
RUN cd /app/smart_alerts_intelligence_utils && python setup.py install && cd /app

# Check smart_alerts_intelligence_utils setup 
RUN python -c "from meltwater_smart_alerts.ml.pipeline import *"

# Expose port 5001 for external access
EXPOSE 5001

# Final command for running the application 
CMD ["python", "/app/application.py"]
