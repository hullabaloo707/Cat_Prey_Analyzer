FROM tensorflow/tensorflow:latest-gpu

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install libgconf-2-4 libatk1.0-0 libatk-bridge2.0-0 libgdk-pixbuf2.0-0 libgtk-3-0 libgbm-dev libnss3-dev libxss-dev
RUN apt-get -y install libglu1
RUN apt-get -y install --allow-change-held-packages --allow-downgrades libcudnn8=8.1.0.77-1+cuda11.2