############Dockerfile###########
FROM openjdk:8

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y git 
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN apt-get install -y tar
RUN apt-get install -y bzip2

RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip

RUN pip3 install Pillow==7.0.0
RUN pip3 install tensorflow==1.14.0
RUN pip3 install keras==2.2.4

RUN python3 -c "from tensorflow.keras.applications.vgg19 import VGG19;model = VGG19(weights='imagenet')"

RUN echo "260515502012"

RUN git clone https://github.com/yanliang12/yan_image_embedding.git
RUN mv yan_image_embedding/* ./
RUN rm -r yan_image_embedding
############Dockerfile###########
