FROM pytorch/pytorch:0.4_cuda9_cudnn7
RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow tensorboardX inflect==0.2.5 Unidecode==1.0.22 jupyter

RUN apt-get update
RUN apt-get install wget
RUN wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

RUN apt-get install git
RUN git clone https://github.com/NVIDIA/tacotron2.git

RUN tar -jxvf LJSpeech-1.1.tar.bz2
WORKDIR tacotron2
RUN sed -i -- 's,DUMMY,../LJSpeech-1.1/wavs,g' filelists/*.txt

RUN apt-get install -y vim-tiny

# manually build tensorflow
RUN pip uninstall tensorflow
RUN git clone https://github.com/tensorflow/tensorflow
WORKDIR tensorflow
RUN git checkout v1.9.0
 