# Contains pytorch, torchvision, cuda, cudnn
FROM nvcr.io/nvidia/pytorch:22.03-py3
 
RUN apt-get update && apt-get install -y git

WORKDIR /code

ENV HOME=/code

COPY requirements.txt requirements.txt 

RUN pip3 install -r requirements.txt

RUN git config --global --add safe.directory .

CMD ["tail", "-f", "/dev/null"]
