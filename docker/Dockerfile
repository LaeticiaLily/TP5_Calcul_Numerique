FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
  apt-get install -y gcc make vim libblas-dev liblapack-dev liblapacke-dev libatlas3-base && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ENV HOSTNAME=ambre
RUN rm -rf bin *.o && mkdir -p bin && \
  make all && \
  make run
