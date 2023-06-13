FROM python:3.10

RUN mkdir -p /usr/src/app/tensorboard
COPY requirements.txt /usr/src/app/tensorboard/
#COPY ../GeneticDeployerServer /usr/src/app/
WORKDIR /usr/src/app/tensorboard

RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
WORKDIR /usr/src/app

EXPOSE 6006
ENTRYPOINT ["tensorboard"]
CMD ["--logdir", "tensorboard_logs", "--host", "0.0.0.0"]