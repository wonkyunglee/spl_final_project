FROM wonkyunglee/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
MAINTAINER Wonkyung Lee <leewk921223@gmail.com>

ADD ["data.tar.gz", "/dataset/"]
ADD ["configs", "/app/signal_processing_lab/configs"]
ADD ["losses", "/app/signal_processing_lab/losses"]
ADD ["optimizers", "/app/signal_processing_lab/optimizers"]
ADD ["schedulers", "/app/signal_processing_lab/schedulers"]
ADD ["datasets", "/app/signal_processing_lab/datasets"]
ADD ["transforms", "/app/signal_processing_lab/transforms"]
ADD ["models", "/app/signal_processing_lab/models"]
ADD ["utils", "/app/signal_processing_lab/utils"]
ADD ["train.py", "/app/signal_processing_lab/"]
ADD ["train_rapn.py", "/app/signal_processing_lab/"]
ADD [".gitignore", "/app/signal_processing_lab/"]

WORKDIR /app/signal_processing_lab/
RUN mkdir preprocessed

CMD ["zsh"]
