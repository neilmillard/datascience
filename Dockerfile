FROM tensorflow/tensorflow:1.0.0-rc0-py3

LABEL maintainer "neil@neilmillard.com"

RUN pip --no-cache-dir install \
        pandas

VOLUME /notebooks
