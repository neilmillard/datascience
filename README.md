# Neil's Neural net code
This repo contains code that I'm using to learn about NN

# Vagrant
The attached vagrant file will clone the neilmillard/puppet-dockerhost repo
and build docker with the official tensorflow container https://hub.docker.com/r/tensorflow/tensorflow/

# Docker

There is a dockerfile to build and launch
```
docker build -t "neilmillard/tensorflow:1.0-pandas" .
```
and launch with ports 6006 for tensorboard and 8888 for the notebook server and
set a notebook server password with env variable PASSWORD
-it gives an interactive session also
volumes can be mounted to /notebooks for the notebook server
docker run -it -e "PASSWORD=somepassword" -p 6006:6006 -p 8888:8888 -v /notebooks ./notebooks neilmillard/tensorflow:1.0-pandas

## installing tensorflow

```
# https://www.tensorflow.org/get_started/os_setup#pip_installation
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
# $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
# sudo pip install --upgrade $TF_BINARY_URL
```