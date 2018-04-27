# Overview [![Docker Pulls](https://img.shields.io/docker/pulls/jamesjue/gabriel-lego.svg)](https://hub.docker.com/r/jamesjue/gabriel-lego/)

This is the world's very first wearable cognitive assistance application!   We chose a deliberately simplified task (assembling 2D lego) since it was our first attempt.  The demo seems easy, but the code to implement it reliably was challenging (especially with flexible user actions and under different lighting conditions).

Please see our [YouTube Demo](https://youtu.be/uy17Hz5xvmY?list=PLmrZVvFtthdP3fwHPy_4d61oDvQY_RBgS)

# Quickstart [![Docker Pulls](https://img.shields.io/docker/pulls/jamesjue/gabriel-lego.svg)](https://hub.docker.com/r/jamesjue/gabriel-lego/)

## Docker 
```bash
docker run -it --rm --name lego -p 0.0.0.0:9098:9098 -p 0.0.0.0:9111:9111 jamesjue/gabriel-lego
```


If you want to build from source, make sure Gabriel is installed and use:

```bash
$ ./gabriel-control -l
$ ./gabriel-ucomm -s 127.0.0.1:8021
$ ./proxy.py -s 127.0.0.1:8021
```
