# Overview [![Docker Image Status][docker-image]][docker] [![License][license-image]][license]

This is the world's very first wearable cognitive assistance application!   We chose a deliberately simplified task (assembling 2D lego) since it was our first attempt.  The demo seems easy, but the code to implement it reliably was challenging (especially with flexible user actions and under different lighting conditions).

[![Demo Video](https://img.youtube.com/vi/7L9U-n29abg/0.jpg)](https://www.youtube.com/watch?v=7L9U-n29abg)

[docker-image]: https://img.shields.io/docker/build/cmusatyalab/gabriel-lego.svg
[docker]: https://hub.docker.com/r/cmusatyalab/gabriel-lego

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

# Dependency Requirement

* OpenCV: 2.4.9.1 (not working with more recent version)
* numpy: 1.11.1

# Installation
## Client
An Android client is available on the Google PlayStore 

<a href='https://play.google.com/store/apps/details?id=edu.cmu.cs.gabrielclient'><img height='125px' width='323px' alt='Get it on Google Play' src='https://play.google.com/intl/en_us/badges/images/generic/en_badge_web_generic.png'/></a>

Google Play and the Google Play logo are trademarks of Google LLC.

## Server
Running the server application using Docker is advised. If you want to install from source, please see [Dockerfile](Dockerfile) for details.

## Lego Set

We used the [lego set](https://www.amazon.com/LEGO-6000207-Life-Of-George/dp/B005UFAG1S) when building this application. Any standard lego bricks would work. However, these bricks need to be placed on this particular [lego board](lego-board.pdf). Print the board on a piece of paper would also work.


# How to Run
## Client
From the main activity one can add servers by name and IP/domain. Subtitles for audio feedback can also been toggled. This option is useful for devices that may not have integrated speakers(like ODG R-7).
Pressing the 'Play' button next to a server will initiate a connection to the Gabriel server at that address.

## Server
### Container
```bash
docker run --rm -it --name lego \
-p 0.0.0.0:9098:9098 -p 0.0.0.0:9111:9111 -p 0.0.0.0:22222:22222 \
-p 0.0.0.0:8080:8080 \
cmusatyalab/gabriel-lego:latest
```
