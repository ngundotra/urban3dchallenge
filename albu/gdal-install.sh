#!/usr/bin/env bash

sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
sudo apt install python3-gdal python-gdal
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
# Here's what worked for me on Ubuntu 16.04
sudo python -m pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal -L/usr/local/lib -lgdal"

