#!/usr/bin/env bash

sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
sudo apt install python3-gdal python-gdal
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
# After this, I just created I new conda env, reinstalled everything, then just did
conda install gdal
