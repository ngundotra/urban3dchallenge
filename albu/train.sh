#!/usr/bin/env bash

PYTHONPATH=/opt/app/src python /opt/app/src/train.py /opt/app/resnet34_1x1080_retrain.json $1
