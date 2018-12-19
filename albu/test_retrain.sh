#!/usr/bin/env bash

PYTHONPATH=/opt/app/src python /opt/app/src/test.py /opt/app/resnet34_1x1080_retrain.json $1 $2 $3
