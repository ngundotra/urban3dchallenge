#!/usr/bin/env bash

PYTHONPATH=/opt/app/src python /opt/app/src/test.py /opt/app/resnet34_4x1080_pretrained.json $1 $2 $3
