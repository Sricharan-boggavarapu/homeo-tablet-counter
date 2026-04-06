#!/bin/bash
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev libgomp1
pip install -r requirements.txt
