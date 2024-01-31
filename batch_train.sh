#!/bin/bash

nohup python3 train.py --mode=BiLSTM
nohup python3 train.py --mode=Transformer
nohup python3 train.py --mode=XceptionTime --config_name=paper/geofeat
