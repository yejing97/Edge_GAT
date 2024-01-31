#!/bin/bash

python3 train.py --mode=BiLSTM
python3 train.py --mode=Transformer
python3 train.py --mode=XceptionTime --config_name=paper/geofeat
