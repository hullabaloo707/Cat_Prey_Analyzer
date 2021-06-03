#!/usr/bin/env bash

echo "Executing Cat_Prey_Analyzer"
# Tensorflow Stuff
export PYTHONPATH=$PYTHONPATH:/home/s0000233/workspace/private/tf_models/models/research:/home/s0000233/workspace/private/tf_models/models/research/slim:/home/pi/.local/lib/python3.7/site-packages:/home/s0000233/workspace/private/Cat_Prey_Analyzer
cd /home/pi/Cat_Prey_Analyzer
python3 cascade.py
