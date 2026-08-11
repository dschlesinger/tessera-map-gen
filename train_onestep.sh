#!/bin/bash -l

#$ -P buaisociety       # Specify the SCC project name you want to use
#$ -l h_rt=4:00:00      # Specify the hard time limit for the job
#$ -N train-onestep     # Give job a name
#$ -j y                 # Merge the error and output streams into a single file
#$ -l gpu=1             # Request 1 GPU

.venv/bin/python train_onestep_decoder.py
