#!/bin/bash

#SBATCH --job-name=Ipln_humor_classification_random_search_1core_64ram_8iter
#SBATCH --ntasks=1
#SBATCH --mem=65536
#SBATCH --time=119:00:00
#SBATCH --tmp=16G
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ignacioferre@gmail.com

source ../../python_env/bin/activate
python3 hyperparameter_search.py
