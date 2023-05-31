#!/bin/bash

#SBATCH --nodes=6
#SBATCH --job-name=GWtuna
#SBATCH -p sciama2.q
#SBATCH --ntasks=16
#SBATCH --time=72:00:00
#SBATCH --output=output_logfile%j
#SBATCH --error=error_logfile%j
#SBATCH --mail-user=susanna.green@port.ac.uk
#SBATCH --mail-type=FAIL

module purge
module load system
module load anaconda3/2022.10

source activate /mnt/lustre/shared_conda/envs/sgreen/PyCBCandOptuna


echo `conda info`
echo `which python`

cd /users/sgreen/gwtuna/LVK/Paper/NSSearch
python GWtunaHanfordGraphs.py
