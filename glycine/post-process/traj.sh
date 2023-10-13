#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --job-name="cat"
#SBATCH --mem-per-cpu 10G

module load openmpi/3.1.5/gcc/8.4.0/zen

export OMP_NUM_THREADS=1

source /scratch/zt1/project/tiwary-prj/user/zzou/anaconda3/etc/profile.d/conda.sh

conda activate patchy-mi

python create_traj.py

date
