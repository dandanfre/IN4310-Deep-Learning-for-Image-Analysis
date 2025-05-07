#!/bin/bash
#set up environment
source ~/.bash_profile
#clear #!/bin/bash
#set up environment
source ~/.bash_profile
#clear any modules
module purge
#load required modules
module load matplotlib/3.4.3-foss-2021b
module load PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised

#print which python and modules are loaded for debugging
which python
python --version
module list
#Run the Python interpreter with the passed arguments
CUDA_VISIBLE_DEVICES=1 nohup python train.py > out1.log 2> error1.log &
echo $! > train_pid.txt
echo "Training started with PID $(cat train_pid.txt)"python -c "import site; print(site.getsitepackages())"
#Run the Python interpreter with the passed arguments
CUDA_VISIBLE_DEVICES=1 nohup python train.py > out1.log 2> error1.log &
echo $! > train_pid.txt
echo "Training started with PID $(cat train_pid.txt)"