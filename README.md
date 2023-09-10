
# Setup

**sg_rlbench**
We used customized rlbench envrionments for the SKillGrounding experiments.
```
pip install -r requirements.txt
pip install -e .
``` 

**jax dependencies**

need to match cuda, cudnn version when installing jax,flax. I used:

```
pip install flax==0.7.0 
pip install jax==0.4.6 
# pip install jaxlib==0.4.6+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jaxlib==0.4.6+cuda11.cudnn82 

```


**other packages**

 jax-resnet

```
pip install --upgrade git+https://github.com/n2cholas/jax-resnet.git
```
Pyrep

follow [Pyrep official github](https://github.com/stepjam/PyRep) to install.

# Usage

**headless mode for runing sg_rlbench**

note: you need to install opencv-python-headless for headless mode
```
python jaxbc/utils/startx.py
export DISPLAY=:0.0                                    
nohup sudo X & 
```
**other commands**

examples are in the scripts folder. 

```
# data collection
python data_collection.py

# train
python train.py --task 'rlbench-pick_and_lift_simple' --policy 'bc' 

# evaluation
python evaluate.py --mode 'pick_and_lift_simple_bc' --load_path '...'
```