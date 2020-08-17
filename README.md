## point-cloud-completion-shape-prior

This is the TensorFlow implementation for the paper "Point Cloud Completion by Learning Shape Priors"

## Getting Started
python version: python-3.5;  cuda version: cuda-10;  TensorFlow version: tensorflow-1.13;   Ubuntu version: 16.04

## Compile Customized TF Operators
Please follow [PointNet++](https://github.com/charlesq34/pointnet2) to compile customized TF operators. You may need to change the related library path if necessary.

## Datasets
[Ours](https://drive.google.com/file/d/1V9X6deLtB0jcK7DPJQ1QL5mYe_v1xfRe/view?usp=sharing)   
[PCN's dataset](https://github.com/wentaoyuan/pcn)
    
## Train the model
To train the models: python train.py      

## Evaluate the models
Our pre-trained models can be downloaded here: [Models](https://drive.google.com/file/d/1JIMgKtlWPxP30nb1BnbPKUbY6mO6uaBt/view?usp=sharing), unzip and put them in the root directory.  
To evaluate models: python test.py 
    
## Citation
@inproceedings{wang2020point,  
&nbsp;&nbsp;&nbsp;&nbsp;      author    = {Wang, Xiaogang and Ang Jr, Marcelo H and Lee, Gim Hee},  
&nbsp;&nbsp;&nbsp;&nbsp;      title     = {Point Cloud Completion by Learning Shape Priors},  
&nbsp;&nbsp;&nbsp;&nbsp;      journal   = {arXiv preprint arXiv:2008.00394},  
&nbsp;&nbsp;&nbsp;&nbsp;      year      = {2020}  
}

## Acknowledgements 
Our implementations use the code from the following repository:  
[PCN](https://github.com/wentaoyuan/pcn)        
[MMD-GAN](https://github.com/mbinkowski/MMD-GAN)
