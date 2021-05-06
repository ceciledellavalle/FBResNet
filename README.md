# FBResNet
Forward Backward ResNet

-------
Il want to thank Marie-Caroline Corbineau
https://github.com/mccorbineau/iRestNet
from which work this code is largely inspired.
-------

### General

* **License**            : GNU General Public License v3.0  
* **Author**             : Cecile Della Valle
* **Institution**        : MAP5, Université de Paris
* **Email**              : cecile.map5@gmail.com 
* **Related publication**: 


### Installation
1. Install miniconda.
2. Create an environment with python version 3.7
   ```
   $ conda create -n FBRestNet_env python=3.7
   ```
3. Inside this environment install the following packages.
   ```
   conda activate FBRestNet_env
   $ conda install pytorch=0.4.0 cuda80 -c pytorch
   $ pip install torchvision==0.2.1 matplotlib==3.0.2 numpy==1.16.0 jupyterlab opencv-python==4.0.0.21 scipy==1.2.0 
   ```
4. Use the example notebook to test and train FBRestNet model.

### Files organization
* `Datasets`   
   * 'Images' : two sets of images from which are extracted 1D signal datasets
   * 'Signals': dataset constructed from images
* `FBRN`    : contains FBRestNet files
    * `main.py`: contains FBResNet class, which is the main class including train and test functions
    * `model.py`: includes the definition of the layers in FBRestNet
    * `myfunc.py`: useful functions used such as convolution and Abel operators
	* 'proxop' : contains he cardan class used to compute the proximity operator of the logarithmic barrier
	    * 'hyperslab.py' : proximity operator for hyperslab constraint
		* 'hyperscube.py' : proximity operator for cubic constraint


### Demo file
`demo.ipynb`: shows how to test and train iRestNet

###### Thank you for using our code, kindly report any suggestion to the corresponding author.
