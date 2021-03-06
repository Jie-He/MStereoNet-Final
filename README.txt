#######################################################################################################################
Conda Environment
## Create a new conda environment
conda create --name myenv python=3.7

## Activate the environment
conda activate myenv

## Install Pytorch (see https://pytorch.org/get-started/locally/ for more info)
## GPU only works with pip version of Pytroch 
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

## Install matplotlib & other tools
## Use pip for those, conda verions has OMP error
pip install matplotlib
pip install scikit-image
pip install opencv-python
pip install pyyaml ## for Generative inpainting network

#######################################################################################################################
Dataset paths
	This submission includes a small sample of training and validation data.
	Path settings can be fouund in "dataset_paths.py"


Running the scripts
	
	CUDA_VISIBLE_DEVICES=1 python main.py --model_name=test_network --fill_mode=opencv --batch_size=3
	
Example training command with GPU device 1, 
save the trained model in models\test_network\ folder, 
use opencv (telea) inpainting for occlusion filling in training data,
use a batch size of 3 per training step

	python benchmark.py --quick_bench=0 --error_log=opencv_result --model_name=opencv

Example benchmarking script.
--quick_bench=1 would only evaluate the latest version of the model, if = 0, evaulate all checkpoints,
if it is not a quick bench, then save the results on a log file called opencv_results
(located in models\logs\),
use the opencv occlusion filling trained model to benchmark.

	python  inference.py --model_name=opencv

Example inference code for example images,
uses the opencv occlusion filling trained model
stereo image should be stored in the following folder
\stereo_images\left\
\stereo_images\right\

Corresponding pair should have the same file name in the two folders
the result of the inference will be saved in 
\stereo_images\out\

#######################################################################################################################
To see the graphs of error metrics, use jupyter notebook to open the notebook file:
	GraphGeneration.ipynb
