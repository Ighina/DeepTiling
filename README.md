# DeepTiling
A TextTiling-based algorithm for text segmentation (aka topic segmentation) that uses neural sentence encoders, as well as extractive summarization and semantic search applications built on top of it.

# Running the Program
## Setting up the workspace and download required libraries

Download this repository, unzip it and move the working directory into it.

To download the required libraries, either pip can be used directly to install them on the python installation of the current environment, or a virtual environment can be created with conda. The current installation assumes no GPU available by default: to install pytorch with GPU, refer to [this page](https://pytorch.org/get-started/locally/).

In both cases, open your command prompt (anaconda prompt is preferred) and do one of the following.

Installing required libraries with pip:
```
pip install -r requirements.txt
```

Creating a virtual-environment where to install required libraries with conda:
```
conda create -n deeptiling --file requirements.txt
```

After which the environment can be activated with:
```
conda activate deeptiling
```
And deactivated with:
```
conda deactivate deeptiling
```
