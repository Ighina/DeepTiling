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
## Execute the scripts

The repository contains three main script: *segment.py*, *summarize.py" and *semantic_search.py*.
The scripts are further described below and in the (upcoming) paper.

To run them, just call them with python after having installed specified libraries and using the desired options and arguments. Required arguments for all of the scripts are the location of your data (suggested to have your data in a folder inside the data folder), the window value (-wd) and one of threshold value (-th) or number of segments (-ns) per document if known.

e.g.
```
python segment.py -data path_to_data_to_segment -wd 10 -th 1.5
```

The available options and arguments are further explained by running:
```
python segment.py --help
```
Or by looking into the scripts themselves.

The results are stored in an output directory that can be specified by the user (default is "results").

### Segment.py
This script is the main one and uses the deeptiling algorithm to segment the given texts into topically coherent extracts, stored in the segments folder inside the specified output directory (default is "results").
A description of all the options can be accessed as described above. The main arguments are:
- -data: the folder containing the documents to be segmented (required)
- -wd: the window value required for the algorithm. Default is None and, if not passed, then the programme assumes that the fit.py programme was run on some held out data to find the optimal window value automatically (see below for details on fit.py).
- -th: the threshold value required for the algorithm. Default is None and, if not passed, then the programme assumes that the fit.py programme was run on some held out data to find the optimal window value automatically (see below for details on fit.py).
- -ns: number of segments per document to be segmented. Default is None and if not provided, the programme will default to use a threshold value, assuming that the number of segments is not known. If this argument is used, pass as many integers as the number of documents to be segmented, where each integer corresponds to the number of segments in the corresponding document (in order of appearence).
- -cfg: the location of the configuration file that can be used to run the program without the need of specifying the parameters on the command line (default: parameters.json)
- -enc: the sentence encoder to be used. All sentence encoders from sentence_transformers library plus the DAN encoder from Universal Sentence Encoder models are available. Default is None and, if not passed in the command line, then the programme will look for it as specified in the json configuration file.
- -od: the directory where to store the results. Default is results. If any other name is passed, the programme will create the folder and, inside, will create a segments folder where to store the segmentation results and an embeddings folder where to store the computed sentence embeddings.
- =cat: whether to concatenate all the provided documents in a single document or not. Default is False.
