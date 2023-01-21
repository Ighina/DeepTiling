# DeepTiling
A TextTiling-based algorithm for text segmentation (aka topic segmentation) that uses neural sentence encoders, as well as extractive summarization and semantic search applications built on top of it. The full description of the algorithm can be found in the accompanying paper [Unsupervised Text Segmentation via Deep Sentence Encoders: a first step towards a common framework for text-based segmentation, summarization and indexing of media content.](https://zenodo.org/record/4744399#.YJfcwLVKjIU)

# Running the Program
## Setting up the workspace and download required libraries
Move your working directory into this repository after having cloned or otherwise downloaded it.

To download the required libraries, either pip can be used directly to install them on the python installation of the current environment, or a virtual environment can be created with conda. The current installation assumes no GPU available by default: to install pytorch with GPU, refer to [this page](https://pytorch.org/get-started/locally/). Also, if using GPU and you want to use Universal Sentence Encoder, than you will need to install tensorflow-gpu by running the following in the environment created as explained below:
```
pip install tensorflow-gpu
```

In both cases, first open your terminal and do one of the following.

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

If you run into errors because conda can't install certain libraries, then, install the libraries directly in the virtual environment by running the following block of codes:

```
conda create -n deeptiling
source activate deeptiling
pip install -r requirements.txt
```

There might be problems to install tensorflow if your python version is greater than 3.8, in that case run the following block of codes:
```
conda create -n deeptiling
source activate deeptiling
conda install python=3.7
pip install -r requirements.txt
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

### segment.py
This script is the main one and uses the deeptiling algorithm to segment the given texts into topically coherent extracts, stored in the segments folder inside the specified output directory (default is "results").
A description of all the options can be accessed as described above. The main arguments are:
- -data: the folder containing the documents to be segmented (required)
- -wd: the window value required for the algorithm. Default is None and, if not passed, then the programme assumes that the fit.py programme was run on some held out data to find the optimal window value automatically (see below for details on fit.py).
- -th: the threshold value required for the algorithm. Default is None and, if not passed, then the programme assumes that the fit.py programme was run on some held out data to find the optimal window value automatically (see below for details on fit.py).
- -ns: number of segments per document to be segmented. Default is None and if not provided, the programme will default to use a threshold value, assuming that the number of segments is not known. If this argument is used, pass as many integers as the number of documents to be segmented, where each integer corresponds to the number of segments in the corresponding document (in order of appearence).
- -cfg: the location of the configuration file that can be used to run the program without the need of specifying the parameters on the command line (default: parameters.json)
- -enc: the sentence encoder to be used. All sentence encoders from sentence_transformers library plus the DAN encoder from Universal Sentence Encoder (USE) models are available. Default is None and, if not passed in the command line, then the programme will look for it as specified in the json configuration file. In general, the names to be passed must correspond to the ones of the pretrained models from the official [sentence_transformers documentation](https://www.sbert.net/docs/pretrained_models.html) or, alternatively, pass "USE" to choose the DAN-based Universal sentence encoders released by Google on [tensorflow_hub](https://tfhub.dev/google/universal-sentence-encoder/4)
- -od: the directory where to store the results. Default is results. If any other name is passed, the programme will create the folder and, inside, will create a segments folder where to store the segmentation results and an embeddings folder where to store the computed sentence embeddings.
- -cat: whether to concatenate all the provided documents in a single document or not. Default is False.

#### Input Structure
The argument to the -data parameter above should be the path to a folder containing the documents to be segmented. The documents should be text documents (no specific extension required, but .txt files would do just fine).
For example, if you have a collection of documents in "folder1":
```
python segment.py -data folder1 -wd 10 -th 1.5
```
With folder1:
- folder1:
  - textfile1.txt
  - textfile2.txt
  - textfile3.txt
  etc.

### summarize.py
This script extract the most important n sentences from each segment, so as to give a summary of the segment itself. The algorithm uses LexRank: please look at the relative github page quoted in the LexRank script for further details on that. If the input documents haven't been segmented yet when this script is called, the segmentation process defined in segment.py will run automatically. Therefore, all arguments are the same as segment.py but for the argument -nt specifying the number of sentences to extract (default is 1).

### semantic_search.py
This script is the same as summarize.py, but instead of returning the top sentences per segment it averages them and compare to a query input by the user (a prompt asks for it when the script is called), so as to return the segment that is the most similar to the query. In this case then the -nt parameter specifies how many sentences are averaged to represent each segment, whereas a very large number will result in all the sentences be averaged and, usually, better performance (but slightly longer runtime).

### fit.py
The repository enables also the automatic search of best window and threshold values among a set of candidates, given a held-out corpus. For the parameters so found to be useful, remember that the data at inference time should be similar (e.g. from the same domain) as the ones used to find the best parameters.

By calling the fit.py script, the automatic search for the best parameters is performed, based on the parameters.json file that specifies, among the other things, the sentence encoder to be used and the set of candidates for the window value and for the threshold value. You can change the parameters.json file according to your needs or otherwise point the script to an alternative json file having same structure by passing its location to the -hyper argument of the script. As a compulsory argument the script requires the training data to be provided after the argument --training_data. 
e.g.
```
python fit.py --training_data location_of_the_data_used_to_find_optimal_parameters
```

N.B. Please note that the script for segmentation will need to download the sentence encoders the first time in which it is run with a specific encoder and this might take a while. Subsequent uses with the same encoder, instead, should run faster. 

# References
The current repository is modelled after [https://github.com/koomri/text-segmentation] and so are the scripts for manipulating the data.
TopicTiling and TextTiling implementations are made after the originale papers, as referenced in the accompanying paper.
If I am missing any reference or for any problem with the repository, please feel free to contact me.
