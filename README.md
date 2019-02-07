# RadiusDirectionPosteriors


## Set up

** Virtual Environment Without conda **
```
git clone https://github.com/ChangYong-Oh/RadiusDirectionPosteriors.git
cd RadiusDirectionPosteriors
source setup_pip.sh
```

** Virtual Environment With conda **
```
conda create -n RadiusDirectionPosteriors python=2.7.14 anaconda --yes
cd "`which python | xargs dirname | xargs dirname`/envs/RadiusDirectionPosteriors"
git clone https://github.com/ChangYong-Oh/RadiusDirectionPosteriors.git
source RadiusDirectionPosteriors/setup_conda.sh
Default python should be the anaconda python.
```

Different python version is possible. For avaialbe version search
```
conda search "^python$"
```

** Import in existing Python environment **

Or to be able to import this code in an existing Python environment, go:
```
pip install -e git+https://github.com/ChangYong-Oh/RadiusDirectionPosteriors.git#egg=RadiusDirectionPosteriors
```


## Directories
Data directory and experiment result directory can be set in 
```
RadiusDirectionPosteriors/utils/dir_utils.py
```

## UCI regression Training
All below files can be called with arguments and information about arguments can be retrieved by calling those file with **--help** option.
```
python RadiusDirectionPosteriors/train_double_uci.py --help
```
For dataset splitting, it is recommended to follow splitting given in
```
https://github.com/yaringal/DropoutUncertaintyExps
```


## Compression Training
You can train model with compression with file 
```
RadiusDirectionPosteriors/main.py
```
Arguments
* model_type : LeNetFC, LeNet5-Bundle, LeNet5-Flatten
* prior_file : json file contraining prior information, when only a filename is given without an absolute path, it looks for a json file in **prior_json** directory.
* epochs : number of epochs
* batch_size : default 100
* lr : default 0.001
* gpu : when this is given, then gpu is used
* model_file : only needed when you want to continue training initialized with setting in model_file
* eval : Using setting in model_file, model is evaluated.

Example
```
python RadiusDirectionPosteriors/main.py --model_type LeNet5 --prior_file HalfCauchy-fc0.01-conv0.01.json --epochs 200
```
For LeNetFC, LeNet5-Bundle, LeNet5-Flatten, cpu is faster.

## Compression plotting
In each of following files, you can plot logarithm of mode of radius posteriors. Threshold and model_file can be specified in files.
```
RadiusDirectionPosteriors/compress_lenet_fc_double.py
RadiusDirectionPosteriors/compress_lenet_conv_double_bundle.py
RadiusDirectionPosteriors/compress_lenet_conv_double_flatten.py
```
