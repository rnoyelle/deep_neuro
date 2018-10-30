# deep_neuro
Continuation of the work of Jannes Schafer  
([Jannes Schafer](https://github.com/schanso/deep_neuro)). 
**deep_neuro** presents multiple ways of exploring the Gray dataset. At its 
core, it features a 
[convolutional neural network](http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf) 
trying to predict either stimulus class or response type from different brain 
regions and at different points in time.
It also transforms the raw data from MATLAB to NumPy-ready and applies the usual
pre-processing steps as well as dealing with the analysis and visualization of the results 


### Installation
To get started, create a project directory using `mkdir my_project/` and change
into it (`cd my_project/`). Then clone this repository using 
`git clone https://github.com/rpaul23/deep_neuro.git`. Once cloned, change into 
the directory (`cd deep_neuro/`) and source the environment generator file:

`. generate_environment.sh`

This will set up your folder structure (see below) and install all required packages. 
Move raw data into `data/raw/` (and/or already pre-processed data into 
`data/pre-processed/`) and you should be ready to go.

```
my_project
|___data
|   |___raw
|
|___scripts
|   |____params
|   |___deep_neuro (git repo)
|       |___lib
|           ...
|
|___results
    |___training
        |___pvals
        |___summary
        |___plots
```

### Train classifier 
The pre-prossesing parameters are set in `param_gen.py`. To pre-process
raw data, just `cd` into the `scripts/deep_neuro/` directory and source the 
`prep_data.sh` file using 

`. prep_data.sh -u <user_name> -s <session> -t <trial_length>`. 

The `trial_length` parameter is optional and set to 500 ms by default. How 
pre-processing using the matnpy module works is that it cuts every trial of a 
given session into five intervals:
* pre-sample (500 ms before stimulus onset)
* sample (500 ms after stimulus onset)
* delay (500 ms after stimulus offset)
* pre-match (500 ms before match onset)
* match (500 ms after match onset)

Example call: `. prep_data.sh -u jannesschaefer -s 141023 -t 500`

### Train classifier
To train a classifier, `cd` into your `scripts/deep_neuro/` directory. Set the processing parameters in `param_gen.py` and source 
the submit file using 

`. submit_training.sh` 

The job will be submitted to the 
cluster and processed once the resources are available.


### Get results
To get results, just `cd` into the `scripts/deep_neuro/` directory and source 
the submit file usiing :
`. get_results.sh` 

This will generate a summary file in 
`results/training/summary/` and a file of pvalues in 
`results/training/pvals/`. 

### Visualize results
To visualize results, run the jupyter notebook `graph_from_pval.py` 

