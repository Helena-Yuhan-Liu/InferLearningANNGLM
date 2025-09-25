# psylearnNNGLM

This repository implements models to approximate animal learning rules using DNNs or RNNs on the IBL dataset. It follows the setup and data used in [1], along with the code environment from https://github.com/pillowlab/psytrack_learning. 


## Usage 

Step 1: Download data using `DownloadData.ipynb`, which follows the preprocessing steps from https://github.com/pillowlab/psytrack_learning. (For review purposes, the processed data has already been provided in ./Figures/ibl_processed.csv so the user may skip this step.) 

Step 2: Run `main.py` to train on the IBL data used in Ashwood et al. (NeurIPS’20). Use `python3 main.py --glmw_mode=0` for RNNGLM or `python3 main.py --glmw_mode=1` for DNNGLM.    


## References

[1] Zoe Ashwood, Nicholas A Roy, Ji Hyun Bak, and Jonathan W Pillow.
Inferring learning rules from animal decision-making. Advances in Neural
Information Processing Systems, 33:3442–3453, 2020. 

