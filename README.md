# TriCondNet
Triple Conductivity Network (TriCondNet) implementation and benchmarking code accompanying "Physics-guided modelling of temperature-dependent electrical conductivity of functional oxides." The full model framework, nested cross-validation protocol, and hyperparameterisation workflow for each network are provided.


## Installation
Python 3.11 or newer is required for the TriCondNet workflow. Clone the repository and install dependencies:
```bash
git clone https://github.com/lrcfmd/TriCondNet
cd TriCondNet
pip install -r requirements.txt
```
The MODNet comparison (TriMODNet) uses the original MODNet implementation and needs a separate Python 3.9 environment with `modnet` and `tensorflow` installed.

## Repository structure
- `models/TriCondNet.py` - TriCondNet networks: the metallic and semiconducting conductivity regressors (`MetalPINN`, `SemiconductorPINN`).
- `models/deployment_tricondnet.py` - wrappers for training and ensembling the final deployment models (including the LightGBM classifier).
- `models/TriMODNet_Implementation.py` - MODNet classifier and regressor used for the TriMODNet comparison.
- `evaluation/` - includes the grouped (by formula) nested cross validation pipelines, plus preprocessing and feature-selection utilities. 
- `hyperparameterisation/` - Bayesian optimisation file for TriCondNet and TriMODNet parameters. 
- `nestedcv_demonstration/` - Numbered notebooks for generating features from input training data, and the comparison of TriCondNet and TriMODNet performance via a nested cross validation protocol.
- `deployment_demonstration/` - Deployment demonstration of TriCondNet. This includes numbered notebooks to extract hyperparameters from nested cross validation, retraining TriCondNet on the full 100% of the training data, and demonstrating the model in inference mode with a set of example formulae and temperatures.

## Usage
See the notebooks in `nestedcv_demonstration/` for a minimal nested cross-validation example, and `deployment_demonstration/` for training and using the final ensemble.


## Data
A sample labelled dataset is provided with this work, originally retrieved from https://zenodo.org/records/15365344 (Leng Tang & Taylor Sparks>, Creative Commons Attribution 4.0 International). It was filtered for functional oxides and labelled as "Metal"/"Semiconductor".


## Citation
This work is currently under review. Citation details will be added upon publication.

## License
MIT - see LICENSE file.
