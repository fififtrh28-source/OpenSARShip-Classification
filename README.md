# OpenSARShip Classification
> Based on https://doi.org/10.1016/j.asr.2021.08.042

## Methods
1. Parsed OpenSARShip dataset and retrieve all PATCH_CAL images.
2. Used CFAR to create bounding boxes and calculate 14 scale-variant features used in training.
3. Resized all images to 64 x 64 x 2 (Stacked VH and VV polarisations).
4. Split dataset to 70 - 20 - 10 (Train - Val - Test) using stratified split to maintain same distribution of classes.
5. Augment dataset using oversampling for fishing classes and undersampling for cargo classes.
6. Standardise images.

## How to Run
### Install
```bash
pip install -r requirements.txt
```
### Preprocess Data
```bash
python data_preparation.py
```
### Train
Set learning parameters on the config object in main.py then run below command.
```bash
python main.py
```