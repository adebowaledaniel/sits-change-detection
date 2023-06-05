## Detecting Land Cover Changes Between Satellite Image Time Series By Exploiting Self-Supervised Representation Learning Capabilities


This repo hosts the code for the paper title above, accepted for publication at the [IGARSS-23](https://2023.ieeeigarss.org/), Pasadena, California

<!-- Link to paper: [here]() -->

## Usage
The supervised SITS classification implementation in `predetect.py` consists of a temporal encoder, a classifier, and post-classification technique for the pre-detection of no-change pixels. The model is trained with labels from one year (2018) and makes predictions on both years. A post-classification technique is used to detect no-change pixels between the predictions.
To use  `predetect.py`, you only need to specify the paths to dataset folder for both years:

```bash
python predetect.py --dataset_folder1 path_to_2018_dataset --dataset_folder2 path_to_2019_dataset
```
 
Once the supervised classification is completed, the best model and the predictions for both years (hard and soft labels) are saved in the output folder; these are input for the contrastive learning module.

To run the contrastive learning `train_BYOL.py`, specify the paths to the datasets for both years and the folder where the model and predictions are saved. Based on the experiment setup, you can choose to explore other evaluation parameters such as `--eval_mode` to either freeze or finetune the learned representation in the downstream task, `--label_mode` to use either the soft pseudo-label, hard pseudo-label or full pixels.
```bash
python train_BYOL.py --dataset_folder1 path_to_2018_dataset --dataset_folder2 path_to_2019_dataset --model_dir path_to_saved_model --eval_mode freeze --label_mode softlabel
```

You can also play around with other model's hyperparameters as specified in the module arguments. 

## Dataset 
A subset of the satellite image time series (SITS) and the annotated label for 2018 and 2019 is available in `data/sits`. These folders contain the SITS data `X`, annotated labels `y` in `.npz` format and additional data for standardization. The dataset can be read using:
```python
with np.load(PATH_TO_FILE) as f:
    X = f['X']
    y = f['y']
``` 

The full dataset can not be distributed here; however, open-source SITS datasets can be found at https://github.com/corentin-dfg/Satellite-Image-Time-Series-Datasets.

## Contributors
[Adebowale Daniel Adebayo](https://adebowaledaniel.com/), [Charlotte Pelletier](https://sites.google.com/site/charpelletier), [Stefan Lang](https://www.plus.ac.at/geoinformatik/department/team/lang/?lang=en) and [Silvia Valero](https://scholar.google.fr/citations?user=8AB1bHkAAAAJ&hl=fr)

## Citation
If you find this code useful, please cite our work as follows:

```bibtex
@inproceedings{adebayoa23,
  title={Detecting Land Cover Changes Between Satellite Image Time Series By Exploiting Self-Supervised Representation Learning Capabilities},
  author={Adebowale Daniel Adebayo and Charlotte Pelletier and Stefan Lang and Silvia Valero},
  booktitle = {International {Geoscience} and {Remote} {Sensing} {Symposium} ({IGARSS})},
  year={2023}
}
```
## Credits
- The Lightweight Temporal Attention Encoder and the classifier are based on the implementations of [Sainte Fare Garnot, Vivien](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch/).
- Credits to https://github.com/lucidrains for Bootstrap Your Own Latent (BYOL) in Pytorch.
- And https://github.com/dl4sits for the Temporal Convolutional Neural Network implementation in Pytorch
