Welcome to our module!
# Aja-pose

AJA-pose helps you train, validate and test your animal pose estimation model.
Check out how we have done it in Google Colab.
We have evaluated our model (PCK@0.05) and the mean accuracy for the 23 keypoints for our model is 93.073%<br>
We recommend using at least Nvidia V100 GPU for faster inferencing but T4 GPUs will still work.  
<br><br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N3v7Y-PN9uvw5V5PUbAYqh9vGkLfm_Km?usp=sharing)
## Getting Started

```bash
git clone https://github.com/Antony-gitau/AJA-pose.git
cd AJA-pose
pip install -e .
```
Getting our model and the dataset.
We have made our model public and can be downloaded here
```python
import urllib.request

# # Get the dataset
url = "https://storage.googleapis.com/figures-gp/animal-kingdom/dataset.zip"
destination = "dataset.zip"

urllib.request.urlretrieve(url, destination)

# Unzip the file
!unzip dataset.zip

# The model file
url = "https://storage.googleapis.com/figures-gp/animal-kingdom/all_animals_no_pretrain_106.pth"
destination = "all_animals_no_pretrain_60.pth"

urllib.request.urlretrieve(url, destination)
```
Test our model<br>
We require the path to the test images directory and the test.json file in MPII format
```python
from aja_pose import Model

# path to the images directory and annotation in mpii json format
images_directory = '' # Path to the images directory
mpii_json = '' # Path to the test.json file
model_file = 'all_animals_no_pretrain_60.pth' # Path to the model file 

# Initialize the class
model = Model()
# Test the model on Protocol 1
model.test(images_directory, mpii_json, model_file)
```
You can also start to train your model or pretrain on top of ours
```python
# train a VHR model
train_json = '/content/AJA-pose/code/annotation/ak_P2/train.json' # labels for the train set
valid_json = '/content/AJA-pose/code/annotation/ak_P2/test.json' # Labels for the validation set
model_file = '' # A pytorch model file to pretrain on.
model.train(images_directory, train_json, valid_json, pretrained=model_file)
```

