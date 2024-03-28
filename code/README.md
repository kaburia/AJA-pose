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
model.test(images_directory, protocol='P1', model=model_file)
# Test the model on Protocol 2
model.test(images_directory, protocol='P2', model=model_file)
# Test the model on birds class Protocol 3
model.test(images_directory, protocol='P3', model=model_file, animal_class='bird')
# Test the model on reptiles class Protocol 3
model.test(images_directory, protocol='P3', model=model_file, animal_class='reptile')
# Test the model on mammals class Protocol 3
model.test(images_directory, protocol='P3', model=model_file, animal_class='mammal')
# Test the model on fish class Protocol 3
model.test(images_directory, protocol='P3', model=model_file, animal_class='fish')
# Test the model on amphibian class Protocol 3
model.test(images_directory, protocol='P3', model=model_file, animal_class='amphibian')
```
You can also start to train your model or pretrain on top of ours
```python
# train a VHR model
train_json = '' # labels for the train set (train.json)
valid_json = '' # Labels for the validation set (test.json)
model_file = '' # A pytorch model file to pretrain on.
model.train(images_directory, train_json, valid_json, pretrained=model_file)

# Train a model on a particular class e.g (Ampibian)
model.train(images_directory, protocol='P3', animal_class='amphibian', model=model_file)
```

## Results
A sanity check on our model.

![image3](https://github.com/Antony-gitau/AJA-pose/assets/88529649/266e526c-48aa-4401-b411-5f161a734c83)
![image6](https://github.com/Antony-gitau/AJA-pose/assets/88529649/615f5498-1be9-4235-8df2-11e46bfb1384)
<br>
Ground Truth
![image](https://github.com/Antony-gitau/AJA-pose/assets/88529649/c7b8275d-04a5-420a-b8c5-1da70eaf6d9f)
<br>
Predictions
![image](https://github.com/Antony-gitau/AJA-pose/assets/88529649/efe360f8-3d5f-44b4-a396-a364096a2b4d)

## Performance
The performance of our model on the different animal classes is as shown below.

| Animal Class | Samples |  | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mouth | Tail | Mean |
|------|---------|--|------|----------|-------|-------|-----|------|-------|-------|------|------|
| Birds | 1705 |  | 95.756 | 93.637 | 89.774 | 88.179 | 98.975 | 97.582 | 94.326 | 98.447 | 95.112 | 95.164 |
| Reptiles | 1209 |  | 91.538 | 85.291 | 84.662 | 85.587 | 90.457 | 88.097 | 85.239 | 96.723 | 83.925 | 89.553 |
| Mammals | 1496 |  | 90.641 | 89.269 | 88.509 | 89.927 | 90.263 | 88.655 | 89.535 | 93.622 | 82.161 | 90.038 |
| Fish | 918  |  | 96.468 | 96.249 | 98.643 | 96.058 | 98.403 | 96.743 | 95.775 | 97.564 | 98.256 | 96.467 |
| Amphibian | 1279 |  | 98.128 | 94.342 | 97.948 | 98.508 | 95.491 | 94.957 | 94.319 | 98.702 | 99.568 | 95.493 |

The model performance on Protocol 1 and Protocol 2 is as shown below.

| Protocol | Samples |  | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mouth | Tail | Mean |
|----------|---------|--|------|----------|-------|-------|-----|------|-------|-------|------|------|
| P1       |    6620     |  | 94.230 | 91.054 | 90.806 | 90.920 | 94.414 | 93.233 | 92.094 | 96.867 | 92.346 | 93.073 |
| P2       | 2883    |  | 88.683 | 75.815 | 80.223 | 81.136 | 85.568 | 83.840 | 82.028 | 94.799 | 72.506 | 83.711 |


