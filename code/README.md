Welcome to our module more Documentation to come soon.
# Aja-pose


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N3v7Y-PN9uvw5V5PUbAYqh9vGkLfm_Km?usp=sharing)
## Getting Started


```bash
pip install -e .
```

Test our model
Required is the images directory and the test.json file in mpii format
```python
from aja_pose import Model
import urllib.request

# Get our model
url = "https://storage.googleapis.com/figures-gp/animal-kingdom/all_animals_no_pretrain_106.pth"
destination = "all_animals_no_pretrain_60.pth"

urllib.request.urlretrieve(url, destination)

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
model.train(images_directory, mpii_json, model_file)
```

