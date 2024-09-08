# [AJA-pose](https://sutdcv.github.io/MMVRAC/)


TL;DR This repo contains work on development of animal pose estimation model using the animal kingdom dataset.

# What is in this repo?
- AJA/pose folder contains a [ReactJS based web app](https://aja-pose.vercel.app/) that describe the challenge we are taking part in.
- Code folder contains:
    - Finetuned VHRNet model on animal kingdom dataset.
    - A [PyPI package](https://pypi.org/project/aja-pose/) that can be downloaded and installed with pip install aja-pose

- Object detection folder contains
    - a fine tuned YOLOV8  model for detecting animals using animal kingdom dataset.

Follow the steps.md file to reproduce the VHRModel.

Examples of the predictions vs ground truth images.

Ground Truth
![image](https://github.com/Antony-gitau/2024_ICME_Challenge/assets/88529649/d8c4c84a-88c9-4ef7-bcf3-842e5e74d42b)



Prediction
![image](https://github.com/Antony-gitau/2024_ICME_Challenge/assets/88529649/1db2b2bc-94a5-4f2d-acce-9bb85986ca05)


https://github.com/kaburia/AJA-pose/assets/88529649/e6f2e29d-a71a-4740-bab3-7540cdac722a
## Citations

If you use this package in your research, please cite it using the following BibTeX entry:

```bibtex
@inproceedings{kibaara2024aja,
  title={AJA-Pose: A Framework for Animal Pose Estimation Based on VHR Network Architecture},
  author={Kibaara, Austin Kaburia and Kabura, Joan and Gitau, Antony and Maina, Ciira},
  booktitle={2024 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}

