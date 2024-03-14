NVIDIA Tesla V100-PCIE 32GB (CUDA10.2, PyTorch1.10.0, Ubuntu18.04)
1. install virtualenv
```bash
sudo apt-get install python3.8-venv
```
2. create virtual environment
```bash
python3.8 -m venv vhr
```
3. activate virtual environment
```bash
source vhr/bin/activate
```
4. install pytorch
```bash
# CUDA 10.2
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
5. Clone deep-high-resolution-net.pytorch
```bash
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git
cd deep-high-resolution-net.pytorch
```
6. install requirements
```bash
pip install -r requirements.txt
```
7. clone Animal-Kingdom repository
```bash
cd .. 
git clone https://github.com/sutdcv/Animal-Kingdom.git
```
8. Copy files from deep-high-resolution-net.pytorch to Animal-Kingdom
hrnet_location2move = 'Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet'
hrnet_experiments_from = 'deep-high-resolution-net.pytorch/experiments'
hrnet_lib_from = 'deep-high-resolution-net.pytorch/lib'
hrnet_tools_from = 'deep-high-resolution-net.pytorch/tools'
```bash
cp -r deep-high-resolution-net.pytorch/experiments Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet
cp -r deep-high-resolution-net.pytorch/lib Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet
cp -r deep-high-resolution-net.pytorch/tools Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet
```
9. Create symbolic links to files in Animal-Kingdom
Change $DIR_AK/pose_estimation/code/code_new/prepare_dir_PE.sh to root directory of Animal-Kingdom
```bash
cd Animal-Kingdom/Animal_Kingdom/pose_estimation/code/code_new
bash prepare_dir_PE.sh
cd ..
cd ..
cd ..
cd ..
cd ..
```
10. Make libs
```bash
cd deep-high-resolution-net.pytorch/lib
sudo apt install make
sudo apt install nvidia-cuda-toolkit
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install python3-dev
make
cd ..
cd ..
```
11. Install COCOAPI
```bash
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
pip install matplotlib
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
cd ..
cd ..
```
12. Make directories
```bash
cd Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet
mkdir output
mkdir log
```
13. Get the pretrained models
```bash
mkdir models
cd ..
cd ..
cd ..
cd ..
cd ..
cd ..
```
14. Clone VHR-BirdPose
```bash
git clone https://github.com/LuoXishuang0712/VHR-BirdPose.git
# Move files around
cp VHR-BirdPose/lib/models/pose_vhr.py VHR-BirdPose/lib/models/cross_attn.py VHR-BirdPose/lib/models/vit.py VHR-BirdPose/lib/models/base_backbone.py Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet/lib/models
cp -r VHR-BirdPose/experiments/mpii/vhrbirdpose Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet/experiments/mpii
cp -f VHR-BirdPose/lib/utils/utils.py Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet/lib/utils
```
15. Install modules
```bash
python -m pip install timm==0.4.9 einops
cd /home/Austin/2024_ICME_Challenge/Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet/
```
16. Append import models.pose_vhr to the end of the file $AK_PE%/lib/models/__init__.py.
make from Animal-kingdom
```bash
mkdir -p output/ak_P3_bird/vhr_birdpose_b
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

python tools/train.py --cfg experiments/mpii/vhrbirdpose/w32_256x256_adam_lr1e-3_ak_vhr_s.yaml
lspci | grep -i nvidia
sudo apt install nvidia-utils-470
sudo apt update
sudo apt remove '^nvidia'
sudo apt autoremove
sudo apt-get purge 'nvidia*'
sudo apt autoremove
sudo reboot now
sudo apt install nvidia-driver-460
sudo apt install nvidia-driver-470 libnvidia-gl-470 libnvidia-compute-470 libnvidia-decode-470 libnvidia-encode-470 libnvidia-ifr1-470 libnvidia-fbc1-470
sudo apt install nvidia-cuda-toolkit
sudo reboot now
nvidia-smi
```