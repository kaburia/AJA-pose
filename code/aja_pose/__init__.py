# This is the entry point for all the code in the package. 

'''
1. Input the images and the annotations in MPII format
2. Test the model on the images and the annotations
3. Visualize the results of the model on the images and the logs
4. Train a model on the images and the annotations with default pretrained weights from modelDir
'''
# get the dir current path
import os
import yaml
import json
import sys
import warnings
import os.path as osp
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO
import numpy as np

# suppress warnings
warnings.filterwarnings("ignore")
# get the current path
module_dir = os.path.dirname(__file__)


 # yaml file path
yaml_file = os.path.join(module_dir, 'experiments', 'mpii', 'vhrbirdpose', 'w32_256x256_adam_lr1e-3_ak_vhr_s.yaml')
pretrained_model = os.path.join(module_dir, 'modelDir', 'all_animals_no_pretrain_106.pth')

class Args:
    def __init__(self, cfg, opts=[],modelDir='', logDir='', dataDir='', prevModelDir=''):
        self.cfg = cfg
        self.opts = opts
        self.modelDir = modelDir
        self.logDir = logDir
        self.dataDir = dataDir
        self.prevModelDir = prevModelDir

args = Args(cfg=yaml_file)

# Model class to test and train the model
class Model:        

    # Choose the annotations to train and test the model from the data and specify the protocol and test/train
    def annot(self, protocol='P1', test_train='test', animal_class=None):
        '''
        Input is either P1,P2 or P3
        If P3 animal class is required which is: [bird, fish, mammal, reptile, amphibian]
        returns the path to the json file
        '''
        if protocol == 'P1' or protocol == 'P2':
            if test_train == 'test':
                return os.path.join(module_dir, 'data', 'annot', f'ak_{protocol}',  'test.json')
            elif test_train == 'train':
                return os.path.join(module_dir, 'data', 'annot', f'ak_{protocol}',  'test.json')
            else:
                raise ValueError('test_train should be either test or train')
        elif protocol == 'P3':
           if animal_class == None:
               raise ValueError('animal_class is required for protocol P3')
           else:
               if test_train == 'test':
                   return os.path.join(module_dir, 'data', 'annot', f'ak_{protocol}_{animal_class}',  'test.json')
               elif test_train == 'train':
                   return os.path.join(module_dir, 'data', 'annot', f'ak_{protocol}_{animal_class}',  'train.json')
               else:
                   raise ValueError('test_train should be either test or train')
        else:
            raise ValueError('protocol should be either P1, P2 or P3')

    # adjust the model path in the yaml file
    def write_yaml(self, yaml_file, pretrained=pretrained_model, test_model=pretrained_model):
        with open(yaml_file, 'r') as file:
            # load the yaml file
            data = yaml.load(file, Loader=yaml.FullLoader)
            data['MODEL']['PRETRAINED'] = pretrained
            data['TEST']['MODEL_FILE'] = test_model
        with open(yaml_file, 'w') as file:
            # write the yaml file
            yaml.dump(data, file)
        
        
    # train the model on the images and the annotations
    def train(self, images_directory_path, train_mpii_json=None, valid_mpii_json=None, 
              protocol=None, animal_class=None, pretrained=None):
        # get the train method
        sys.path.append(os.path.join(module_dir, 'tools'))
        from aja_pose.tools.train import main as train_main
        # make the images and annotations in the correct format
        '''
        data
        ├── images
        │   ├── image1.jpg
        │   ├── image2.jpg
        
        ├── annot
        │   ├── train_mpii_json.json # train.json
        │   ├── valid_mpii_json.json # valid.json
        
        output # logs and the image results from output/vhr_s and log/vhr_s
        ├── output
        │   ├── vhr_s
        │   ├── log
        '''
        if protocol is not None:
            train_mpii_json = self.annot(protocol=protocol, test_train='train', animal_class=animal_class)
            valid_mpii_json = self.annot(protocol=protocol, test_train='test', animal_class=animal_class)
        else:
            if train_mpii_json is None or valid_mpii_json is None:
                raise ValueError('train_mpii_json and valid_mpii_json is required if protocol is not provided')            

       
        # create data/annot and output/vhr_s directories at the current working directory
        current_dir = os.getcwd()
        os.makedirs(f'{current_dir}/data/annot', exist_ok=True)
        os.makedirs(f'{current_dir}/data/images', exist_ok=True)
        os.makedirs(f'{current_dir}/output/vhr_s', exist_ok=True)


        # copy files 
        os.system(f'cp -r {images_directory_path}/* {current_dir}/data/images')
        os.system(f'cp -r {train_mpii_json} {current_dir}/data/annot')
        os.system(f'cp -r {valid_mpii_json} {current_dir}/data/annot')
        
        if pretrained is None:
            # The model file based on the argument passed
            self.write_yaml(yaml_file, pretrained_model)
        else:
            # The model file based on the argument passed
            self.write_yaml(yaml_file, pretrained=pretrained)

        # train the model
        print('Training the model...')
        train_main(args)
        
        # Create the output directory and move the logs and the images to the output directory
        output_directory = os.path.join(images_directory_path, '..')
        output_directory = os.path.join(output_directory, 'output_train')
        
        os.makedirs(output_directory, exist_ok=True)
        os.system(f'mv {current_dir}/output/vhr_s {output_directory}')
        os.system(f'mv {current_dir}/log/vhr_s {output_directory}')
        print(f'Find results at: {output_directory}')

        # clean up the data directory and the output directory 
        os.system(f'rm -r {current_dir}/data/*')
        # os.system(f'rm -r {current_dir}/data/*')
        os.system(f'rm -r {current_dir}/output/*')
        os.system(f'rm -r {current_dir}/log/*')
    
    def test(self, images_directory_path, annotations_mpii_json=None, protocol=None, animal_class=None, model=pretrained_model):
        # get the test method
        sys.path.append(os.path.join(module_dir, 'tools'))
        from aja_pose.tools.test import main as test_main
        # Note the name of the files need to be the same as the column image in the annotations_mpii_json
        # make the images and annotations in the correct format
        
        '''
        data
        ├── images
        │   ├── image1.jpg
        │   ├── image2.jpg
        
        ├── annot
        │   ├── annotations_mpii_json.json
        
        output # logs and the image results from output/vhr_s and log/vhr_s
        ├── output
        │   ├── vhr_s
        │   ├── log    
        '''
        # copy the images input to the data/images directory and the annotations to the data/annot directory where the module is
        # os.system(f'cp -r {images_directory_path}/* {module_dir}/data/images')
        # os.system(f'cp -r {annotations_mpii_json} {module_dir}/data/annot')

        if protocol is not None:
            annotations_mpii_json = self.annot(protocol=protocol, test_train='test', animal_class=animal_class)
        else:
            if annotations_mpii_json is None:
                raise ValueError('annotations_mpii_json is required if protocol is not provided')

        # The model file based on the argument passed
        self.write_yaml(yaml_file, test_model=model)    
        print(module_dir)    
        # create data/annot and output/vhr_s directories at the current working directory
        current_dir = os.getcwd()
        os.makedirs(f'{current_dir}/data/annot', exist_ok=True)
        os.makedirs(f'{current_dir}/data/images', exist_ok=True)
        os.makedirs(f'{current_dir}/output/vhr_s', exist_ok=True)


        # copy files 
        os.system(f'cp -r {images_directory_path}/* {current_dir}/data/images')
        os.system(f'cp -r {annotations_mpii_json} {current_dir}/data/annot')

        # test the model
        print('Testing the model...')
        # change directory to the module directory but after run     change back to working directory
        test_main(args)
        
        # command  = f'python {module_dir}/tools/test.py --cfg {yaml_file}'
        # subprocess.call(command, shell=True)
        # subprocess.run(['python', f'{module_dir}/tools/test.py'] + arguments)
        print('Model tested successfully!')
        
        # Create the output directory and move the logs and the images to the output directory
        output_directory = os.path.join(images_directory_path, '..')
        output_directory = os.path.join(output_directory, 'output_test')

        os.makedirs(output_directory, exist_ok=True)
        os.system(f'mv {current_dir}/output/vhr_s {output_directory}')
        os.system(f'mv {current_dir}/log/vhr_s {output_directory}')
        print(f'Find results at: {output_directory}')

        # clean up the data directory and the output directory 
        os.system(f'rm -r {current_dir}/data/*')
        # os.system(f'rm -r {current_dir}/data/annot/*')
        os.system(f'rm -r {current_dir}/output/*')
        os.system(f'rm -r {current_dir}/log/*')

    # def visualize(self):
    #     # visualize the model
    #     os.system(f'python tools/visualize.py --cfg {yaml_file} MODEL.MODEL_FILE {pretrained_model}')

class Args:
    def __init__(self, cfg, opts=[],modelDir='', logDir='', dataDir='', prevModelDir=''):
        self.cfg = cfg
        self.opts = opts
        self.modelDir = modelDir
        self.logDir = logDir
        self.dataDir = dataDir
        self.prevModelDir = prevModelDir


# path to data directory
class PoseInference(Model): # Inherit from the Model class
    # add path to lib directory
    def add_path(self):
        lib_path = osp.join(module_dir, 'lib')
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
    
    def load_model(self, pretrained=pretrained_model):
        sys.path.append(os.path.join(module_dir, 'lib'))
        from aja_pose.lib.config import cfg
        from aja_pose.lib.config import update_config

        
        super().write_yaml(yaml_file, pretrained=pretrained)
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        args = Args(yaml_file)
        
        update_config(cfg, args)
        
        # load the model
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )

        # load the model weights
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=device), strict=False)
        model.eval()

        return model
    # load image
    def load_image(self, image_path, transform=True):
        # check if image path is a string or an image
        if isinstance(image_path, str):
            # load the image
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError('Image path must be a string or an image')        
         # image transformation
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if transform:
            image = image_transform(image)
            # add batch dimension
            image = image.unsqueeze(0)
        return image
    
    # Image inference
    def image_pose(self, image_path, model=pretrained_model, thresh=0.5,):
        sys.path.append(os.path.join(module_dir, 'lib'))
        from aja_pose.lib.core.inference import get_max_preds       
        # load the model
        model = self.load_model(pretrained=model)
        # load the transformed image
        image_t = self.load_image(image_path)
        # load the original image
        image = self.load_image(image_path, transform=False)
        image_size = image.size
        # get the keypoints from the model
        with torch.no_grad():
            output = model(image_t) # shape (1,23,64,64) Heatmap
                # get the image centre
        def get_image_center():
            return 128,128

        # get the image scale with respect to 200px
        def get_image_scale():
            from math import sqrt
            width = 256
            height = 256
            return 200/256

        center = list(get_image_center())
        scale = get_image_scale()

        cent = np.array([[center]])
        sc = [scale]
        # print(sc)
        # coords, maxvals = get_final_preds(
        #                 cfg, output.clone().cpu().numpy(), cent, sc)
        coords, maxvals = get_max_preds(output.clone().cpu().numpy())
        # print(coords)
        # convert image to numpy
        # img = image.numpy().transpose(1,2,0)
        # resize to the original image size
        # img = cv2.resize(img, image_size)
        # thresh = 0.5
        # remove normalization
        # img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        # img = np.clip(img, 0, 1)
        # subset maxvals > thresh
        # maxvals = np.array([np.array([i for i in maxvals[0] if i >= thresh])])
     
        return coords, maxvals

        
    def draw_pose(self, image_path, thresh=0.1, model=pretrained_model):
        # load the image
        image = self.load_image(image_path, transform=False)
        # get the coords and maxvals
        coords, maxvals = self.image_pose(image_path, thresh=thresh, model=model)
        plt.imshow(image)
        plt.title('Image and predicted keypoints')
        plt.axis('off')

        # Plot the keypoints
        for i in range(len(coords[0])):  # Iterate through each keypoint
            x, y = coords[0][i]
            # x,y = x/256*image_size[0], y/256*image_size[1]
            x,y = x*image.size[0]/64, y*image.size[1]/64
            # print(x,y)
            if maxvals[0][i] < thresh:
                continue

            plt.scatter(x,y, color='red', s=10)  # Plot the keypoint on the image
        plt.show()
        
# object detection
class ObjectDetection:
    def load_model(self, model_path='yolov8s.pt'):
        model = model = YOLO(model_path)
        return model
    def image_detection(self, image_path, model='yolov8s.pt'):
        model = self.load_model(model)
        # load image
        image = Image.open(image_path).convert('RGB')
        results = model.predict(image)
        return results

# combine the two classes
class DetectionPoseInference(ObjectDetection):
    def __init__(self):
        self.det_model = self.load_model()
        # self.image_detect = self.image_detection()
        # initialize pose model
        self.pose_model = PoseInference()
    
    # handle detection by passing in an image and passing out the detected images bounding boxes
    def detect_objects(self, image_path):
        results = self.image_detection(image_path)
        # result = results[0]
        # len(result.boxes)
        # box = result.boxes[0]
        # class_id = result.names[box.cls[0].item()]
        # cords = box.xyxy[0].tolist()
        # cords = [round(x) for x in cords]
        # conf = round(box.conf[0].item(), 2)
        return results[0]
    
    # Get the deteted bounding boxes and pass them to the pose model
    def fit_pose(self, image_path, thresh=0.5, bbox_thresh=0.5, model=pretrained_model):
        # append the coords and maxvals to a list
        pose_coords = []
        pose_maxvals = []
        # get the data to a dictionary for each bounding box in frame
        frame_data = dict()
        # get the bounding boxes
        result = self.detect_objects(image_path)
        # load the image
        image = Image.open(image_path).convert('RGB')
        count = 0
        for box in result.boxes:
            bbox_data = dict()
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            # print("Object type:", class_id)
            # print("Coordinates:", cords)
            # print("Probability:", conf)
            # print("---")
            if conf < bbox_thresh:
                continue
            x1, y1, x2, y2 = cords
            cropped_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            coords, maxvals = self.pose_model.image_pose(cropped_image, thresh=thresh, model=model)
            bbox_data['class_id'] = class_id
            bbox_data['coords'] = cords
            bbox_data['threshold'] = conf
            bbox_data['pose_coords'] = coords
            bbox_data['pose_maxvals'] = maxvals
            frame_data['bbox_'+str(count)] = bbox_data
            count += 1
        return frame_data
        # # pass the cropped image to the pose model based on the bounding boxes if any
        # for box in boxes:
        #     x1, y1, x2, y2 = box
        #     # crop the image
        #     # cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        #     cropped_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
        #     # get the pose coords and maxvals for each cropped image
        #     # self.pose_model.image_pose(cropped_image, thresh=thresh)
        #     coords, maxvals = self.pose_model.image_pose(cropped_image, thresh=thresh, model=model)
        #     pose_coords.append(coords)
        #     pose_maxvals.append(maxvals)
        #     print('Detected keypoints:', coords)
        # return pose_coords, pose_maxvals
    
    def draw_detection_pose(self, image_path, thresh=0.1, bbox_thresh=0.5, model=pretrained_model):
        # load the image
        image = Image.open(image_path).convert('RGB')
        # get the frame data for all the bounding boxes
        frame_data = self.fit_pose(image_path, thresh=thresh, bbox_thresh=bbox_thresh, model=model)
        plt.imshow(image)
        plt.title('Image and predicted keypoints')
        plt.axis('off')
        print('Bounding boxes detected:', len(frame_data))
        print('---')
        print('Bounding box data:', frame_data)
        
        # get the data for each bounding box in the frame and plot all the keypoints for each bounding box as well
        for key in frame_data.keys():
            print('Bounding box:', key)
            bbox_data = frame_data[key]
            class_id = bbox_data['class_id']
            cords = bbox_data['coords']
            pose_coords = bbox_data['pose_coords']
            pose_maxvals = bbox_data['pose_maxvals']
            print('Class:', class_id)
            print('Bounding box:', cords)
            print('Pose keypoints:', pose_coords)
            print('Pose maxvals:', pose_maxvals)
            print('---')
            # get the width and height of the bounding box
            width = cords[2] - cords[0]
            height = cords[3] - cords[1]
            # Plot the keypoints
            for i in range(len(pose_coords[0])):  # Iterate through each keypoint
                x, y = pose_coords[0][i]
                print(x,y)
                print('---')
                print(pose_maxvals[0][i])
                x,y = x*width/64 + cords[0], y*height/64 + cords[1]
                
                if pose_maxvals[0][i] < thresh:
                    continue
                plt.scatter(x,y, color='red', s=10)  # Plot the keypoint on the image
        plt.show()            
            