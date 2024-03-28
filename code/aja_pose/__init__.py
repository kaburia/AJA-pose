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
warnings.filterwarnings("ignore")
# get the current path
module_dir = os.path.dirname(__file__)

sys.path.append(os.path.join(module_dir, 'tools'))

from aja_pose.tools.train import main as train_main
from aja_pose.tools.test import main as test_main


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
