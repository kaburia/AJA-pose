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



# get the current path
module_dir = os.path.dirname(__file__)

 # yaml file path
yaml_file = os.path.join(module_dir, 'experiments', 'mpii', 'vhrbirdpose', 'w32_256x256_adam_lr1e-3_ak_vhr_s.yaml')
pretrained_model = os.path.join(module_dir, 'modelDir', 'all_animals.pth')

class Args:
    def __init__(self, cfg, opts=[],modelDir='', logDir='', dataDir='', prevModelDir=''):
        self.cfg = cfg
        self.opts = opts
        self.modelDir = modelDir
        self.logDir = logDir
        self.dataDir = dataDir
        self.prevModelDir = prevModelDir




# test the model
def test_model(dataDir, modelDir='', logDir='', outputDir=''):
    from tools.test import main
    # if logdir is not provided, create at the working directory
    if logDir == '':
        # make the log directory
        if os.path.exists(os.path.join(dataDir, 'logs')):
            logDir = os.path.join(dataDir, 'logs')
        else:
            os.mkdir(os.path.join(dataDir, 'logs'))
            logDir = os.path.join(dataDir, 'logs')
    if outputDir == '':
        # make the output directory
        if os.path.exists(os.path.join(dataDir, 'output')):
            outputDir = os.path.join(dataDir, 'output')
        else:
            os.mkdir(os.path.join(dataDir, 'output'))
            outputDir = os.path.join(dataDir, 'output')
    

    args = Args(
        cfg=yaml_file,
        modelDir=modelDir,
        logDir=logDir,
        dataDir=dataDir
    )
    main(args)

def run_inference(image):
    pass

# train the model
def train_model():
    from tools.train import main
    main()

# visualize the results
    
