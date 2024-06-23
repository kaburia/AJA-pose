from aja_pose import Model

import os
import yaml
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
import json
import datetime

warnings.filterwarnings("ignore")
# get the current path
module_dir = os.getcwd()


 # yaml file path
yaml_file = os.path.join(module_dir, 'aja_pose', 'experiments', 'mpii', 'vhrbirdpose', 'w32_256x256_adam_lr1e-3_ak_vhr_s.yaml')
pretrained_model = os.path.join(module_dir, 'aja_pose', 'modelDir', 'all_animals_no_pretrain_106.pth')
yolo_model = os.path.join(module_dir, 'aja_pose', 'modelDir', 'yolo_det_best.pt')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# this_dir = osp.dirname(__file__)

lib_path = osp.join(module_dir, 'aja_pose', 'lib')
add_path(lib_path)

import models

from aja_pose.lib.config import cfg
from aja_pose.lib.core.inference import *
from aja_pose.lib.utils.transforms import transform_preds
# Import the Sort Algorithm
from aja_pose.sort import Sort
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


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
        coords, maxvals = get_max_preds(output.clone().cpu().numpy())
     
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
        # Add the option to load to GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # Load the model 
        model = YOLO(model_path)
        model.to(device)
        return model
    
    def image_detection(self, image_path, model='yolov8s.pt', stream=False):
        model = self.load_model(model)
        # load image 
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        # image = Image.open(image_path).convert('RGB')
        results = model.predict(image, stream=stream)
        return results

class DetectionPoseInference(ObjectDetection):
    def __init__(self):
        self.det_model = self.load_model()
        self.pose_model = PoseInference()

    def detect_objects(self, image_path, model='yolov8s.pt', stream=False):
        results = self.image_detection(image_path, model=model, stream=stream)
        return results[0]

    def _process_bbox(self, args):
        image, box, result, thresh, bbox_thresh, model = args
        bbox_data = dict()
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        if conf < bbox_thresh:
            return None
        
        x1, y1, x2, y2 = cords
        cropped_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
        coords, maxvals = self.pose_model.image_pose(cropped_image, thresh=thresh, model=model)
        bbox_data['class_id'] = class_id
        bbox_data['coords'] = cords
        bbox_data['threshold'] = conf
        bbox_data['pose_coords'] = coords
        bbox_data['pose_maxvals'] = maxvals

        return bbox_data

    def fit_pose(self, image_path, thresh=0.5, bbox_thresh=0.5, model='pretrained_model', 
                 stream=False, detection_model='yolov8s.pt'):
        
        # get the bounding boxes
        result = self.detect_objects(image_path, stream=stream, model=detection_model)

        # can be an image or a video frame check if string or image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path

        # Prepare arguments for multiprocessing
        args = [(image, box, result, thresh, bbox_thresh, model) for box in result.boxes]

        num_cores = int(multiprocessing.cpu_count() * 0.5)
        print(num_cores)

        # Use multiprocessing to process bounding boxes in parallel
        with multiprocessing.Pool(num_cores) as pool:
            results = pool.map(self._process_bbox, args)

        # Collect results
        frame_data = {f'bbox_{i}': bbox_data for i, bbox_data in enumerate(results) if bbox_data is not None}

        return frame_data
    
    
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

# Add a class to track the object detection and pose detection (ObjectTracking)
# Track objects as they move in the frame
class ObjectTracking:
    def __init__(self):
        # initialize the object detection and pose detection classes
        self.det_pose = DetectionPoseInference()
        # self.tracker = Sort()  # Initialize the tracker

    # Read video frames or live feed
    def read_video(self, video_path): # video_path is the path to the video or 0/1 for live feed
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                raise ValueError('Error reading video')
            yield frame
        cap.release()

    # Track objects in the frame
    def track_object(self, video_path, detection_model='yolov8s.pt', pose_model=pretrained_model, 
                     thresh=0.1, bbox_thresh=0.5):
        # get the video frames
        video = self.read_video(video_path)
        # Define the output video path
        output_path = 'video.avi'

        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Frames per second
        frame_size = (640, 480)  # Frame size (width, height)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Iterate over the frames and write them to the video file
        # for frame in video:
            
        for frame in video:
            # every frame goes through the detection model
            results = self.det_pose.fit_pose(frame, thresh=thresh, bbox_thresh=bbox_thresh, 
                                             model=pose_model, detection_model=detection_model) # dict of bounding boxes and keypoints
            # dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            dets = np.empty((0,5))
            for key in results.keys():
                bbox_data = results[key]
                cords = bbox_data['coords']
                x1, y1, x2, y2 = cords
                score = bbox_data['threshold']
                dets = np.vstack((dets, [x1, y1, x2, y2, score]))
            # get the tracking results
            tracks = self.tracker.update(dets) # sending detections to the tracker func
            # iterate over each track and draw the bounding box
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            video_writer.write(frame)

                # cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the video writer and close the video file
        video_writer.release()
        cv2.destroyAllWindows()

# Class to process videos
class VideoProcessor:
    def __init__(self):
        self.fit_pose = DetectionPoseInference()

    # Read video frames or live feed
    def read_video(self, video_path): # video_path is the path to the video or 0/1 for live feed
        print(f"Reading video from path: {video_path}")
        if not os.path.exists(video_path):
            raise ValueError(f"Video path does not exist: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame at position {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
                break
            yield frame
        cap.release()
    # Read video frames or live feed
    # def read_video(self, video_path): # video_path is the path to the video or 0/1 for live feed
    #     cap = cv2.VideoCapture(video_path)
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             raise ValueError('Error reading video')
    #         yield frame
    #     cap.release()
    
    # get video metadata
    def get_video_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Release the video capture object
        cap.release()

        return fps, frame_count, frame_width, frame_height
    
    def _process_frame(self, frame_no, frame, detection_model, pose_model, thresh, bbox_thresh):
        results = self.fit_pose.fit_pose(frame, thresh=thresh, bbox_thresh=bbox_thresh, model=pose_model, detection_model=detection_model)
        print(f'frame_{frame_no}')
        return frame_no, results
    
    # Process video frames
    def process_video(self, video_path, detection_model='yolov8s.pt', pose_model='pretrained_model', 
                      thresh=0.1, bbox_thresh=0.5, output_path='video.avi'):
        
        # store frame data
        video_frames = dict()
        # get the video frames
        video = self.read_video(video_path)

        # Use ThreadPoolExecutor to parallelize frame processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for no, frame in enumerate(video):
                futures.append(executor.submit(self._process_frame, no, frame, detection_model, pose_model, thresh, bbox_thresh))
            
            for future in as_completed(futures):
                frame_no, results = future.result()
                video_frames[f'frame_{frame_no}'] = results

        return video_frames
    
    # display frame
    def display_frame(self, video_path, frame_no=None, display=False, fps=False):
        '''
        Input:
            frame_no: int
                The frame number to display
            video_path: str
                The path to the video
        Output:
            frame: np.array
        '''
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()
        # check if the frame is read
        if not success:
            raise ValueError('The frame cannot be read')
        # get the fps
        if fps:
            fps = cap.get(cv2.CAP_PROP_FPS)
            # get the number of frames
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # release the video
            cap.release()
            return fps, num_frames
        if display:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        else:
            return frame
    
    # write to video/ file
    def write_data(self, video_frames, video_path=False, json_path=False, output_path='video/file_path'):
        # write the frames on top of the video
        if video_path:
            # video = self.read_video(video_path)

            # Define the video codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # read the fps and frame size from the video
            fps, frame_count, frame_width, frame_height = self.get_video_metadata(video_path)
            frame_size = (frame_width, frame_height)  # Frame size (width, height)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

            # write the video_frames for each frame to the video file
            for frame_no, frame_val in video_frames.items():
                # get the frame
                frame_ = frame_no.split('_')[1]
                frame = self.display_frame(video_path, frame_no=frame_)
                # Draw the bounding boxes and keypoints on the frame
                for key in frame_val.keys():
                    bbox_data = frame_val[key]
                    cords = bbox_data['coords']
                    pose_coords = bbox_data['pose_coords']
                    pose_maxvals = bbox_data['pose_maxvals']
                    # get the width and height of the bounding box
                    width = cords[2] - cords[0]
                    height = cords[3] - cords[1]            
                    # Plot the keypoints
                    for i in range(len(pose_coords[0])):
                        x, y = pose_coords[0][i]
                        x,y = x*width/64 + cords[0], y*height/64 + cords[1]
                        # if pose_maxvals[0][i] < thresh:
                        #     continue
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                        cv2.rectangle(frame, (int(cords[0]), int(cords[1])), (int(cords[2]), int(cords[3])), (0, 255, 0), 2)
                        # put text on top of the bounding box
                        cv2.putText(frame, bbox_data['class_id'], (int(cords[0]), int(cords[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                video_writer.write(frame)
                # cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
            # Release the video writer and close the video file
            video_writer.release()
            cv2.destroyAllWindows()

        elif json_path:
            # write to json file
            # Save the landmarks to a JSON file 
            if not os.path.exists('landmarks'):
                os.mkdir('landmarks')
            # write the landmarks dict to a JSON file with the current timestamp
            time_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            with open(f'landmarks/landmarks_{time_now}.json', 'w') as f:
                json.dump(video_frames, f, indent=4)
                    

inf = PoseInference()
obj_det = ObjectDetection()
pose_det = DetectionPoseInference()
obj_track = ObjectTracking()   
vid_process = VideoProcessor() 

import os

# data directory
dir_data_path = os.path.join(module_dir, 'aja_pose', 'dataDir')
# image path
img_path = os.path.join(dir_data_path, 'licensed-image.jfif')
# video path
vid_path = os.path.join(dir_data_path, 'sheep_grazing.mp4')


if __name__ == '__main__':
    print('Running AJA-pose begins... ')

    vid_process.write_data(video_frames = vid_process.process_video(vid_path, detection_model='yolov8n.pt', pose_model=pretrained_model,
                            thresh=0.1, bbox_thresh=0.5, output_path='sheep_grazing_det.avi'), json_path='k.json')