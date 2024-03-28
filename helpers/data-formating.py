import os
import json

class_ids = {
    "Bird": 0,
    "Mammal":1,
    "Amphibian":2,
    "Fish":3,
    "Reptile":4
}

def convert_to_yolo_format_using_scale(annotation, img_width, img_height):
    #the annotation is in format of a list
    # Extract the class ID, scale, and center coordinates from the annotation
    class_id = class_ids[annotation[0]['animal_parent_class']]
    scale = annotation[0]['scale']
    #extract the center coordinates
    center_x = annotation[0]['center'][0]
    center_y = annotation[0]['center'][1]
    
    # Calculate bounding box dimensions based on scale
    bbox_width = scale * 200
    bbox_height = scale * 200  # Assuming the scaling factor applies uniformly to width and height
    
    # Normalize the bounding box parameters
    x_center = center_x / img_width
    y_center = center_y / img_height
    bbox_width /= img_width
    bbox_height /= img_height
    
    return class_id, x_center, y_center, bbox_width, bbox_height

# Define the base directory where the class folders are located
base_dir = 'D:/animal kingdom challenge/pose_estimation-20240304T032712Z-002/pose_estimation/annotation'

# Output directory for YOLO formatted annotations
output_dir = 'annotations'

# Your class folders
class_folders = ['ak_P1', 'ak_P2','ak_P3_amphibian', 'ak_P3_bird','ak_P3_fish', 'ak_P3_reptile', 'ak_P3_mammal']

# Iterate over each class folder
for folder in class_folders:
    current_folder = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    
    # Iterate over each annotation file in the current class folder
    for filename in os.listdir(current_folder):
        if filename.endswith('.json'):  # Ensure we're only processing JSON files
            file_path = os.path.join(current_folder, filename)
            
            # Read the annotation file
            with open(file_path, 'r') as file:
                annotation = json.load(file)
            
            # Convert the annotation to YOLO format
            yolo_format_annotation = convert_to_yolo_format_using_scale(annotation, 640, 360)
            
            # Construct the output filename and path
            output_filename = os.path.splitext(filename)[0] + '.txt'  # Change the extension to .txt
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the YOLO formatted annotation
            with open(output_path, 'w') as output_file:
                output_file.write(" ".join(map(str, yolo_format_annotation)) + '\n')

print("Conversion complete.")
