'''
Functions to view the images with the bounding boxes and the keypoints
The images are from the dataset subclass
The images are displayed using the opencv library
press any key to close the image window this acts as a next button
'''
import cv2
import os


# view the images 
def view_images(df, category):
    image_dir = 'dataset'
    # get the images in the csv file
    images = df[df.animal_class == category].image.values
    # get the images in the dataset folder
    images = [os.path.join(image_dir, image) for image in images]
    # display the images
    for image in images:
        img = cv2.imread(image)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# view the images with the bounding boxes multiply the scale by 200 for the width and height of the bounding box
def view_images_with_bounding_boxes(df, category):
    image_dir = 'dataset'
    # get the images in the csv file
    images = df[df.animal_subclass == category].image.values
    # get the images in the dataset folder
    images = [os.path.join(image_dir, image) for image in images]
    # multiply the scale by 200 for the width and height of the bounding box from the selected category
    scale  = df[df.animal_subclass == category].scale.values * 200
    # the center of the bounding box
    centre = df[df.animal_subclass == category].center.values
    # the width = scale, height = scale 
    width = scale
    height = scale

    # display the images with the bounding boxes
    for i in range(len(images)):
        img = cv2.imread(images[i])
        x, y = centre[i]
        x1 = int(x - width[i]/2)
        y1 = int(y - height[i]/2)
        x2 = int(x + width[i]/2)
        y2 = int(y + height[i]/2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# view the images with the keypoints do not display the keypoints that are not visible
def view_images_with_keypoints(df, category):
    image_dir = 'dataset'
    # get the images in the csv file
    images = df[df.animal_subclass == category].image.values
    # get the images in the dataset folder
    images = [os.path.join(image_dir, image) for image in images]
    # the keypoints
    keypoints = df[df.animal_subclass == category].joints.values
    # the visibility of the keypoints
    visibility = df[df.animal_subclass == category].joints_vis.values

    # display the images with the keypoints
    for i in range(len(images)):
        img = cv2.imread(images[i])
        for j in range(len(keypoints[i])):
            x, y = keypoints[i][j]
            if visibility[i][j] == 1:
                img = cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# view the images with the keypoints and bounding boxes
# view with the bounding boxes and the keypoints
def view_images_with_bounding_boxes_and_keypoints(df, category):
    image_dir = 'dataset'
    # get the images in the csv file
    images = df[df.animal_subclass == category].image.values
    # get the images in the dataset folder
    images = [os.path.join(image_dir, image) for image in images]
    # multiply the scale by 200 for the width and height of the bounding box from the selected category
    scale  = df[df.animal_subclass == category].scale.values * 200
    # the center of the bounding box
    centre = df[df.animal_subclass == category].center.values
    # the width = scale, height = scale
    width = scale
    height = scale
    # the keypoints
    keypoints = df[df.animal_subclass == category].joints.values
    # the visibility of the keypoints
    visibility = df[df.animal_subclass == category].joints_vis.values

    # display the images with the bounding boxes and the keypoints
    for i in range(len(images)):
        img = cv2.imread(images[i])
        x, y = centre[i]
        x1 = int(x - width[i]/2)
        y1 = int(y - height[i]/2)
        x2 = int(x + width[i]/2)
        y2 = int(y + height[i]/2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for j in range(len(keypoints[i])):
            x, y = keypoints[i][j]
            if visibility[i][j] == 1:
                img = cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()