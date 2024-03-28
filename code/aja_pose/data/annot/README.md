Annotations for the are stored in this directory. The annotations are stored in the MPII format with train and test json files. The annotations are stored in the following format:

| Field                | Description                                                |
|----------------------|------------------------------------------------------------|
| image                | Path to image                                              |
| animal               | Name of animal                                             |
| animal_parent_class  | Parent class of the animal (e.g., Amphibian)               |
| animal_class         | Class of the animal (e.g., Amphibian)                       |
| animal_subclass      | Subclass of the animal (e.g., Frog / Toad)                  |
| joints_vis           | Visibility of joints (1 means visible, 0 means not visible) |
| joints               | Coordinates of the joints. All images are in 640×360 px(width × height) resolution. Invisible joint coordinates are [-1, -1]. There are 23 keypoints |
| scale                | Scale of bounding box with respect to 200px                |
| centre               | Coordinates of the centre point of the bounding box         |


