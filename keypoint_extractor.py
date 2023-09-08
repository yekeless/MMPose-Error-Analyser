import json
import os

# Load the COCO person keypoints JSON file
with open('person_keypoints_val2017.json', 'r') as f:
    data = json.load(f)

# Create a dictionary to store image data
image_data_dict = {}

# Iterate through the annotations in the JSON data
for annotation in data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    bbox[2]=bbox[0]+bbox[2]
    bbox[3]=bbox[1]+bbox[3]
    keypoints = annotation['keypoints']
    num_keypoints = annotation["num_keypoints"]
    # Extract only the first two numbers from each keypoint triplet
    keypoints = [keypoints[i:i+3][:2] for i in range(0, len(keypoints), 3)]

    # Create a sub-dictionary for the current annotation
    sub_dict = {
        'bbox': bbox,
        'keypoints': keypoints,
        'num_keypoints': num_keypoints
    }

    # Check if the image ID already exists in the dictionary
    if image_id in image_data_dict:
        # If it exists, append the sub-dictionary to the list
        image_data_dict[image_id]['annotations'].append(sub_dict)
    else:
        # If it doesn't exist, create a new dictionary entry
        image_data_dict[image_id] = {
            'image_id': image_id,
            'annotations': [sub_dict]
        }

# Convert the dictionary values to a list
image_data_list = list(image_data_dict.values())

# Create a directory to store the JSON files
output_directory = 'keypoint_files'
os.makedirs(output_directory, exist_ok=True)

# Iterate over the list and save individual JSON files
for image_data in image_data_list:
    image_id = image_data['image_id']
    file_path = os.path.join(output_directory, f'{image_id}.json')
    with open(file_path, 'w') as outfile:
        json.dump(image_data, outfile)

    print(f"Saved data for Image ID {image_id} to {file_path}")

