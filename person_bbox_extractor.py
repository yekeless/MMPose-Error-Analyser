import json
import os
# Load the list of dictionaries from the JSON file
with open('person_keypoints_val2017.json', 'r') as json_file:
    data_list = json.load(json_file)

# Initialize a dictionary to store the transformed data
transformed_data_dict = {}

# Loop through each dictionary in the list
for data in data_list["annotations"]:
    image_id = data["image_id"]
    
    if image_id not in transformed_data_dict:
        transformed_data_dict[image_id] = {
            "image_id": image_id,
            "labels": [],
            "scores": [],
            "bboxes": [],
        }
    
    transformed_data_dict[image_id]["labels"].append(0)
    transformed_data_dict[image_id]["scores"].append(1.0)
    transformed_data_dict[image_id]["bboxes"].append(data["bbox"])

# Convert the dictionary values to a list
transformed_data_list = list(transformed_data_dict.values())
output_folder = 'gt_bbox_results_correct_format'
os.makedirs(output_folder, exist_ok=True)
for data in transformed_data_list:
    image_id = data["image_id"]
    labels = data["labels"]
    scores = data["scores"]
    #formatting bounding box information (COCO => Detector Output)
    for bbox in data["bboxes"]:
        bbox[2]=bbox[0]+bbox[2]
        bbox[3]=bbox[1]+bbox[3]
    bboxes = data["bboxes"]    	
    # Create a new dictionary without the image ID
    new_data = {
        "labels": labels,
        "scores": scores,
        "bboxes": bboxes,
    }

    # Convert the new data dictionary to JSON format
    new_json_data = json.dumps(new_data)

    # Save the new JSON data to a file named with the image ID within the output folder
    output_filename = os.path.join(output_folder, f'{image_id}.json')
    with open(output_filename, 'w') as output_file:
        output_file.write(new_json_data)

print("Files saved.")
