import os
import json
import math

# Set the paths to your result and ground truth folders
result_folder = 'json_output'
gt_folder = 'keypoint_files'
output_file = 'sorted_results.json'  # Specify the output JSON file name

result_list = []

# Iterate through result files
for result_file in os.listdir(result_folder):
    if result_file.endswith('.json'):
        result_file_path = os.path.join(result_folder, result_file)

        # Extract the file number from the result file name
        file_number = result_file.split('_')[-1].split('.')[0]
        file_number = str(int(file_number))

        # Find the corresponding ground truth file
        gt_file_name = file_number + '.json'
        gt_file_path = os.path.join(gt_folder, gt_file_name)

        # Load the JSON data from the result file
        with open(result_file_path, 'r') as result_json_file:
            result_data = json.load(result_json_file)

        # Extract the "instance_info" section from the result data
        instance_info = result_data.get("instance_info", [])

        # Load the JSON data from the corresponding ground truth file
        with open(gt_file_path, 'r') as gt_json_file:
            gt_data = json.load(gt_json_file)

        # Extract the "annotations" section from the ground truth data
        annotations = gt_data["annotations"]
        image_id = gt_data["image_id"]

        # Iterate through each annotation in the ground truth
        error_sum = 0
        for annotation in annotations:
            gt_keypoints = annotation["keypoints"]
            gt_bbox = annotation["bbox"]
            for instance in instance_info:
                check = instance["bbox"]
                if check[0] == gt_bbox:
                    match = instance
                    break
                else:
                    continue

            result_keypoints = match["keypoints"]
            bbox_error = 0
            for i in range(len(gt_keypoints)):
                keypoint1 = gt_keypoints[i]
                keypoint2 = result_keypoints[i]
                if keypoint1 == [0, 0]:
                    continue
                elif keypoint1 != [0, 0] and keypoint2 == [0, 0]:
                    bbox_error += 100
                else:
                    x1, y1 = keypoint1
                    x2, y2 = keypoint2
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    bbox_error += distance

            error_sum += bbox_error

        result_dict = {"image_id": image_id, "error_score": error_sum}
        result_list.append(result_dict)

# Sort the result_list by error_score in descending order
sorted_list = sorted(result_list, key=lambda x: x["error_score"], reverse=True)

# Save the sorted_list to a JSON file
with open(output_file, 'w') as output_json_file:
    json.dump(sorted_list, output_json_file, indent=4)

print(f"Sorted results saved to {output_file}")

