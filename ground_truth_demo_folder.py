import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run pose estimation on a folder of images and JSON files')
    parser.add_argument('model_config', help='Path to the model config file')
    parser.add_argument('model_checkpoint', help='Path to the model checkpoint file')
    parser.add_argument('image_folder', help='Folder containing images')
    parser.add_argument('json_folder', help='Folder containing JSON files')
    parser.add_argument('output_folder', help='Output folder for saving results')
    parser.add_argument('--device', default='cpu', help='Device for inference')
    
    args = parser.parse_args()

    # List image files in the image folder
    image_files = os.listdir(args.image_folder)

    for image_file in image_files:
        # Check if the file is an image (you can use a more robust method)
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            # Construct the full path to the image file
            image_path = os.path.join(args.image_folder, image_file)

            # Construct the corresponding JSON file name
            image_id, _ = os.path.splitext(image_file)
            image_id = str(int(image_id))
            json_file_name = f"{image_id}.json"
            json_path = os.path.join(args.json_folder, json_file_name)
            # Check if the JSON file exists
            if os.path.exists(json_path):
                # Build the command
                
                model_cfg=args.model_config
                model_checkpoint=args.model_checkpoint
                output_path=args.output_folder
                device=args.device
                command = f"python3 demo/ground_truth_demo.py {model_cfg} {model_checkpoint} {json_path} --input {image_path} --output-root {output_path} --save-predictions --device={device}"

                # Run the command
                subprocess.run(command, shell=True)
            else:
                print(f"Skipping {image_file} because JSON file not found")

if __name__ == '__main__':
    main()

