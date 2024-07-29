import os
import shutil
import pandas as pd

# Paths to the directories and CSV files
train_dir = './dataset/train'
test_dir = './dataset/test'
train_csv = './dataset/Training_set.csv'
test_csv = './dataset/Testing_set.csv'

# Function to create directory if it does not exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to move images based on CSV labels
def move_images(csv_path, image_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)
 
    # Iterate through each row in the CSV
    for _, row in df.iterrows():
        # Ensure row contains 'filename' and 'label'
        if 'filename' not in row or 'label' not in row:
            print(f"Row {row} is missing 'filename' or 'label'. Skipping.")
            continue

        filename = row['filename']
        label = row['label']
        
        # Create the label directory if it does not exist
        label_dir = os.path.join(image_dir, label)
        print(f"Moving {filename} to {label_dir}")
        create_dir(label_dir)
        
        # Move the image to the label directory
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(label_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"File {src_path} does not exist.")

print(f"Starting...")

# Create the main dataset directories if they do not exist
create_dir('dataset/train')
create_dir('dataset/test')

print(f"Created dirs")

# Move training images
move_images(train_csv, train_dir)

# Move testing images
move_images(test_csv, test_dir)

