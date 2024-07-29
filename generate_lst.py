import os

def generate_lst_file(dataset_path, output_lst_file):
    lst_content = []
    idx = 0

    for class_label, class_folder in enumerate(os.listdir(dataset_path)):
        class_folder_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_folder_path):
            for img_file in os.listdir(class_folder_path):
                img_file_path = os.path.join(class_folder_path, img_file)
                if os.path.isfile(img_file_path):
                    lst_content.append(f"{idx}\t{class_label}\t{class_folder}/{img_file}\n")
                    idx += 1

    with open(output_lst_file, 'w') as f:
        f.writelines(lst_content)

# Paths
train_dataset_path = "./dataset/train/train"
validation_dataset_path = "./dataset/train/val"

# Output .lst files
train_lst_file = "train.lst"
validation_lst_file = "validation.lst"

# Generate .lst files
generate_lst_file(train_dataset_path, train_lst_file)
generate_lst_file(validation_dataset_path, validation_lst_file)

