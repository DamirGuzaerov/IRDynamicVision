import os

path_to_first_dataset = 'path/to/first_dataset/annotations'
path_to_second_dataset = 'path/to/second_dataset/annotations'

class_mapping = {0: 0, 1: 1, 2: 2}

num_classes_first_dataset = 15

def update_annotation(annotations_path, class_mapping, offset):
    for filename in os.listdir(annotations_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(annotations_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in class_mapping:
                    parts[0] = str(class_mapping[class_id])
                else:
                    parts[0] = str(class_id + offset)
                updated_lines.append(' '.join(parts))

            with open(file_path, 'w') as file:
                file.write('\n'.join(updated_lines))

update_annotation(path_to_second_dataset, class_mapping, num_classes_first_dataset)