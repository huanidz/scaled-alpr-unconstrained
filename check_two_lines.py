import os

def find_multiline_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    kek = lines[0].split(",")
                    if len(kek) != 8:
                        print("file: ", file_path)
                    # total_coords = lines.split(",")
                    # print(f"==>> total_coords: {total_coords}")
                    if len(lines) > 1:
                        print(file_path)

# Usage example
folder_path = '/mnt/sda/vollmont_data/wpod_data/splitted_sub/eval'
find_multiline_files(folder_path)