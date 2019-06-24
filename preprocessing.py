import os
import shutil
from os import path


def parse_number_dataset():
    file_path = r"..\numbers-master"
    final_path = r"..\numbers"

    file_dict = {str(i): 0 for i in range(10)}

    for directory in os.listdir(file_path):
        dir_path = path.join(file_path, directory)
        for image_dir in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, image_dir)):
                image_path = path.join(dir_path, image_dir)
                for image_file in os.listdir(image_path):
                    current_image_path = os.path.join(image_path, image_file)
                    if os.path.isfile(current_image_path):
                        final_file_path = os.path.join(final_path, image_dir)
                        if not path.exists(final_file_path):
                            os.mkdir(final_file_path)
                        final_image_name = str(file_dict[image_dir]) + '.png'
                        file_dict[image_dir] += 1
                        shutil.copy(current_image_path, final_file_path)
                        os.rename(path.join(final_file_path, image_file), path.join(final_file_path, final_image_name))


parse_number_dataset()
