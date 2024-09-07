import os
import shutil
import random

dataset = 'C:/Users/Lenovo/Desktop/The IQ-OTHNCCD lung cancer dataset'

def split_data(source_dir, train_dir, validation_dir, test_dir, train_size=0.7, validation_size=0.2):
    categories = ['Bengin', 'Malignant', 'Normal']

    for category in categories:
        source_category_dir = os.path.join(source_dir, category)
        all_files = os.listdir(source_category_dir)
        random.shuffle(all_files)

        num_files = len(all_files)
        num_train = int(num_files * train_size)
        num_validation = int(num_files * validation_size)

        train_files = all_files[:num_train]
        validation_files = all_files[num_train:num_train + num_validation]
        test_files = all_files[num_train + num_validation:]

        for file_list, target_dir in zip([train_files, validation_files, test_files], [train_dir, validation_dir, test_dir]):
            category_dir = os.path.join(target_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            for file in file_list:
                shutil.copy(os.path.join(source_category_dir, file), category_dir)

source_directory = dataset
train_directory = 'Train'
validation_directory = 'Val'
test_directory = 'Test'

split_data(source_directory, train_directory, validation_directory, test_directory)
