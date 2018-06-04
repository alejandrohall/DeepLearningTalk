import math
import random
import shutil

from tqdm import tqdm
from pathlib import Path


def split_train_valid(path, valid_rate=0.2):
    train_path = Path(path) / 'train'

    for path_in_train in train_path.iterdir():
        files_in_path = list(path_in_train.glob('*'))

        if len(files_in_path) * valid_rate > 1:
            number_files_to_valid = int(math.ceil(len(files_in_path) * valid_rate))
        else:
            number_files_to_valid = 0

        destination_path = Path(path) / 'valid' / path_in_train.name
        destination_path.mkdir(exist_ok=True, parents=True)

        for _ in range(number_files_to_valid):
            choosen_file = random.choice(files_in_path)
            files_in_path.remove(choosen_file)

            shutil.move(choosen_file.as_posix(), destination_path.as_posix())


def create_sample_dataset(path, sample_rate=0.2):
    path_train = Path(path) / 'train'
    path_valid = Path(path) / 'valid'

    for path_in_list in (path_train, path_valid):
        for path_inside_path in path_in_list.iterdir():
            files_in_path = list(path_inside_path.glob('*'))
            number_files_to_sample = int(math.ceil(len(files_in_path) * sample_rate))

            destination_path = Path(path).parent / 'sample' / path_in_list.stem / path_inside_path.name
            destination_path.mkdir(exist_ok=True, parents=True)

            for _ in range(number_files_to_sample):
                choosen_file = random.choice(files_in_path)
                files_in_path.remove(choosen_file)

                shutil.copy2(choosen_file.as_posix(), destination_path.as_posix())


def remove_imbalance_in_dataset(path):
    base_path = Path(path)

    files_by_folder = dict()

    for folder_in_base in base_path.iterdir():
        files_by_folder[folder_in_base] = list(folder_in_base.glob('*'))

    max_files_in_folder = max([len(value) for value in files_by_folder.values()])

    for path, files in files_by_folder.items():
        if files:
            number_files_to_reconstruct = max_files_in_folder - len(files)

            for i in tqdm(range(number_files_to_reconstruct)):
                choice = random.choice(files)
                shutil.copy2(choice.as_posix(), (choice.parent / (choice.stem+str(i)+choice.suffix)).as_posix())