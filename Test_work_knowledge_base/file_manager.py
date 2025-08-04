import os
import shutil

DATA_DIR = "data"


def upload_files(file_paths):
    os.makedirs(DATA_DIR, exist_ok=True)
    for path in file_paths:
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(DATA_DIR, os.path.basename(path)))


def list_uploaded_files():
    return os.listdir(DATA_DIR)


def delete_file(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False
