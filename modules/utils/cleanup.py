import os
import shutil

def clear_directory(dir_path):
    if not os.path.exists(dir_path):
        return

    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Cleanup failed for {path}: {e}")
