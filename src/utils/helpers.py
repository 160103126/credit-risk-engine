import os

def ensure_dir(path):
    """
    Ensure directory exists.
    """
    os.makedirs(path, exist_ok=True)