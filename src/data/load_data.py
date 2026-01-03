import pandas as pd
import yaml
import os

def load_config(config_path='src/config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_train_data(config=None):
    if config is None:
        config = load_config()
    path = os.path.join(config['paths']['data_raw'], 'train.csv')
    return pd.read_csv(path)

def load_test_data(config=None):
    if config is None:
        config = load_config()
    path = os.path.join(config['paths']['data_raw'], 'test.csv')
    return pd.read_csv(path)

def load_sample_submission(config=None):
    if config is None:
        config = load_config()
    path = os.path.join(config['paths']['data_raw'], 'sample_submission.csv')
    return pd.read_csv(path)