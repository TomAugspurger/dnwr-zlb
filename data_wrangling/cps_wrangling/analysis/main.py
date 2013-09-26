import json

import pandas as pd


def load_settings(settings_path):
    with open(settings_path, 'r') as f:
        settings = json.load(f)

    return settings


def get_ee(df, col="PEMLR"):
    try:
        lf = df[col]
    except KeyError as e:
        log_missing_col(e)
        raise e
    unemp = lf[(lf == 3) | (lf == 4)]
    emp = lf[(lf == 1) | (lf == 2)]

def get_eu(df):
    pass


def log_missing_col(e):
    pass

# @cache
settings = load_settings('settings.json')
store = pd.HDFStore(settings['store_path'])
for k, _ in store.iteritems():
    df = store.select(k)
    ee = get_ee(df)
    eu = get_eu(df)

store.close()
