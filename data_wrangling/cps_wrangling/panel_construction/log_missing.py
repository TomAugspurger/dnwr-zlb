import argparse

import pandas as pd


def check(store, items):
    for k, _ in store.iteritems():
        df = store.select(k)
        for item in items:
            try:
                df[item]
            except KeyError as e:
                _log(k, e, "missing.log")
                print("Missing {} on {}".format(item, k))


def _log(k, e, log):
    with open(log, 'a') as f:
        f.write("{},{}\n".format(k, e))


def main(store_path, items):
    store = pd.HDFStore(store_path)
    check(store, items)

parser = argparse.ArgumentParser()
parser.add_argument('store_path', help='relative or absolute path to HDFStore',
                    type=str, nargs=1)
parser.add_argument('columns', help='column names to check for each month',
                    nargs='+')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.store_path, args.columns)
