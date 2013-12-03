import pandas as pd

from data_wrangling.cps_wrangling.analysis.helpers import get_useful, df_unique


def fix_timestamp(wp_original):
    wp = wp_original.copy()
    wp = get_useful(wp)
    years = wp['year'].dropna()
    months = wp['month'].dropna()
    ts = wp['timestamp'].dropna()

    ncols = len(years.columns)
    idx = range(1, ncols + 1)

    stamp = df_unique(ts, index=idx)
    years = df_unique(years, index=idx)
    months = df_unique(months, index=idx)

    expected = pd.to_datetime(years.astype(int).astype(str) + '-' +
                              months.astype(int).astype(str) + '-01')

    needfix = expected[stamp != expected].index

    if needfix is not None:
        for month in needfix:
            wp_original['timestamp'][month] = expected[month]

        return wp_original

def main():
    import json

    with open('../panel_construction/settings.txt') as f:
        settings = json.load(f)

    panel_store = pd.HDFStore(settings['panel_store_path'])
    keys = panel_store.keys()
    for k in keys:
        wp = panel_store.select(k)
        prev_shape = wp.shape
        wp = fix_timestamp(wp)

        if wp is not None:
            assert prev_shape == wp.shape
            wp.to_hdf(panel_store, k, append=False)
            print("Fixed " + k)
        else:
            print("Skipped " + k)

if __name__ == '__main__':
    main()
