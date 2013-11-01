import json

import arrow
import numpy as np
import pathlib
import pandas as pd


def get_id_by_dd():
    """
        -> ({dd_name: id}, id)

    """
    dd_to_id = {'jan2013': 'PEEDUCA', 'sep1995': 'PEEDUCA', 'may2012': 'PEEDUCA',
                'may2004': 'PEEDUCA', 'jan1998': 'PEEDUCA', 'jan2010': 'PEEDUCA',
                'jan2009': 'PEEDUCA', 'jan2007': 'PEEDUCA', 'jan2003': 'PEEDUCA',
                'aug2005': 'PEEDUCA', 'jun1995': 'PEEDUCA', 'jan1994': 'PEEDUCA',
                'apr1994': 'PEEDUCA'}
    official_id = 'PEEDUCA'
    return dd_to_id, official_id


def append_to_cpsstore(month, cps_store, dds, dd_to_id, official_id, settings):
    month_name = month.parts[-1].rstrip('.gz')
    original = cps_store.select('m' + month_name)
    if official_id in original.columns:
        return None
    dd_name = settings["month_to_dd"][''.join(month_name.split('_'))]
    new_id = dd_to_id[dd_name]
    ids = settings["dd_to_ids"][dd_name]
    dd = dds[dd_name]

    try:
        specs = dd[dd.id.isin(ids + [new_id])][['start', 'end']]
    except KeyError:
        specs = dd[dd.id.isin(ids + [new_id])][['start', 'stop']]
    specs = specs + [-1, 1]  # 0 index and half open on right
    specs = [(x[0], x[1]) for x in specs.values.tolist()]  # need [ tuples ]
    names = dd.id[dd.id.isin(ids + [new_id])].values
    data = pd.read_fwf(str(month), colspecs=specs,
                       compression='gzip', names=names)
    data[new_id] = data[new_id].str.rstrip('-')
    data[ids[-1]] = data[ids[-1]].str.rstrip('-')
    data = data.convert_objects(convert_numeric=True)
    data[data == -1] = np.nan
    data = data.set_index(ids)
    data = data.rename(columns={new_id: official_id})

    updated = original.join(data, how='left')
    writer(updated, cps_store, month_name)


def append_to_panel_store(month, panel_store, cps_store, official_id):
    """ We should use the official name for id from here out.
    Month refers to the panel we're fixing. so grab m, m+1, ... m+15
    """
    month_name = month.parts[-1].rstrip('.gz')
    try:
        original = panel_store.select('m' + month_name)
    except KeyError:
        print("No panel for {}. Move along.".format(month_name))
        return None

    if official_id in original.items:
        return None

    def gen_dfs():
        start_ar = arrow.get(month_name, 'YYYY_MM')
        rng = [0, 1, 2, 3, 12, 13, 14, 15]
        ars = (start_ar.replace(months=x).strftime('/m%Y_%m') for x in rng)
        for i, x in enumerate(ars, 1):
            try:
                df = cps_store.select(x, columns=[official_id, 'HRMIS'])
                yield df[df['HRMIS'] == i]['PEEDUCA']
            except KeyError:
                yield None

    dfs = gen_dfs()

    idx = original.major_axis
    df = pd.concat(dfs, axis=1, keys=range(1, 9)).loc[idx]
    original[official_id] = df
    writer(original, panel_store, month_name)

def append_to_earn_store(month, earn_store, panel_store, official_id):
    month_name = month.parts[-1].rstrip('.gz')
    earn_k = arrow.get(month_name,
                       "YYYY_MM").replace(months=15).strftime('/m%Y_%m')
    try:
        original = earn_store.select(earn_k)
    except KeyError:
        print("No panel for {}. Move along.".format(month_name))
        return None
    wp = panel_store.select('/m' + month_name)
    s1, s2 = wp[4][official_id], wp[8][official_id]



def writer(ndframe, store, month_name):
    """
    The panel should already be transposed so that the dtypes align.
    """
    try:
        store.remove('m' + month_name)
        store.append('m' + month_name, ndframe)
    except TypeError:
        ndframe.to_hdf(store, key='m' + month_name, format='f')
    except KeyError:
        store.append('m' + month_name, ndframe)
    print("Added {}".format(month_name))


def generate_months(raw_path, keys, month_to_dd):
    """
    path -> {k} -> ( path )
    """
    months = (month for month in raw_path if
              month_to_dd.get(''.join(month.parts[-1].strip('.gz').split('_')))
              in keys)
    return months


def main():
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    store_path = settings['store_path']
    cps_store = pd.HDFStore(store_path)
    panel_store = pd.HDFStore(settings['panel_store_path'])
    earn_store = pd.HDFStore(settings["earn_store_path"])
    dds = pd.HDFStore(settings['dd_store_path'])
    dd_to_id, official_id = get_id_by_dd()
    keys = dd_to_id.keys()

    month_to_dd = settings['month_to_dd']
    raw_path = pathlib.Path(str(settings['raw_monthly_path']))

    months = generate_months(raw_path, keys, month_to_dd)
    for month in months:
        append_to_cpsstore(month, cps_store, dds, dd_to_id,
                           official_id, settings)
        append_to_panel_store(month, panel_store, cps_store, official_id)
        append_to_earn_store(month, new_id, earn_store, dds, dd_to_id, settings)

if __name__ == '__main__':
    main()
