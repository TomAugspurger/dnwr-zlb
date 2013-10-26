#-----------------------------------------------------------------------------
# -- misc --

def plot_changes(earn_store):
    fig, ax = plt.subplots()

    for k, _ in earn_store.iteritems():
        wp = earn_store.select(k)
        if len(wp.A) == 0:
            print("Empty panel for {}".format(k))
            continue
        elif (wp.A.dtypes == object).all():
            print("Dtype issues panel for {}".format(k))
            continue

        try:
            diff = (wp.A - wp.B)['PRERNWA']
        except KeyError:
            diff = (wp.A - wp.B)['PTERNWA']
        try:
            diff.plot(kind='kde', ax=ax)
            print("Added {}".format(k))
        except TypeError as e:
            print('Skipping {} due to'.format(k, e))

    return fig, ax
