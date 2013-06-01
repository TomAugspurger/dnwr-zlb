import pathlib

from generic_data_dictionary_parser import Parser
#-----------------------------------------------------------------------------
base_path = '/Volumes/HDD/Users/tom/DataStorage/CPS/'
monthly_base = base_path + 'monthly/'
month_path = pathlib.Path(monthly_base)

outfile = base_path + 'cps_store.h5'

logfile = 'fail_log.txt'


def run_parse_dictionaries(month_path, overwrite=False):
    data_dictionaries = (file_ for file_ in month_path if
                         file_.parts[-1].endswith('ddf'))
    for file_ in data_dictionaries:
        str_file = str(file_) + '\n'
        if not overwrite:
            with open('processed.txt', 'r') as f:
                if str_file in f.readlines():
                    print('Skipped {}'.format(file_))
                    continue
        if file_.parts[-1] in ['cpsbaug05.ddf', 'cpsbjan07.ddf',
                               'cpsbjan09.ddf', 'cpsbjan10.ddf',
                               'cpsbmay04.ddf', 'cpsbmay12.ddf']:
            style = 'aug2005'
        elif file_.parts[-1] in ['cpsbjan98.ddf']:
            style = 'jan1998'
        else:
            style = None
        try:
            kls = Parser(str(file_), outfile, style=style)
            kls.run()
            kls.writer
            print('Added {}'.format(file_))
            str_file = str(file_) + '\n'
            with open('processed.txt', 'a') as f:
                f.write(str_file)

        except Exception as e:
            with open(logfile, 'a') as f:
                f.write(str(file_) + '\n')
                print('Failed on {}'.format(file_))
            raise e

if __name__ == '__main__':
    import sys
    which = sys.argv[1]

    if which == 'dd':
        run_parse_dictionaries(month_path)

    else:
        print('Acceptable entries are {}'.format('dd'))
