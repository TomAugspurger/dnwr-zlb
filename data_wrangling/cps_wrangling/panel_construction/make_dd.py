import pathlib

from generic_data_dictionary_parser import Parser
#-----------------------------------------------------------------------------


def run_parse_dictionaries(month_path, overwrite=False):
    data_dictionaries = (file_ for file_ in month_path if
                         file_.parts[-1].endswith('ddf') and
                         file_.parts[-1] != 'cpsrwdec07.ddf')
    for file_ in data_dictionaries:
        str_file = str(file_) + '\n'
        if not overwrite:
            try:
                f = open('processed.txt', 'r')
                f.close()
            except IOError:
                with open('processed.txt', 'w'):
                    pass

            with open('processed.txt', 'r') as f:
                if str_file in f.readlines():
                    print('Skipped {}'.format(file_))
                    continue
        if file_.parts[-1] in ['cpsbaug05.ddf', 'cpsbjan07.ddf',
                               'cpsbjan09.ddf', 'cpsbjan10.ddf',
                               'cpsbmay04.ddf', 'cpsbmay12.ddf',
                               'cpsbjan03.ddf', 'cpsbsep95.ddf',
                               'cpsb_apr94_may95.ddf',
                               'cpsb_jan94_mar94.ddf',
                               'cpsb_jun95_aug95.ddf']:
            style = 'aug2005'
        elif file_.parts[-1] in ['cpsbjan98.ddf']:
            style = 'jan1998'
        else:
            style = None
        try:
            kls = Parser(str(file_), outfile, style=style)
            kls.run()
            kls.writer()
            print('Added {}'.format(file_))
            str_file = str(file_) + '\n'
            with open('processed.txt', 'a') as f:
                f.write(str_file)

        except Exception as e:
            with open(logfile, 'a') as f:
                f.write(str(file_) + '\n')
                print('Failed on {}'.format(file_))
            raise e


def main():
    import json

    settings = json.load(open('settings.txt'))
    base_path = settings['base_path']
    monthly_base = settings['raw_monthly_path']
    month_path = pathlib.Path(monthly_base)
    outfile = base_path + 'cps_store.h5'
    logfile = 'fail_log.txt'

    run_parse_dictionaries(month_path)

if __name__ == '__main__':
    main()
