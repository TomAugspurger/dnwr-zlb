import os
import unittest

import pandas as pd
import pandas.util.testing as tm

from ..hdf_wrapper import HDFHandler

class TestHDFWrapper(unittest.TestCase):

    def setUp(self):
        self.fdir = os.path.join('.', 'test_files', 'panel')

        self.settings = {'base_path': './test_files/'}
        months = ['1994_01', '1994_02', '1994_03']
        frequency = 'monthly'
        self.handler = HDFHandler(self.settings, 'panel', months, frequency)

    def test_file_creation(self):
        # _ = self.handler._select_stores(self.handler)
        expected = [os.path.join('test_files', 'panel', x) for x in
                    ('m1994_01.h5', 'm1994_02.h5', 'm1994_03.h5')]
        print(os.listdir('test_files'))

        assert all([os.path.exists(x) for x in expected])

    def test_create_from_list(self):
        settings = {'base_path': './test_files/'}
        months = [['1994_01', '1994_02', '1994_03']]
        frequency = 'Q'
        handler = HDFHandler(settings, kind='panel', months=months,
                             frequency=frequency)
        assert handler.stores.keys() == ['long_1994_Q1']
        handler.close()

    def test_getitem(self):
        result = self.handler['1994_01']
        expected = self.handler.stores['m1994_01']
        assert result is expected

        result = self.handler['m1994_01']
        assert result is expected

    def test_write(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        self.handler.write(df, 'm1994_01', format='f', append=False)
        res = self.handler.stores['m1994_01'].select('m1994_01')
        tm.assert_frame_equal(df, res)

    # def test_select_all(self):
    #     import ipdb; ipdb.set_trace()
    #     h = HDFHandler(self.settings, 'panel', frequency='monthly')
    #     assert len(h.stores) == 3

    def tearDown(self):
        self.handler.close()
        def unlink(subdir):
            for f in os.listdir(os.path.join('test_files', subdir)):
                file_path = os.path.join(self.fdir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except OSError:
                    pass
                except Exception, e:
                    print e
            os.rmdir(os.path.join('test_files', subdir))
        unlink('panel')
        os.rmdir('test_files')

