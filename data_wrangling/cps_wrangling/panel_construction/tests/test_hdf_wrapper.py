import os
import unittest

import pandas as pd
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.panel_construction.hdf_wrapper import HDFHandler


class TestHDFWrapper(unittest.TestCase):

    def setUp(self):
        self.fdir = os.path.join('.', 'test_files', 'panel')
        months = ['1994_01', '1994_02', '1994_03']
        frequency = 'monthly'
        self.handler = HDFHandler('./test_files', 'panel', months, frequency)

    def test_file_creation(self):
        # _ = self.handler._select_stores(self.handler)
        expected = [os.path.join('test_files', 'panel', x) for x in
                    ('m1994_01.h5', 'm1994_02.h5', 'm1994_03.h5')]
        print(os.listdir('test_files'))

        assert all([os.path.exists(x) for x in expected])

    def test_create_from_list(self):
        months = [['1994_01', '1994_02', '1994_03']]
        frequency = 'Q'
        handler = HDFHandler('./test_files', kind='panel', months=months,
                             frequency=frequency)
        self.assertEqual(handler.stores.keys(), ['long_1994_Q1'])
        handler.close()

    def test_getitem(self):
        result = self.handler['m1994_01']
        expected = self.handler.stores['m1994_01']
        self.assertIs(result, expected)

    def test_write(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        self.handler.write(df, 'm1994_01', format='f', append=False)
        res = self.handler.stores['m1994_01'].select('m1994_01')
        tm.assert_frame_equal(df, res)

    def test_iter(self):
        result = [x for x in self.handler]
        expected = ['m1994_01', 'm1994_02', 'm1994_03']
        self.assertEqual(result, expected)

    # def test_select_all(self):
    #     import ipdb; ipdb.set_trace()
    #     h = HDFHandler(self.settings, 'panel', frequency='monthly')
    #     assert len(h.stores) == 3

    def test_sanitize_key(self):
        result = self.handler._sanitize_key('1994-01')
        expected = 'm1994_01'
        self.assertEqual(result, expected)

        result = self.handler._sanitize_key('1994_01')
        self.assertEqual(result, expected)

        result = self.handler._sanitize_key('m1994-01')
        self.assertEqual(result, expected)

        # With a different pre
        settings = {'base_path': './test_files/'}
        months = [['1994_01', '1994_02', '1994_03']]
        frequency = 'Q'
        # handler = HDFHandler(settings, kind='long', months=months,
        #                      frequency=frequency)
        # result = handler._sanitize_key('long_1996_Q1')
        # print(handler.pre)
        # expected = "long_1996_Q1"
        # self.assertEqual(result, expected)

    def test_iteritems(self):
        g = self.handler.iteritems()
        name, value = next(g)
        self.assertEqual(name, 'm1994_01')
        self.assertIs(value, None)

    # getting a `is not a regular file` error.
    def test_from_directory(self):
        # setup should have created some.
        os.mkdir('from_dir')
        with open(os.path.join('from_dir', 'file5.h5'), 'w'):
            pass
        try:
            handler = HDFHandler.from_directory('from_dir', kind='M')
            self.assertEqual(len(handler.stores), 1)
            handler.close()
        except Exception as e:
            print(e)
        finally:
            os.remove(os.path.join('from_dir', 'file5.h5'))
            os.removedirs('from_dir')

    # def test_map(self):
    #     res = self.handler.map(sum)

    def test_apply(self):
        # TOOD: make multiindex with a stamp.  with names
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'a', 'b', 'b']})

        idx1 = lambda year: pd.to_datetime([year] * 4)
        idx2 = ('L1', 'L2', 'L1', 'L2')
        df.index = pd.MultiIndex.from_tuples(zip(idx1('1994-01-01'), idx2),
                                             names=['stamp', 'l2'])
        self.handler.write(df, 'm1994_01', format='table', append=False)

        df.index = pd.MultiIndex.from_tuples(zip(idx1('1994-02-01'), idx2),
                                             names=['stamp', 'l2'])
        self.handler.write(df, 'm1994_02', format='table', append=False)
        df.index = pd.MultiIndex.from_tuples(zip(idx1('1994-03-01'), idx2),
                                             names=['stamp', 'l2'])
        self.handler.write(df, 'm1994_03', format='table', append=False)

        # agg, groupby=None, level='stamp'
        result = self.handler.apply('mean', level='stamp', selector='A')
        expected = pd.Series([2.5, 2.5, 2.5], name='A',
                             index=pd.to_datetime(['1994-01-01',
                                                   '1994-02-01',
                                                   '1994-03-01']))
        tm.assert_series_equal(result, expected)

        # agg, groupby=None, level='stamp', list selector
        result = self.handler.apply('mean', level='stamp', selector=['A'])
        expected = pd.DataFrame(expected)
        expected.index.names = ['stamp']
        tm.assert_frame_equal(result, expected)

        # with select kwargs
        result = self.handler.apply('mean', level='stamp', selector=['A'],
                                    select_kwargs={'columns': ['A']})
        expected = pd.DataFrame(expected)
        expected.index.names = ['stamp']
        tm.assert_frame_equal(result, expected)

        # agg, groupby='B', level='None'
        # will fail
        # result = self.handler.apply('mean', groupby='B', selector=['A'])
        # expected = pd.Series([1.5, 3.5], name='A', index=['a', 'b'])
        # tm.assert_series_equal(result, expected)

    def test_select(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        self.handler.write(df, 'm1994_01', format='table', append=False)

        # actual test
        result = self.handler.stores['m1994_01'].select('m1994_01')
        tm.assert_frame_equal(result, df)

        result = self.handler.stores['m1994_01'].select('m1994_01', columns=['A'])
        tm.assert_frame_equal(result, df[['A']])

    def tearDown(self):
        self.handler.close()

        def unlink(subdir):
            for f in os.listdir(os.path.join('test_files', subdir)):
                file_path = os.path.join('.', 'test_files', subdir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except OSError:
                    pass
                except Exception as e:
                    print(e)

            os.rmdir(os.path.join('test_files', subdir))
        unlink('panel')
        # unlink('long')
        os.rmdir('test_files')
