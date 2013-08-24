# From https://github.com/pydata/pandas/blob/master/test_fast.sh

nosetests -A "not slow" . --with-id $*
