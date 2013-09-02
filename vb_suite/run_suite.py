#!/usr/bin/env python

# Taken from pandas on 2013-08-08
from vbench.api import BenchmarkRunner
from suite import *


def run_process():
    runner = BenchmarkRunner(benchmarks, REPO_PATH, REPO_URL,
                             BUILD, DB_PATH, TMP_DIR, PREPARE,
                             always_clean=True,
                             run_option='eod', start_date=START_DATE,
                             )
    runner.run()

if __name__ == '__main__':
    run_process()
