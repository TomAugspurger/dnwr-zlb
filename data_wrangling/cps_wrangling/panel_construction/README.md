Main Files
----------

* generic\_data\_dictionary\_parser.py
* make\_hdf\_store.py


Notes
-----

Matching
========

May 2004 data dictionary is special for HHNUM:

```
  ** This variable was originally called HRHHID just as the variable beginning in column 1 **
HRHHID2     5      HOUSEHOLD IDENTIFICATION NUMBER (partII)      (71 - 75)

                       EDITED UNIVERSE:   ALL HHLD's IN SAMPLE

                       Part I of this number is found in columns 1-15 of the record.
                       Concatenate this item with Part I for matching forward in time.

                       The component parts of this number are as follows:

                          71-72  Numeric component of the sample number (HRSAMPLE)
                          73-74  Serial suffix-converted to numerics (HRSERSUF)
                          75     Household Number (HUHHNUM)
```

same for August 2005, January 2007, January 2009, May 2012.



There was an error in the data dictionary for apr94_may95.ddf from
http://smpbff2.dsd.census.gov/pub/cps/basic/199404-199505/apr94_may95_dd.txt

Line number 713:
    PEAFNOW      2     ARE YOU NOW IN THE ARMED FORCES      (134 - 136)
should read
    PEAFNOW      2     ARE YOU NOW IN THE ARMED FORCES      (135 - 136)

