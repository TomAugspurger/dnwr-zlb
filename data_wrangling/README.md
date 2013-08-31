Data Wrangling
--------------

As of this writing, most of the work is contained in `cps_wrangling/`.
I would never claim that this directory of code is the cleanest
in the world, but it's sad that it took this many LOC to parse a dataset.
I should note that for many cases, something like IPUMS's cleaned up
CPS will be a much nicer experience.  But they didn't have the monthly
files done yet.

### cps_wrangling

To download the data files, take a look at `monthly_data_downloader.py`.
As the module docstring indicates, it can be used to download the monthly
datafiles from the NBER's website.  Call it with `python monthly_data_downloader.py`
Keep in mind that even compressed, there's a lot of data: It's about 5.57 GB on
my machine. This places the data in the location specified in `settings.txt`

Once you have the compressed data files, we need the *data dictionary* associated
with each month.  These are text file of varying degrees of quality specifying
the layout of the dataset.  The raw CPS files themselves are stored in
fixed-width-format.  The data dictionaries specify which columns go with which
field.

Parsing the data dictionaries was surprisingly difficult (just look at the regex
in `generic_data_dictionary_parser.py`).  Most of methods are pretty well
documented there, so I'll leave it at that.

I parse the data dictionaries and dump the results into a much saner
HDF5 storage format.  To get the data itself into HDF5 format,
we use `make_hdf_store.py`.  The file itself is verbose,
but should be straightforward.  Start with `main()` and work
your way through.

I started by parsing the entire dataset and storing all of the fields in a single
HDF5 store.  75 GB and a full hard drive later, I found out that selecting a subset
of fields is probably a good idea.  Specify the ones you want in `settings.txt`.
