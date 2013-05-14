These files accompany the paper:

        " A Note on Longitudinally Matching Current Population Survey Respondents"
        NBER Technical Working Paper No. 247

        Brigitte C. Madrian, University of Chicago and NBER
        Lars John Lefgren, University of Chicago


The following files (all STATA do files) are used in the alorithm to match
CPS respondents in this paper:

        matchcps.do
        educate.do
        educate2.do
        married.do
        married2.do
        multobs.do
        race.do
        year.do

matchcps.do is the master program.  The other files are all called within this program.
This program is written to match consecutive March CPS surveys, but can be easily modified
to match other CPS surveys where matching is possible.

educate.do and educate2.do redefine the education variables in the CPS which are not
consistently coded over time for the time t and time t+1 data to be used in the CPS merge.

married.do and married2.do redefine the marital status variables in the CPS for the
time t and time t+1 data to be used in the CPS merge.

race.do recodes the race variable for the CPS merge.

year.do defines the year for the time t data.

multobs.do deals with the fact that some individuals do not actually have a unique
identifier at a point in time in the CPS.  The working paper does not have much detail
on this problem.  This is discussed further below.


The steps involved in matching CPS surveys are as follows:

1) Make two data extracts, one for time t and one for time t+1,
both of which contain the variables necessary to merge and any additional
variables to be used in a statistical analysis.  For the analysis in this paper,
this was done on a PC using the CPS Utilities data extraction program.
The variables that we extracted for this paper are listed in Appendix Table B1.
For a March-to-March merge, respondents with a MIS of 1-4 should be included in
the time t extract, while respondents with a MIS of 5-8 should be included in
the time t+1 extract.  Respondents with a MIS of 5-8 in time t or with a MIS of 1-4
in time t+1 can be excluded since these respondents are not included in the
sampling frame of both surveys.  For a month-to-month merge (e.g. March-to-April),
respondents with a MIS of 1-3 or 5-7 should be included in the time t extract,
while respondents with a MIS of 2-4 or 6-8 should be included in the time t+1 extract.
The program matchcps.do is written to perform a March-to-March merge of the CPS.
It includes code to restrict the sample to the appropriate MIS ranges in time t and t+1
if extracts not subsetted on the basis of MIS are being used.  matchcps.do
assumes that the CPS extracts are in a directory called
        e:\cpswin\cpsdata\match\marchXX.dta
This will clearly need to be changed to reflect the actual location of the data
being used.

2) Run the program matchcps.do  This program does the following:

        A) Recodes MIS in the time t+1 data to correspond the appropriate value
        that respondents in time t would have if they were in both surveys.
        For March-to-March merges, this subtract 4 from the t+1 MIS.
        For a month-to-month merge, subtract 1 from the t+1 MIS.

        B) Renames other variables that will be used to determine the validity
        of matches (e.g. sex, race, age, etc.) in the two extracts so that
        both the time t and time t+1 values are preserved.

        C) Sorts the time t and t+1 data by MIS, HHID, HHNUM and LINENO.
        For a March 1994-to-March 1995 merge, the data must be sorted by
        MIS, STATE, HHID, HHNUM, and LINENO.  This would be true for some of
        the month-to-month merges in the 1994-1995 time period as well and results
        from the fact that the CPS only assigns unique household identifiers (HHID)
        within state over part of this time period.

        D) Match merges the sorted t and t+1 data extracts on the basis of the
        variables used to sort the data above.

        E) Deals with the problem of potential multiple observations on merged
        individuals (multobs.do).  This is discussed further below.

        F) Applies the criteria discussed in the paper to flag those merged
        observations that do not appear to represent the same individuals.
        The program creates flags that correspond to the following merge criteria
        discussed in the paper:  s|r|a|e, S|R|A|E, s|r|a, S|R|A, s|2, S|2, any2 and ANY2.
        Any of the merge criteria listed in Table 3 or Figure 4 can easily be coded from
        the variables sexdif, racedif, nragedif, ragedif, nredudif, and redudif
        that are defined in the program.  Similarly, other merge criteria relying on
        these or other variables not included in our matchcps.do program could also
        be defined at this point.

3) Determine which merged observations to keep.  The program matchcps.do does not
apply any of the merge criteria discussed in the paper--all naively merged observations
are retained and it is up to the user to determine any further observations to be deleted.


More on the issue of multiple merged observations on the same individual.

One problem that may arise (depending on which CPSs are being merged),
is the presence of multiple post-merge observations with the same identifying
variables (HHID, HHNUM, LINENO).  This occurs because even though
HHID, HHNUM and LINENO are meant to uniquely identify individuals,
in some CPS surveys there are multiple respondents who have the same
HHID, HHNUM and LINENO.  If, for example, there are two individuals with
the same HHID, HHNUM and LINEO in both of the CPS surveys being matched,
we will end of with four merged observations.  Two of the merged observations
will be (potentially) correct, and two of them will be incorrect.  We deal
with this issue in a way designed to preserve as many potentially correct
matches as possible (see the program multobs.do).

First, we create a unique identifier for all respondents in both t (obsno) and t+1 (obsno2)
(these identifiers are not unique across t and t+1, only within t and t+1--
we do not merge on the basis of these identifiers).  After merging the t and t+1
data extracts as described above, we flag the post-merge observations that do not have a
unique value of the t and/or t+1 identifiers that we create.  Among these flagged observations,
we then deleted those that do not have the same sex in t and t+1.  We then flag the remaining
post-merge observations that still did not have a unique value of the t and/or t+1 identifiers
that we created.  Among these flagged observations, we then delete those that do not have the
same race in t and t+1.  We repeat this process, deleting those observations with different
 values of education according to the less-restrictive age criteria and then according to
the more-restrictive age criteria, different values of education according to the
less-restrictive education criteria and then according to the more-restrictive education
criteria, and finally, we delete those flagged observations with differences in their
relationship to household head in time t and t+1.  We then go through and flag any remaining
observations with non-unique identifiers and delete all of these observations.

The table below shows how many post-merge observations there on time t and time t+1
respondents with non-unique individual identifiers for each of the 1980-1998
March-to-March merges that are possible.  It also notes how many observations
with non-unique identifiers remain after we apply each of the deletion criteria just described.
Note that for many of the March-to-March merges, non-unique identifiers are not a problem.
For the March-to-March merges in which there are individuals with non-unique identifiers,
these individuals constitute only a small fraction of the total sample.


Details on Observations with Non-Unique Individual Identifiers When Longitudinally Merging the CPS

                                                Observations with non-unique identifiers remaining
                                                after deletion on the basis of differences in:
                        Number of post-merge
                        observations with
                        non-unique IDs          Sex     Race    Age     Educ.   HH Re.

1980-1981
   1980 respondents             210             40      40      2       2       2
   1981 respondents             261             64      64      8       6       4
1981-1982
   1981 respondents             208             62      60      6       6       2
   1982 respondents             228             34      34      12      10      6
1982-1983
   1982 respondents             298             86      86      10      10      6
   1983 respondents             234             62      60      6       6       2
1983-1984
   1983 respondents             266             76      76      16      14      6
   1984 respondents             343             104     100     22      116     8
1984-1985
   1984 respondents             280             62      60      10      8       2
   1985 respondents             285             66      64      18      18      6
1986-1987
   1986 respondents             264             86      86      16      14      10
   1987 respondents             357             102     92      8       8       4
1987-1988
   1987 respondents             0               0       0       0       0       0
   1988 respondents             318             92      82      14      8       0
1988-1989
   1988 respondents             0               0       0       0       0       0
   1989 respondents             0               0       0       0       0       0
1989-1990
   1989 respondents             0               0       0       0       0       0
   1990 respondents             0               0       0       0       0       0
1990-1991
   1990 respondents             0               0       0       0       0       0
   1991 respondents             0               0       0       0       0       0
1991-1992
   1991 respondents             0               0       0       0       0       0
   1992 respondents             0               0       0       0       0       0
1992-1993
   1992 respondents             0               0       0       0       0       0
   1993 respondents             0               0       0       0       0       0
1993-1994
   1993 respondents             24              10      10      0       0       0
   1994 respondents             0               0       0       0       0       0
1994-1995
   1994 respondents             0               0       0       0       0       0
   1995 respondents             243             149     132     6       6       6
1996-1997
   1996 respondents             0               0       0       0       0       0
   1997 respondents             2               0       0       0       0       0
1997-1998
   1997 respondents             0               0       0       0       0       0
   1998 respondents             0               0       0       0       0       0



