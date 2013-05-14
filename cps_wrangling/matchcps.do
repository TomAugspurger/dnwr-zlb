* PROGRAM:  matchcps.do
* DATE:  December 3, 1999

*********************************************************
* THIS PROGRAM MERGES TWO CONSECUTIVE MARCH CPS SURVEYS *
*********************************************************

set more 1
log using matchcps, replace

* Load programs to recode variables that are            *
* inconsistently coded over time.  These programs will  *
* be called later on in the program                     *

quietly do educate
quietly do educate2
quietly do married
quietly do married2


*********************************************************
* Define program to match cps surveys                   *
*********************************************************

program define matchcps

  global year "`1'"

  local tplus1=${year}+1


* STEP 1:  Load data for individuals in time t+1

  clear
  display "Reading in t+1 Data"

  use e:\cpswin\cpsdata\match\mar`tplus1' if mis>=5

    replace mis=mis-4
    assert mis>=1 & mis<=4

  gen byte year=`tplus1'
  quietly do race
  quietly married 2
  if `tplus1'<92 {
    quietly educate2
  }

* Rename t+1 variables

  ren age age2
  capture ren edu edu2
  capture ren grdatn grdatn2
  ren race race2
  ren sex sex2
  ren migsam migsam2
  ren married married2
  ren _relhd _relhd2
  ren wgt wgt2

  label var age2 "Age in time t+1"
  capture label var edu2 "Highest grade completed in time t+1"
  capture label var grdatn2 "Highest grade attended in time t+1"
  label var race2 "Race in time t+1"
  label var sex2 "Sex in time t+1"
  label var migsam2 "Lived in same house 12 months ago"
  label var married2 "Marital status time t+1"
  label var _relhd2 "Relationship to HH head t+1"
  label var wgt2 "Weight time t+1"

* Sort and save t+1 data

if ${year}==94 {
  display "Sorting and Saving t+1 Data"
  sort mis state hhid hhnum lineno
  gen obsno2=_n
  save e:\matchcps\match${year}B, replace
}

else if ${year}~=94 {
  display "Sorting and Saving t+1 data"
  sort mis hhid hhnum lineno
  gen obsno2=_n
  save e:\matchcps\match${year}B, replace
}

clear


* STEP 2:  Load data for individuals in time t

  display "Reading in individual data"
  use e:\cpswin\cpsdata\match\mar${year} if mis<=4
  tab mis

  quietly do year
  quietly do race
  quietly married
  if ${year}<92 {
    quietly educate
  }


* STEP 3:  Merge t+1 data to t data

if ${year}==94 {
  sort mis state hhid hhnum lineno
  gen obsno=_n
  display "Merging t+1 data"
  merge mis state hhid hhnum lineno using e:\matchcps\match${year}B
}

else if ${year}~=94 {
  sort mis hhid hhnum lineno
  gen obsno=_n
  display "Merging t+1 data"
  merge mis hhid hhnum lineno using e:\matchcps\match${year}B
}


* STEP 4:  Define merge quality variables

  gen byte sexdif=sex==sex2

  gen byte racedif=race==race2

  gen byte relhddif=_relhd==_relhd2

  gen byte mardif=married==married2

  gen byte agedif=age2-age

  gen byte ragedif=(agedif==0 | agedif==1 | agedif==2)
  gen byte nragedif=(agedif==-1 | agedif==0 | agedif==1 | agedif==2 | agedif==3)

if ${year}<=90 {
  gen byte edudif=edu2-edu

  gen byte redudif=(edudif==0 | edudif==1)
  gen byte nredudif=(edudif==-1 | edudif==0 | edudif==1 | edudif==2)

}

if ${year}==91 {
  gen byte redudif=0
    replace redudif=1 if grdatn2==31 & edu==0
    replace redudif=1 if grdatn2==32 & (edu>=0 & edu<=3)
    replace redudif=1 if grdatn2==33 & (edu>=4 & edu<=6)
    replace redudif=1 if grdatn2==34 & (edu>=6 & edu<=8)
    replace redudif=1 if grdatn2==35 & (edu==8 | edu==9)
    replace redudif=1 if grdatn2==36 & (edu==9 | edu==10)
    replace redudif=1 if grdatn2==37 & (edu==10 | edu==11)
    replace redudif=1 if grdatn2==38 & (edu==11 | edu==12)
    replace redudif=1 if grdatn2==39 & (edu==11 | edu==12)
    replace redudif=1 if grdatn2==40 & (edu>=11 & edu<=15)
    replace redudif=1 if grdatn2==41 & (edu>=11 & edu<=15)
    replace redudif=1 if grdatn2==42 & (edu>=11 & edu<=15)
    replace redudif=1 if grdatn2==43 & (edu>=15 & edu<=16)
    replace redudif=1 if grdatn2==44 & (edu>=16 & edu<=18)
    replace redudif=1 if grdatn2==45 & (edu>=16 & edu<=18)
    replace redudif=1 if grdatn2==46 & (edu>=17 & edu<=18)

  gen byte nredudif=1
    replace nredudif=0 if grdatn2==31 & edu>2
    replace nredudif=0 if grdatn2==32 & edu>5
    replace nredudif=0 if grdatn2==33 & (edu<3 | edu>7)
    replace nredudif=0 if grdatn2==34 & (edu<5 | edu>9)
    replace nredudif=0 if grdatn2==35 & (edu<7 | edu>10)
    replace nredudif=0 if grdatn2==36 & (edu<8 | edu>11)
    replace nredudif=0 if grdatn2==37 & (edu<9 | edu>12)
    replace nredudif=0 if grdatn2==38 & (edu<10 | edu>13)
    replace nredudif=0 if grdatn2==39 & (edu<10 | edu>13)
    replace nredudif=0 if grdatn2==40 & (edu<11 | edu>16)
    replace nredudif=0 if grdatn2==41 & (edu<11 | edu>16)
    replace nredudif=0 if grdatn2==42 & (edu<11 | edu>16)
    replace nredudif=0 if grdatn2==43 & (edu<14 | edu>18)
    replace nredudif=0 if grdatn2==44 & edu<16
    replace nredudif=0 if grdatn2==45 & edu<16
    replace nredudif=0 if grdatn2==46 & edu<17
}

else if ${year}>=92 {
  gen byte redudif=1
    replace redudif=0 if grdatn==31 & (grdatn2>32 | grdatn2<31)
    replace redudif=0 if grdatn==32 & (grdatn2>33 | grdatn2<32)
    replace redudif=0 if grdatn==33 & (grdatn2<33 | grdatn2>34)
    replace redudif=0 if grdatn==34 & (grdatn2<34 | grdatn2>35)
    replace redudif=0 if grdatn==35 & (grdatn2<35 | grdatn2>36)
    replace redudif=0 if grdatn==36 & (grdatn2<36 | grdatn2>37)
    replace redudif=0 if grdatn==37 & (grdatn2<37 | grdatn2>39)
    replace redudif=0 if grdatn==38 & (grdatn2<38 | grdatn2>42)
    replace redudif=0 if grdatn==39 & (grdatn2<39 | grdatn2>42)
    replace redudif=0 if grdatn==40 & (grdatn2<40 | grdatn2>43)
    replace redudif=0 if grdatn==41 & (grdatn2<41 | grdatn2>43)
    replace redudif=0 if grdatn==42 & (grdatn2<41 | grdatn2>43)
    replace redudif=0 if grdatn==43 & (grdatn2<43)
    replace redudif=0 if grdatn==44 & (grdatn2<44)
    replace redudif=0 if grdatn==45 & (grdatn2<44)
    replace redudif=0 if grdatn==46 & (grdatn2<44)

  gen byte nredudif=1
    replace nredudif=0 if grdatn==31 & grdatn2>32
    replace nredudif=0 if grdatn==32 & grdatn2>33
    replace nredudif=0 if grdatn==33 & (grdatn2<32 | grdatn2>34)
    replace nredudif=0 if grdatn==34 & (grdatn2<33 | grdatn2>36)
    replace nredudif=0 if grdatn==35 & (grdatn2<34 | grdatn2>37)
    replace nredudif=0 if grdatn==36 & (grdatn2<35 | grdatn2>38)
    replace nredudif=0 if grdatn==37 & (grdatn2<36 | grdatn2>40)
    replace nredudif=0 if grdatn==38 & (grdatn2<37 | grdatn2>42)
    replace nredudif=0 if grdatn==39 & (grdatn2<37 | grdatn2>42)
    replace nredudif=0 if grdatn==40 & (grdatn2<38 | grdatn2>43)
    replace nredudif=0 if grdatn==41 & (grdatn2<38 | grdatn2>43)
    replace nredudif=0 if grdatn==42 & (grdatn2<38 | grdatn2>43)
    replace nredudif=0 if grdatn==43 & (grdatn2<40 | grdatn2>46)
    replace nredudif=0 if grdatn==44 & (grdatn2<43 | grdatn2>46)
    replace nredudif=0 if grdatn==45 & (grdatn2<43 | grdatn2>46)
    replace nredudif=0 if grdatn==46 & (grdatn2<43 | grdatn2>46)
}


* Deal with duplicate observations

  do multobs

* Assess merge

display "Assess merge"

  sort hhid hhnum lineno
  assert lineno~=lineno[_n-1] if hhid==hhid[_n-1] & hhnum==hhnum[_n-1] & _merge==3, rc0

  tab _merge

  replace migsam2=2 if migsam2==3

  tab _merge migsam2, row, if _merge>=2

gen byte s_r_a  =(sexdif==0 | racedif==0 | ragedif==0)
gen byte s_r_a_e=(sexdif==0 | racedif==0 | ragedif==0 | redudif==0)
gen byte any2 =0
  replace any2=1 if (sexdif==0 & racedif==0)
  replace any2=1 if (sexdif==0 & ragedif==0)
  replace any2=1 if (sexdif==0 & redudif==0)
  replace any2=1 if (racedif==0 & ragedif==0)
  replace any2=1 if (racedif==0 & redudif==0)
  replace any2=1 if (ragedif==0 & redudif==0)
gen byte s_2    =(sexdif==0 | (racedif==0 & ragedif==0) | (racedif==0 & redudif==0) | (ragedif==0 & redudif==0))

gen byte S_R_A  =(sexdif==0 | racedif==0 | nragedif==0)
gen byte S_R_A_E=(sexdif==0 | racedif==0 | nragedif==0 | nredudif==0)
gen byte ANY2   =0
  replace ANY2=1 if (sexdif==0 & racedif==0)
  replace ANY2=1 if (sexdif==0 & nragedif==0)
  replace ANY2=1 if (sexdif==0 & nredudif==0)
  replace ANY2=1 if (racedif==0 & nragedif==0)
  replace ANY2=1 if (racedif==0 & nredudif==0)
  replace ANY2=1 if (nragedif==0 & nredudif==0)
gen byte S_2    =(sexdif==0 | (racedif==0 & nragedif==0) | (racedif==0 & nredudif==0) | (nragedif==0 & nredudif==0))


  tab _merge if _merge>1
  tab s_r_a if _merge>1
  tab s_r_a_e if _merge>1
  tab any2 if _merge>1
  tab s_2 if _merge>1
  tab S_R_A if _merge>1
  tab S_R_A_E if _merge>1
  tab ANY2 if _merge>1
  tab S_2 if _merge>1

  tab _merge migsam2, row
  tab s_r_a migsam2, row
  tab s_r_a_e migsam2, row
  tab any2 migsam2, row
  tab s_2 migsam2, row
  tab S_R_A migsam2, row
  tab S_R_A_E migsam2, row
  tab ANY2 migsam2, row
  tab S_2 migsam2, row

drop if _merge==2

* Sort and save

display "Sorting and saving merged data"
sort hhid hhnum lineno
save e:\matchcps\mrgd${year}, replace

end

*************************************************
* Run program to match March CPS surveys        *
*************************************************

  matchcps 80
  matchcps 81
  matchcps 82
  matchcps 83
  matchcps 84
  matchcps 86
  matchcps 87
  matchcps 88
  matchcps 89
  matchcps 90
  matchcps 91
  matchcps 92
  matchcps 93
  matchcps 94
  matchcps 96
  matchcps 97

log close
