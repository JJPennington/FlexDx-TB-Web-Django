#!/usr/bin/env python
import os,sys,json

#model imports
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import fsolve
#import csv
# for optimization
from functools import wraps

#Model Code
###############################################################
# 3. DEFINE PARAMETERS
###############################################################
# INSERTING ALL OF BELOW IN FUNCTION runmodel()

# generic memoization for optimization 
def memoize(f):
    cache = {}
    @wraps(f)
    def wrap(*arg):
        if arg not in cache:
            cache[arg] = f(*arg)
        return cache[arg]
    return wrap

#def readparz(filename):
#    #print 'Reading data from ' + filename + '...'
#    store = np.genfromtxt(filename,delimiter=',',skip_header=2,dtype=['a32','a32','a32'],filling_values='-0') # a numpy array
#    for i in range(len(store)):
#        store[i][0] = store[i][0].strip()
#        store[i][1] = store[i][1].strip()
#        store[i][2] = store[i][2].lstrip()
#        #print store[i]
#        exec 'global ' + store[i][0] # declare as global
#        exec store[i][0]+'='+store[i][1] in globals() # assign value

def runmodel( target_inc, target_hiv, target_mdr, cost, 
              sm_cost, gxp_cost, cx_cost, mods_cost, dst_cost, sd_cost, sdgxp_cost, drug2_cost, drug3_cost, outpt_cost, 
              #JP added new cost parameters
              int_select, as_data, tmp_path ): #JP added JSON data and tmp path to parameters

    intopt = 0
    drug1_cost = cost

    # In low-incidence scenario, fit initially to target incidence of 50:
    target_inc2 = target_inc 
    if target_inc2<50:
        target_inc=50.

    # Parameters that will later be fit to user inputs above.
    # The actual values inputted here are NOT used by the program.
    beta_s = 8.0             # transmission rate
    fit_m = 0.707            # relative fitness of MDR-TB
    fit_i = 1.-(1.-fit_m)*0.25 # relative fitness of INHr-TB
    hiv_inc = 0.00064        # HIV incidence

    # quantities that will be needed for later calculation.
    force = 0.               # force of infection, initiate at zero
    tb_num = 100.            # number of compartments = 100
    mortv = np.zeros(tb_num) # mortality vector (to maintain stable pop)

    # Keep the population size at 100,000 for understanding of results.
    pop_size = 100000.       # population size = 100,000 (for std rates)

    ###################################################################################
    ### The following parameters can all be modified according to user preferences: ###
    ###################################################################################

    # in emerging-MDR scenario, change "target_mdr" below to the desired
    # MDR prevalence among new cases at the END of 5 years
    #   (input value will define the prevalence at the beginning of 5 years)
    # INPUT AS A PROPORTION (e.g., 0.1 = 10% prevalence)
    # Thus, for example, if you want a scenario that goes from MDR prevalence
    #   of 4.0% to 6.0% over 5 years, input "4.0" when asked in the interface,
    #   and change the line below to read, "target_mdr2 = 0.06".
    target_mdr2 = target_mdr

    react       =0.0005 # HIV-negative reactivation rate
    react_h     =0.05 # HIV-positive reactivation rate
    relbeta_sn  =0.22 #relative infectiousness of smear-neg TB: incl extrapulm
    rapid       =0.14 # proportion of infections w/ primary progression
    rapid_h     =0.47 # primary progression proportion: HIV+
    prot        =0.79 # reduction in rate of reinfection if LTBI & HIV-neg
    cure_sn     =0.267 #""self-cure"" rate: smear-neg (20% CFR at 3 yrs)"
    cure_sp     =0.1 # self-cure rate: smear-pos (70% CFR at 3 yrs)
    mort        =1.0/55. #background mortality 
    mort_h      =0.05 #mortality/prevalence of HIV
    mort_sn     =0.0667# mortality rate: smear-negative: 20% CFR at 3 yrs
    mort_sp     =0.233 # mortality rate: smear-positive: 70% CFR at 3 yrs
    mort_tbh    =1.01 # assume 1-year duration of TB before uniform death if HIV+
    prop_inf    =0.63 # proportion of incident cases that are smear-pos
    prop_inf_h  =0.5 # proportion smear-pos if HIV-pos
    fail_s      =0.04 # probability of failure or relapse: DS-TB
    fail_rt     =0.04 # probability of failure or relapse: retreatment DS-TB
    fail_i1     =0.21 # prob of failure or relapse: INHr treated with 1st-line
    fail_i2     =0.16 # prob of failure or relapse: INHr treated with streptomycin
    fail_m1     =0.5 # prob of failure or relapse: MDR-TB treated with 1st-line
    fail_m2     =28./93. # prob of failure or relapse: MDR-TB treated w/ 2nd-line
    predx_dur   =0.75 # duration of infectiousness before seeking care: HIV-
    predx_dur_h =1./12 #duration of infectiousness before seeking care: HIV+
    acq_s       =0.003/0.04 # proportion of failed treatments acquiring resistance: DS-TB
    acq_mdr     =0.33 # proportion of acq resi among DS-TB that is MDR (vs. INH-mono)
    acq_i1      =0.045/0.21 # proportion of failed treatments acquiring resistance: INHr tx w/ 1st-line
    acq_i2      =0.017/0.16 # proportion of failed treatments acquiring resistance: INHr tx w/ 2nd-line
    defvfail_s  =6./7. # proportion of inappropriate tx's that are default (vs. failure): DS-TB
    defvfail_i  =2./3. # proportion of inapp tx's that are default: INHr
    defvfail_m  =11./25. # proportion of inapp tx's that are default: MDR
    dx_rate     =5.001 # diagnostic rate: 5 diagnostic attempts per year
    ltfu        =0.15 # 15% initial default proportion: smear/GXP
    ltfu_cx     =0.25 # 25% initial default proportion: culture
    emp_tx      =0.25 # 25% of pts treated empirically
    cx_sens     =0.85 # Sensitivity of single culture
    gxp_sens    =0.72 # Sensitivity of single GXP
    sm_spec     =0.98 # specificity of smear
    cx_spec     =0.98 # specificity of culture
    gxp_spec    =0.98 #, specificity of GXP
    cxr_sens    =0.98 # sensitivity of MODS for RIF resistance
    cxi_sens    =0.98 # sensitivity of MODS for INH resistance
    cxr_spec    =0.994 # specificity of MODS for RIF
    cxi_spec    =0.958 # specificity of MODS for INH
    gxr_sens    =0.944 # sensitivity of GXP for RIF
    gxr_spec    =0.983 # specificity of GXP for RIF
    t_smgxp     =7.0/365 # delay from smear/GXP result to tx: 7 days
    t_cx        =30.0/365 #delay from culture result to tx: 30 days
    t_cxr       =30.0/365 # time to culture-based DST is the same (30 days)
    dur_fail    =6.001 # number of months until failing cases are re-evaluated
    prop_cough  =0.01 # proportion of people w/ non-TB cough eval'd per yr
#    fn = os.path.join(os.path.dirname(__file__),'/home/verloc/dev/django-dev/flexdxtb/media/parameters.csv')
#    readparz(fn)
#    drug2_cost = 2.*drug1_cost  # Assume that cat2 costs twice as much as cat1
#    drug3_cost = 10.*drug1_cost # Assume that 2nd-line/MDR costs 10 times as much
#    outpt_cost = drug1_cost/50. # Outpatient visit costs 2% of a full course of therapy

    ##################################
    ### End modifiable parameters. ###
    ##################################


    ##########################################################
    # 4. DEFINE INITIAL POPULATION
    ##########################################################
    # The values in each of the initial compartments were manually determined
    #   to provide a reasonable starting population -- needed for the equation solver to
    #   converge on appropriate roots later.

    I = np.zeros(100)           # initial population vector
    inc_fact = target_inc/100.  # factor based on inputted incidence
    lat_prob = 0.5*inc_fact/(1+0.5*inc_fact) # odds of latency relate to this

    mdr_prob = target_mdr
    inh_prob = target_mdr*2.3

    hiv_prob = target_hiv*0.01
    retx_prob = 0.2

    # I[0] through I[99] are the initial "guesses" at each compartment size.
    I[1] = 68900*lat_prob*(1-inh_prob-mdr_prob)*(1-hiv_prob)
    I[2] = 100103*lat_prob*(inh_prob)*(1-hiv_prob)
    I[3] = 72300*lat_prob*(mdr_prob)*(1-hiv_prob)
    I[4:7] = I[1:4]*0.0026    
    smpos = np.array([7,9,11])
    smneg = np.array([8,10,12])
    I[smpos]=I[1:4]*0.00046
    I[smneg]=I[1:4]*0.00089
    I[smpos+6] = I[1:4]*0.00003
    I[smneg+6] = I[1:4]*0.000015
    I[19]=I[1]*0.000035
    I[20]=I[1]*0.000019
    I[21]=I[2]*0.00017
    I[22]=I[2]*0.0001
    I[23]=I[3]*0.0006
    I[24]=I[3]*0.0003
    I[25] = 103000*(1-lat_prob+inh_prob+mdr_prob)*hiv_prob
    I[26:29] = I[1:4]* hiv_prob*0.97/((1-hiv_prob)*0.996)*0.7
    I[29:32] = I[26:29]*0.0053
    I[smpos+25] = I[26:29]*0.00576
    I[smneg+25] = I[26:29]*0.0135
    I[smpos+31] = I[26:29]*0.00035
    I[smneg+31] = I[26:29]*0.00025
    I[19+25]=I[1+25]*0.0003
    I[20+25]=I[1+25]*0.0002
    I[21+25]=I[2+25]*0.0015
    I[22+25]=I[2+25]*0.001
    I[23+25]=I[3+25]*0.0052
    I[24+25]=I[3+25]*0.0034
    I[50] = (1-lat_prob)*(prop_cough)*(1-hiv_prob)*100000.
    I[51:54] = I[1:4]* retx_prob/(1.-retx_prob)*0.7
    I[54:57] = I[51:54]*0.00064
    I[57] = I[51]*0.00018
    I[58] = I[51]*0.00034
    I[59] = I[52]*0.00042
    I[60] = I[52]*0.00077
    I[61] = I[53]*0.0011
    I[62] = I[53]*0.002
    I[63] = I[51]*0.000015
    I[64] = I[51]*0.000008
    I[65] = I[52]*0.00003
    I[66] = I[52]*0.000015
    I[67] = I[53]*0.00012
    I[68] = I[53]*0.000023
    I[19+50]=I[1+50]*0.000015
    I[20+50]=I[1+50]*0.000008
    I[21+50]=I[2+50]*0.0002
    I[22+50]=I[2+50]*0.0001
    I[23+50]=I[3+50]*0.0021
    I[24+50]=I[3+50]*0.0012
    I[75] = (1-lat_prob)*(prop_cough)*(hiv_prob)*100000.
    I[76:79] = I[26:29]* retx_prob/(1.-retx_prob)/0.7/0.7
    I[79:82] = I[76:79]*0.0043
    I[82] = I[76]*0.0048
    I[83] = I[76]*0.011
    I[84] = I[77]*0.0055
    I[85] = I[77]*0.012
    I[86] = I[78]*0.0075
    I[87] = I[78]*0.015
    I[88] = I[76]*0.00038
    I[89] = I[76]*0.00025
    I[90] = I[77]*0.00039
    I[91] = I[77]*0.00024
    I[92] = I[78]*0.0008
    I[93] = I[78]*0.00018
    I[94]=I[76]*0.00025
    I[95]=I[76]*0.00016
    I[96]=I[77]*0.0014
    I[97]=I[77]*0.00086
    I[98]=I[78]*0.0081
    I[99]=I[78]*0.0052

    I[0] = 100000.-np.sum(I[:])
    ##########################################################
    # 5. DEFINE CALCULATION ROUTINES
    ##########################################################

    # This function encodes the decision tree.
    # It takes every combination of:
    # intervention, smear/infectious status, HIV status, drug resistance status, and new/retreatment status,
    #
    # and then calculates:
    # 0. Rate of successful treatment (dx rate * prob of successful tx)
    # 1. Delay from sending diagnostic test to initiation of treatment
    # 2. Rate of treatment that will fail (dx rate * prob of unsuccessful tx)
    # 3. Cost of diagnosis
    # 4. Cost of treatment if successful
    # 5. Cost of treatment if unsuccessful
    @memoize
    def txrate(interv,smear,dr,retx,hiv):
        tx = 0.
        tx_inapp = 0.
        delay = 1.
        cost_dx = 0.
        cost_tx = 0.
        cost_inapp = 0.
        # smear, with culture for retreatment smear-positives only
        if interv==0 or (interv==7 and hiv==0):
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                if dr==1:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:                    
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                if dr==2:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*(1-cxr_sens)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*cxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost

        # smear, with automatic culture for retreatment suspects
        if interv==1:
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu_cx)*cx_sens*cxr_spec+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + cx_sens*dst_cost
                if dr==1:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu_cx)*cx_sens*cxr_spec+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + cx_sens*dst_cost
                if dr==2:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu_cx)*cx_sens*(1-cxr_sens)+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(cxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + cx_sens*dst_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*(1-cxr_sens)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*cxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost

        # GXP
        if interv==2 or (interv==7 and hiv==1):
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu)*gxp_sens*gxr_spec + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*gxp_sens*gxr_spec + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu)*gxp_sens*gxr_spec + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*gxp_sens*gxr_spec + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)*gxp_sens*(1-gxr_sens) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(gxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*gxp_sens*(1-gxr_sens) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(gxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)*(1-gxr_sens) + ltfu*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu)*(1-gxr_sens) + ltfu*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost
        # MODS/TLA
        if interv==3:
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu_cx)*cx_sens*cxr_spec+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu_cx)*cx_sens*cxr_spec+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu_cx)*cx_sens*cxr_spec+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_i1)+(fail_i1-fail_i2)*(1-ltfu_cx)*cx_sens*cxr_spec*cxi_sens/p_1st     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx)+ \
                                 (t_cxr-t_cx)*((fail_i1-fail_i2)*(1-ltfu_cx)*cx_sens*cxr_spec*cxi_sens/p_1st))/ \
                                 (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug1_cost + (drug2_cost-drug1_cost)*((1-ltfu_cx)*cx_sens*cxr_spec*cxi_sens/p_1st)      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu_cx)*cx_sens*cxr_spec+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))       # time before 1st-line tx is started
                        c_1st = drug2_cost       # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu_cx)*cx_sens*(1-cxr_sens)+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(cxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                    if retx==1:
                        p_1st = (1-ltfu_cx)*cx_sens*(1-cxr_sens)+emp_tx*(ltfu_cx+(1-ltfu_cx)*(1-cx_sens)) # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = (t_smgxp*emp_tx+t_cx*(1-emp_tx)*cx_sens*(1-ltfu_cx))/ \
                                (emp_tx + (1-emp_tx)*cx_sens*(1-ltfu_cx))        # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu_cx)*cx_sens*(cxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + mods_cost
                    if retx==1:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + mods_cost
                    if retx==1:
                        p_1st = (1-ltfu)*cxr_spec+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)*(1-cxr_sens)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*cxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc))
                        cost_dx = sm_cost + outpt_cost + mods_cost
                    if retx==1:
                        p_1st = (1-ltfu)*(1-cxr_sens)+ltfu*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*cxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + mods_cost + outpt_cost

        # same-day smear
        if interv==4:
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                    if retx==1:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                if dr==1:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                    if retx==1:                    
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                if dr==2:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                    if retx==1:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = 1          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                    if retx==1:
                        p_1st = cxr_spec         # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost + sd_cost
                if dr==1:
                    if retx==0:
                        p_1st = 1          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                    if retx==1:
                        p_1st = cxr_spec         # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost + sd_cost
                if dr==2:
                    if retx==0:
                        p_1st = 1          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost + sd_cost
                    if retx==1:
                        p_1st = (1-cxr_sens)          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = cxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_cxr           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + cx_cost + outpt_cost + dst_cost + sd_cost
        # same-day GXP
        if interv==5:
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = gxp_sens*gxr_spec + (1-gxp_sens)*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                    if retx==1:
                        p_1st = gxp_sens*gxr_spec + (1-gxp_sens)*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                if dr==1:
                    if retx==0:
                        p_1st = gxp_sens*gxr_spec + (1-gxp_sens)*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                    if retx==1:
                        p_1st = gxp_sens*gxr_spec + (1-gxp_sens)*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = gxp_sens*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                if dr==2:
                    if retx==0:
                        p_1st = gxp_sens*(1-gxr_sens) + (1-gxp_sens)*emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = gxp_sens*(gxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                    if retx==1:
                        p_1st = gxp_sens*(1-gxr_sens) + (1-gxp_sens)*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = gxp_sens*(gxr_sens)               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                    if retx==1:
                        p_1st = (gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                if dr==1:
                    if retx==0:
                        p_1st = (gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                    if retx==1:
                        p_1st = (gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-gxr_sens)         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = gxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost
                    if retx==1:
                        p_1st = (1-gxr_sens)         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = 1.0/365         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = gxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = 1.0/365           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sdgxp_cost

        # smear, with GXP for smear-pos only
        if interv==6:
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                if dr==1:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:                    
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                if dr==2:
                    if retx==0:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
                    if retx==1:
                        p_1st = emp_tx          # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = 0               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = sm_cost + outpt_cost
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sm_cost
                    if retx==1:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sm_cost
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sm_cost
                    if retx==1:
                        p_1st = (1-ltfu) + ltfu*emp_tx - (1-ltfu)*(1-gxr_spec)     # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sm_cost
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)*(1-gxr_sens) + ltfu*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sm_cost
                    if retx==1:
                        p_1st = (1-ltfu)*(1-gxr_sens) + ltfu*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxr_sens               # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + sm_cost

        # GXP + cx confirmation
        if interv==8:
            if smear==0:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu)*gxp_sens*(gxr_spec + (1-gxr_spec)*cxr_spec) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)*(1-cxr_spec)               # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                    if retx==1:
                        p_1st = (1-ltfu)*gxp_sens*(gxr_spec + (1-gxr_spec)*cxr_spec) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)*(1-cxr_spec)                 # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu)*gxp_sens*(gxr_spec + (1-gxr_spec)*cxr_spec) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)*(1-cxr_spec)                 # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                    if retx==1:
                        p_1st = (1-ltfu)*gxp_sens*(gxr_spec + (1-gxr_spec)*cxr_spec) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(1-gxr_spec)*(1-cxr_spec)                # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)*gxp_sens*(1-gxr_sens*cxr_sens) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(gxr_sens)*(cxr_sens)                 # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (gxr_sens)*(cx_cost+dst_cost)
                    if retx==1:
                        p_1st = (1-ltfu)*gxp_sens*(1-gxr_sens*cxr_sens) + (ltfu+(1-ltfu)*(1-gxp_sens))*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxp_sens*(gxr_sens)*(cxr_sens)                 # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (gxr_sens)*(cx_cost+dst_cost)
            if smear==1:
                if dr==0:
                    if retx==0:
                        p_1st = (1-ltfu)*(gxr_spec + (1-gxr_spec)*cxr_spec) + ltfu*emp_tx     # prob of 1st-line tx
                        p_1suc = (1-fail_s)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)*(1-cxr_spec)                 # prob 2nd-line tx
                        p_2suc = (1-fail_s)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                    if retx==1:
                        p_1st = (1-ltfu)*(gxr_spec + (1-gxr_spec)*cxr_spec) + ltfu*emp_tx     # prob of 1st-line tx
                        p_1suc = (1-fail_rt)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)*(1-cxr_spec)                 # prob 2nd-line tx
                        p_2suc = (1-fail_rt)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                if dr==1:
                    if retx==0:
                        p_1st = (1-ltfu)*(gxr_spec + (1-gxr_spec)*cxr_spec) + ltfu*emp_tx      # prob of 1st-line tx
                        p_1suc = (1-fail_i1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)*(1-cxr_spec)                 # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                    if retx==1:
                        p_1st = (1-ltfu)*(gxr_spec + (1-gxr_spec)*cxr_spec) + ltfu*emp_tx     # prob of 1st-line tx
                        p_1suc = (1-fail_i2)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*(1-gxr_spec)*(1-cxr_spec)                # prob 2nd-line tx
                        p_2suc = (1-fail_i2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))                    
                        cost_dx = gxp_cost + outpt_cost + (1-gxr_spec)*(cx_cost+dst_cost)
                if dr==2:
                    if retx==0:
                        p_1st = (1-ltfu)*(1-gxr_sens*cxr_sens) + ltfu*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug1_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxr_sens*cxr_sens                # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (gxr_sens)*(cx_cost+dst_cost)
                    if retx==1:
                        p_1st = (1-ltfu)*(1-gxr_sens*cxr_sens) + ltfu*emp_tx         # prob of 1st-line tx
                        p_1suc = (1-fail_m1)     # prob 1st-line is successful
                        d_1st = t_smgxp         # time before 1st-line tx is started
                        c_1st = drug2_cost      # cost of 1st-line
                        p_2nd = (1-ltfu)*gxr_sens*cxr_sens                # prob 2nd-line tx
                        p_2suc = (1-fail_m2)     # prob 2nd-line is successful
                        d_2nd = t_smgxp           # time before 2nd-line is started
                        c_2nd = drug3_cost      # cost of 2nd-line
                        delay = 1/(d_1st + (d_2nd-d_1st)*p_2nd*(p_2suc-p_1suc)/(p_1st*p_1suc+p_2nd*p_2suc))
                        cost_dx = gxp_cost + outpt_cost + (gxr_sens)*(cx_cost+dst_cost)
        tx = dx_rate * ((p_1st*p_1suc) + (p_2nd*p_2suc))
        tx_inapp = dx_rate * (p_1st*(1-p_1suc) + p_2nd*(1-p_2suc))
        cost_tx = ((p_1st*p_1suc*c_1st) + (p_2nd*p_2suc*c_2nd))/(tx/dx_rate)
        cost_inapp = ((p_1st*(1-p_1suc)*c_1st) + (p_2nd*(1-p_2suc)*c_2nd))/(tx_inapp/dx_rate)

        return tx, delay, tx_inapp, cost_dx, cost_tx, cost_inapp

    # calculate specificity of testing for people without TB, according to HIV status
    def invspec(interv, hiv):
        if interv==0 or interv==1 or interv==4 or interv==6 or (interv==7 and hiv==0):
            sp = sm_spec
        if interv==2 or interv==5 or (interv==7 and hiv==1) or interv==8:
            sp = gxp_spec
        if interv==3:
            sp = cx_spec
        return sp

    # simple routine to define proportion of rapid progression, according to HIV status
    def rap(hiv):
        if hiv<25 or (hiv>=50 and hiv<75):
            rp = rapid
        if (hiv>=25 and hiv<50) or (hiv>=75):
            rp = rapid_h
        return rp

    # calculate mortality rate according to TB and HIV status
    def mortality(tb,hiv):
        m = mort
        if hiv==1:
            m += mort_h
        if tb==1:
            if hiv==0:
                m += mort_sp
            elif hiv==1:
                m += mort_tbh
        if tb==2:
            if hiv==0:
                m += mort_sn
            elif hiv==1:
                m += mort_tbh
        return m

    # cost of diagnosing & treating TB-negatives who present with symptoms,
    #    according to selected intervention, prior TB treatment status, and HIV status
    def dxcost(interv, retx, hiv):
        cost = 0.
        if interv==0 or interv==4 or interv==6 or (interv==7 and hiv==0):
            if retx==0:
                cost = sm_cost + outpt_cost + (1-sm_spec)*drug1_cost
            if retx==1:
                cost = sm_cost + outpt_cost + (1-sm_spec)*(drug2_cost + cx_cost) + \
                       (1-sm_spec)*(1-cx_spec)*dst_cost + (1-sm_spec)*(1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost)
            if interv==4:
                cost+=sd_cost
        if interv==1:
            if retx==0:
                cost = sm_cost + outpt_cost + (1-sm_spec)*drug1_cost
            if retx==1:
                cost = sm_cost + outpt_cost + cx_cost + (1-cx_spec)*dst_cost + \
                       (1-sm_spec)*drug2_cost + (1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost)
        if interv==2 or interv==5 or (interv==7 and hiv==1):
            if retx==0:
                cost = gxp_cost + outpt_cost + (1-gxp_spec)*drug1_cost + \
                       (1-gxp_spec)*(1-gxr_spec)*(drug3_cost-drug1_cost)
            if retx==1:
                cost = gxp_cost + outpt_cost + (1-gxp_spec)*drug2_cost + \
                       (1-gxp_spec)*(1-gxr_spec)*(drug3_cost-drug2_cost)
            if interv==5:
                cost+=sdgxp_cost
        if interv==8:
            if retx==0:
                cost = gxp_cost + outpt_cost + (1-gxp_spec)*drug1_cost + \
                       (1-gxp_spec)*(1-gxr_spec)*(cx_cost + dst_cost + \
                                                  (1-cxr_spec)*(drug3_cost-drug1_cost))
            if retx==1:
                cost = gxp_cost + outpt_cost + (1-gxp_spec)*drug2_cost + \
                       (1-gxp_spec)*(1-gxr_spec)*(cx_cost + dst_cost + \
                                                  (1-cxr_spec)*(drug3_cost-drug2_cost))
        if interv==3:
            if retx==0:
                cost = sm_cost + outpt_cost + mods_cost + (1-cx_spec)*drug1_cost + \
                       (1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug1_cost)
            if retx==1:
                cost = sm_cost + outpt_cost + mods_cost + (1-cx_spec)*drug2_cost + \
                       (1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost)
        return cost


    ##########################################################


    ##########################################################
    # 6. DIFFERENTIAL EQUATION FUNCTION
    ##########################################################
    # returns the 'change array'
    def getcharray( force_s, force_i, force_m, X):
        chg_arr = np.zeros((tb_num,tb_num))
        i1 = np.array([0,50]) # HIV-neg only
        i2 = np.array([25,75]) # HIV-pos only
        i3 = np.array([0,25,50,75]) # all
        i4 = np.array([0,25]) # new dx
        i5 = np.array([50,75]) # retx
        chg_arr[i1,i1+1] = force_s*(1-rapid)*X[i1] # infection that becomes latent
        chg_arr[i1,i1+2] = force_i*(1-rapid)*X[i1]
        chg_arr[i1,i1+3] = force_m*(1-rapid)*X[i1]
        chg_arr[i2,i2+1] = force_s*(1-rapid_h)*X[i2]
        chg_arr[i2,i2+2] = force_i*(1-rapid_h)*X[i2]
        chg_arr[i2,i2+3] = force_m*(1-rapid_h)*X[i2]
        chg_arr[i1,i1+4] = force_s*rapid*X[i1]     # primary progressive TB
        chg_arr[i1,i1+5] = force_i*rapid*X[i1]
        chg_arr[i1,i1+6] = force_m*rapid*X[i1]
        chg_arr[i2,i2+4] = force_s*rapid_h*X[i2]
        chg_arr[i2,i2+5] = force_i*rapid_h*X[i2]
        chg_arr[i2,i2+6] = force_m*rapid_h*X[i2]
        chg_arr[i1+1,i1+4] += react*X[i1+1]         # reactivation
        chg_arr[i1+2,i1+5] += react*X[i1+2]
        chg_arr[i1+3,i1+6] += react*X[i1+3]
        chg_arr[i2+1,i2+4] += react_h*X[i2+1]
        chg_arr[i2+2,i2+5] += react_h*X[i2+2]
        chg_arr[i2+3,i2+6] += react_h*X[i2+3]
        chg_arr[i1+1,i1+4] += force_s*rapid*(1-prot)*X[i1+1]     # reinfection to active
        chg_arr[i1+1,i1+5] += force_i*rapid*(1-prot)*X[i1+1]
        chg_arr[i1+1,i1+6] += force_m*rapid*(1-prot)*X[i1+1]
        chg_arr[i1+2,i1+4] += force_s*rapid*(1-prot)*X[i1+2]
        chg_arr[i1+2,i1+5] += force_i*rapid*(1-prot)*X[i1+2]
        chg_arr[i1+2,i1+6] += force_m*rapid*(1-prot)*X[i1+2]
        chg_arr[i1+3,i1+4] += force_s*rapid*(1-prot)*X[i1+3]
        chg_arr[i1+3,i1+5] += force_i*rapid*(1-prot)*X[i1+3]
        chg_arr[i1+3,i1+6] += force_m*rapid*(1-prot)*X[i1+3]
        chg_arr[i1+1,i1+2] += force_i*(1-rapid*(1-prot))*X[i1+1] # reinfection to latent
        chg_arr[i1+1,i1+3] += force_m*(1-rapid*(1-prot))*X[i1+1]
        chg_arr[i1+2,i1+1] += force_s*(1-rapid*(1-prot))*X[i1+2]
        chg_arr[i1+2,i1+3] += force_m*(1-rapid*(1-prot))*X[i1+2]
        chg_arr[i1+3,i1+1] += force_s*(1-rapid*(1-prot))*X[i1+3]
        chg_arr[i1+3,i1+2] += force_i*(1-rapid*(1-prot))*X[i1+3]
        chg_arr[i2+1,i2+4] += force_s*rapid_h*X[i2+1]     # reinfection to active (HIV)
        chg_arr[i2+1,i2+5] += force_i*rapid_h*X[i2+1]
        chg_arr[i2+1,i2+6] += force_m*rapid_h*X[i2+1]
        chg_arr[i2+2,i2+4] += force_s*rapid_h*X[i2+2]
        chg_arr[i2+2,i2+5] += force_i*rapid_h*X[i2+2]
        chg_arr[i2+2,i2+6] += force_m*rapid_h*X[i2+2]
        chg_arr[i2+3,i2+4] += force_s*rapid_h*X[i2+3]
        chg_arr[i2+3,i2+5] += force_i*rapid_h*X[i2+3]
        chg_arr[i2+3,i2+6] += force_m*rapid_h*X[i2+3]
        chg_arr[i1+4,i1+5] += force_i*(1-rapid_h)*X[i1+4]  # reinfection to latent (HIV)
        chg_arr[i1+4,i1+6] += force_m*(1-rapid_h)*X[i1+4]
        chg_arr[i1+5,i1+4] += force_s*(1-rapid_h)*X[i1+5]
        chg_arr[i1+5,i1+6] += force_m*(1-rapid_h)*X[i1+5]
        chg_arr[i1+6,i1+4] += force_s*(1-rapid_h)*X[i1+6]
        chg_arr[i1+6,i1+5] += force_i*(1-rapid_h)*X[i1+6]
        chg_arr[i1+4,i1+7] += (1/predx_dur-cure_sn)*prop_inf*X[i1+4]  # progression to dx-seeking
        chg_arr[i1+4,i1+8] += (1/predx_dur-cure_sn)*(1-prop_inf)*X[i1+4]
        chg_arr[i1+5,i1+9] += (1/predx_dur-cure_sn)*prop_inf*X[i1+5]
        chg_arr[i1+5,i1+10] += (1/predx_dur-cure_sn)*(1-prop_inf)*X[i1+5]
        chg_arr[i1+6,i1+11] += (1/predx_dur-cure_sn)*prop_inf*X[i1+6]
        chg_arr[i1+6,i1+12] += (1/predx_dur-cure_sn)*(1-prop_inf)*X[i1+6]
        chg_arr[i2+4,i2+7] += (1/predx_dur_h)*prop_inf_h*X[i2+4]  # progress to dx-seek, HIV
        chg_arr[i2+4,i2+8] += (1/predx_dur_h)*(1-prop_inf_h)*X[i2+4]
        chg_arr[i2+5,i2+9] += (1/predx_dur_h)*prop_inf_h*X[i2+5]
        chg_arr[i2+5,i2+10] += (1/predx_dur_h)*(1-prop_inf_h)*X[i2+5]
        chg_arr[i2+6,i2+11] += (1/predx_dur_h)*prop_inf_h*X[i2+6]
        chg_arr[i2+6,i2+12] += (1/predx_dur_h)*(1-prop_inf_h)*X[i2+6]


        # The equations below use the output from the decision-tree
        # "txrate" function.
        # Arguments for this function are: intervention, smear, DR, retreat, hiv
        # Outputs are: tx rate, dx delay, tx_inapp, cost_dx, cost_tx, cost_inapp
        chg_arr[7,13] += txrate(intopt,1,0,0,0)[0]*X[7] # progress to "dx in progress"
        chg_arr[8,14] += txrate(intopt,0,0,0,0)[0]*X[8] # new
        chg_arr[9,15] += txrate(intopt,1,1,0,0)[0]*X[9]
        chg_arr[10,16] += txrate(intopt,0,1,0,0)[0]*X[10]
        chg_arr[11,17] += txrate(intopt,1,2,0,0)[0]*X[11]
        chg_arr[12,18] += txrate(intopt,0,2,0,0)[0]*X[12]
        chg_arr[57,63] += txrate(intopt,1,0,1,0)[0]*X[57] # progress to "dx in progress"
        chg_arr[58,64] += txrate(intopt,0,0,1,0)[0]*X[58] # retreat
        chg_arr[59,65] += txrate(intopt,1,1,1,0)[0]*X[59]
        chg_arr[60,66] += txrate(intopt,0,1,1,0)[0]*X[60]
        chg_arr[61,67] += txrate(intopt,1,2,1,0)[0]*X[61]
        chg_arr[62,68] += txrate(intopt,0,2,1,0)[0]*X[62]
        chg_arr[7,19] += txrate(intopt,1,0,0,0)[2]*(1-acq_s)*X[7] # progress to "inapp tx"
        chg_arr[8,20] += txrate(intopt,0,0,0,0)[2]*(1-acq_s)*X[8] # new
        chg_arr[9,21] += txrate(intopt,1,1,0,0)[2]*(1-acq_i1)*X[9]
        chg_arr[10,22] += txrate(intopt,0,1,0,0)[2]*(1-acq_i1)*X[10]
        chg_arr[11,23] += txrate(intopt,1,2,0,0)[2]*X[11]
        chg_arr[12,24] += txrate(intopt,0,2,0,0)[2]*X[12]
        chg_arr[7,21] += txrate(intopt,1,0,0,0)[2]*(acq_s)*(1-acq_mdr)*X[7] # new resistance
        chg_arr[8,22] += txrate(intopt,0,0,0,0)[2]*(acq_s)*(1-acq_mdr)*X[8]
        chg_arr[7,23] += txrate(intopt,1,0,0,0)[2]*(acq_s)*(acq_mdr)*X[7] # new resistance
        chg_arr[8,24] += txrate(intopt,0,0,0,0)[2]*(acq_s)*(acq_mdr)*X[8]
        chg_arr[9,23] += txrate(intopt,1,1,0,0)[2]*(acq_i1)*X[9]
        chg_arr[10,24] += txrate(intopt,0,1,0,0)[2]*(acq_i1)*X[10]
        chg_arr[57,69] += txrate(intopt,1,0,1,0)[2]*(1-acq_s)*X[57] # progress to "inapp tx"
        chg_arr[58,70] += txrate(intopt,0,0,1,0)[2]*(1-acq_s)*X[58] # retreat
        chg_arr[59,71] += txrate(intopt,1,1,1,0)[2]*(1-acq_i2)*X[59]
        chg_arr[60,72] += txrate(intopt,0,1,1,0)[2]*(1-acq_i2)*X[60]
        chg_arr[61,73] += txrate(intopt,1,2,1,0)[2]*X[61]
        chg_arr[62,74] += txrate(intopt,0,2,1,0)[2]*X[62]
        chg_arr[57,71] += txrate(intopt,1,0,1,0)[2]*(acq_s)*(1-acq_mdr)*X[57] # new resistance
        chg_arr[58,72] += txrate(intopt,0,0,1,0)[2]*(acq_s)*(1-acq_mdr)*X[58]
        chg_arr[57,73] += txrate(intopt,1,0,1,0)[2]*(acq_s)*(acq_mdr)*X[57] # new resistance
        chg_arr[58,74] += txrate(intopt,0,0,1,0)[2]*(acq_s)*(acq_mdr)*X[58]
        chg_arr[59,73] += txrate(intopt,1,1,1,0)[2]*(acq_i2)*X[59]
        chg_arr[60,74] += txrate(intopt,0,1,1,0)[2]*(acq_i2)*X[60]
        chg_arr[13,51] += txrate(intopt,1,0,0,0)[1]*X[13] # progress to treated
        chg_arr[14,51] += txrate(intopt,0,0,0,0)[1]*X[14] # new
        chg_arr[15,52] += txrate(intopt,1,1,0,0)[1]*X[15]
        chg_arr[16,52] += txrate(intopt,0,1,0,0)[1]*X[16]
        chg_arr[17,53] += txrate(intopt,1,2,0,0)[1]*X[17]
        chg_arr[18,53] += txrate(intopt,0,2,0,0)[1]*X[18]
        chg_arr[63,51] += txrate(intopt,1,0,1,0)[1]*X[63] # progress to treated
        chg_arr[64,51] += txrate(intopt,0,0,1,0)[1]*X[64] # retreat
        chg_arr[65,52] += txrate(intopt,1,1,1,0)[1]*X[65]
        chg_arr[66,52] += txrate(intopt,0,1,1,0)[1]*X[66]
        chg_arr[67,53] += txrate(intopt,1,2,1,0)[1]*X[67]
        chg_arr[68,53] += txrate(intopt,0,2,1,0)[1]*X[68]

        chg_arr[7+25,13+25] += txrate(intopt,1,0,0,1)[0]*X[7+25] # progress to "dx in progress"
        chg_arr[8+25,14+25] += txrate(intopt,0,0,0,1)[0]*X[8+25] # new
        chg_arr[9+25,15+25] += txrate(intopt,1,1,0,1)[0]*X[9+25]
        chg_arr[10+25,16+25] += txrate(intopt,0,1,0,1)[0]*X[10+25]
        chg_arr[11+25,17+25] += txrate(intopt,1,2,0,1)[0]*X[11+25]
        chg_arr[12+25,18+25] += txrate(intopt,0,2,0,1)[0]*X[12+25]
        chg_arr[57+25,63+25] += txrate(intopt,1,0,1,1)[0]*X[57+25] # progress to "dx in progress"
        chg_arr[58+25,64+25] += txrate(intopt,0,0,1,1)[0]*X[58+25] # retreat
        chg_arr[59+25,65+25] += txrate(intopt,1,1,1,1)[0]*X[59+25]
        chg_arr[60+25,66+25] += txrate(intopt,0,1,1,1)[0]*X[60+25]
        chg_arr[61+25,67+25] += txrate(intopt,1,2,1,1)[0]*X[61+25]
        chg_arr[62+25,68+25] += txrate(intopt,0,2,1,1)[0]*X[62+25]
        chg_arr[7+25,19+25] += txrate(intopt,1,0,0,1)[2]*(1-acq_s)*X[7+25] # progress to "inapp tx"
        chg_arr[8+25,20+25] += txrate(intopt,0,0,0,1)[2]*(1-acq_s)*X[8+25] # new
        chg_arr[9+25,21+25] += txrate(intopt,1,1,0,1)[2]*(1-acq_i1)*X[9+25]
        chg_arr[10+25,22+25] += txrate(intopt,0,1,0,1)[2]*(1-acq_i1)*X[10+25]
        chg_arr[11+25,23+25] += txrate(intopt,1,2,0,1)[2]*X[11+25]
        chg_arr[12+25,24+25] += txrate(intopt,0,2,0,1)[2]*X[12+25]
        chg_arr[7+25,21+25] += txrate(intopt,1,0,0,1)[2]*(acq_s)*(1-acq_mdr)*X[7+25] # new resistance
        chg_arr[8+25,22+25] += txrate(intopt,0,0,0,1)[2]*(acq_s)*(1-acq_mdr)*X[8+25]
        chg_arr[7+25,23+25] += txrate(intopt,1,0,0,1)[2]*(acq_s)*(acq_mdr)*X[7+25] # new resistance
        chg_arr[8+25,24+25] += txrate(intopt,0,0,0,1)[2]*(acq_s)*(acq_mdr)*X[8+25]
        chg_arr[9+25,23+25] += txrate(intopt,1,1,0,1)[2]*(acq_i1)*X[9+25]
        chg_arr[10+25,24+25] += txrate(intopt,0,1,0,1)[2]*(acq_i1)*X[10+25]
        chg_arr[57+25,69+25] += txrate(intopt,1,0,1,1)[2]*(1-acq_s)*X[57+25] # progress to "inapp tx"
        chg_arr[58+25,70+25] += txrate(intopt,0,0,1,1)[2]*(1-acq_s)*X[58+25] # retreat
        chg_arr[59+25,71+25] += txrate(intopt,1,1,1,1)[2]*(1-acq_i2)*X[59+25]
        chg_arr[60+25,72+25] += txrate(intopt,0,1,1,1)[2]*(1-acq_i2)*X[60+25]
        chg_arr[61+25,73+25] += txrate(intopt,1,2,1,1)[2]*X[61+25]
        chg_arr[62+25,74+25] += txrate(intopt,0,2,1,1)[2]*X[62+25]
        chg_arr[57+25,71+25] += txrate(intopt,1,0,1,1)[2]*(acq_s)*(1-acq_mdr)*X[57+25] # new resistance
        chg_arr[58+25,72+25] += txrate(intopt,0,0,1,1)[2]*(acq_s)*(1-acq_mdr)*X[58+25]
        chg_arr[57+25,73+25] += txrate(intopt,1,0,1,1)[2]*(acq_s)*(acq_mdr)*X[57+25] # new resistance
        chg_arr[58+25,74+25] += txrate(intopt,0,0,1,1)[2]*(acq_s)*(acq_mdr)*X[58+25]
        chg_arr[59+25,73+25] += txrate(intopt,1,1,1,1)[2]*(acq_i2)*X[59+25]
        chg_arr[60+25,74+25] += txrate(intopt,0,1,1,1)[2]*(acq_i2)*X[60+25]
        chg_arr[13+25,51+25] += txrate(intopt,1,0,0,1)[1]*X[13+25] # progress to treated
        chg_arr[14+25,51+25] += txrate(intopt,0,0,0,1)[1]*X[14+25] # new
        chg_arr[15+25,52+25] += txrate(intopt,1,1,0,1)[1]*X[15+25]
        chg_arr[16+25,52+25] += txrate(intopt,0,1,0,1)[1]*X[16+25]
        chg_arr[17+25,53+25] += txrate(intopt,1,2,0,1)[1]*X[17+25]
        chg_arr[18+25,53+25] += txrate(intopt,0,2,0,1)[1]*X[18+25]
        chg_arr[63+25,51+25] += txrate(intopt,1,0,1,1)[1]*X[63+25] # progress to treated
        chg_arr[64+25,51+25] += txrate(intopt,0,0,1,1)[1]*X[64+25] # retreat
        chg_arr[65+25,52+25] += txrate(intopt,1,1,1,1)[1]*X[65+25]
        chg_arr[66+25,52+25] += txrate(intopt,0,1,1,1)[1]*X[66+25]
        chg_arr[67+25,53+25] += txrate(intopt,1,2,1,1)[1]*X[67+25]
        chg_arr[68+25,53+25] += txrate(intopt,0,2,1,1)[1]*X[68+25]

        chg_arr[i4+19,i4+57] += (12./dur_fail)*defvfail_s*X[i4+19] # default from "inapp tx" (new cases)   
        chg_arr[i4+20,i4+58] += (12./dur_fail)*defvfail_s*X[i4+20] # assume zero cost b/c 1/2 of default is cured
        chg_arr[i4+21,i4+59] += (12./dur_fail)*defvfail_i*X[i4+21] 
        chg_arr[i4+22,i4+60] += (12./dur_fail)*defvfail_i*X[i4+22]    
        chg_arr[i4+23,i4+61] += (12./dur_fail)*defvfail_m*X[i4+23]
        chg_arr[i4+24,i4+62] += (12./dur_fail)*defvfail_m*X[i4+24]    
        chg_arr[i5+19,i5+7] += (12./dur_fail)*defvfail_s*X[i5+19] # default from "inapp tx" (retx cases)
        chg_arr[i5+20,i5+8] += (12./dur_fail)*defvfail_s*X[i5+20] # assume zero cost b/c 1/2 of default is cured
        chg_arr[i5+21,i5+9] += (12./dur_fail)*defvfail_i*X[i5+21]
        chg_arr[i5+22,i5+10] += (12./dur_fail)*defvfail_i*X[i5+22]    
        chg_arr[i5+23,i5+11] += (12./dur_fail)*defvfail_m*X[i5+23]
        chg_arr[i5+24,i5+12] += (12./dur_fail)*defvfail_m*X[i5+24] 
        chg_arr[i4+19,i4+51] += (12./dur_fail)*(1-defvfail_s)*X[i4+19]*(1-fail_rt) # failure from "inapp tx" (new TB cases)    
        chg_arr[i4+20,i4+51] += (12./dur_fail)*(1-defvfail_s)*X[i4+20]*(1-fail_rt) # these are the pts that are then successfully treated 
        chg_arr[i4+21,i4+52] += (12./dur_fail)*(1-defvfail_i)*X[i4+21]*(1-fail_i2) # the ones below re-fail
        chg_arr[i4+22,i4+52] += (12./dur_fail)*(1-defvfail_i)*X[i4+22]*(1-fail_i2)   
        chg_arr[i4+23,i4+53] += (12./dur_fail)*(1-defvfail_m)*X[i4+23]*(1-fail_m2)
        chg_arr[i4+24,i4+53] += (12./dur_fail)*(1-defvfail_m)*X[i4+24]*(1-fail_m2)
        chg_arr[i4+19,i4+69] += (12./dur_fail)*(1-defvfail_s)*X[i4+19]*(fail_rt) # re-failure from "inapp tx" -> now classified as retx    
        chg_arr[i4+20,i4+70] += (12./dur_fail)*(1-defvfail_s)*X[i4+20]*(fail_rt) 
        chg_arr[i4+21,i4+71] += (12./dur_fail)*(1-defvfail_i)*X[i4+21]*(fail_i2) 
        chg_arr[i4+22,i4+72] += (12./dur_fail)*(1-defvfail_i)*X[i4+22]*(fail_i2)   
        chg_arr[i4+23,i4+73] += (12./dur_fail)*(1-defvfail_m)*X[i4+23]*(fail_m2)
        chg_arr[i4+24,i4+74] += (12./dur_fail)*(1-defvfail_m)*X[i4+24]*(fail_m2)
        chg_arr[i5+19,i5+1] += (12./dur_fail)*(1-defvfail_s)*X[i5+19]*(1-fail_rt) # failure from "inapp tx"    
        chg_arr[i5+20,i5+1] += (12./dur_fail)*(1-defvfail_s)*X[i5+20]*(1-fail_rt) # retx pts, so "re-fail" stays in the same box
        chg_arr[i5+21,i5+2] += (12./dur_fail)*(1-defvfail_i)*X[i5+21]*(1-fail_i2)  
        chg_arr[i5+22,i5+2] += (12./dur_fail)*(1-defvfail_i)*X[i5+22]*(1-fail_i2)   
        chg_arr[i5+23,i5+3] += (12./dur_fail)*(1-defvfail_m)*X[i5+23]*(1-fail_m2)
        chg_arr[i5+24,i5+3] += (12./dur_fail)*(1-defvfail_m)*X[i5+24]*(1-fail_m2)
        i6 = np.arange(25)                          # HIV infection
        chg_arr[i6,i6+25] += hiv_inc*X[i6]
        chg_arr[i6+50,i6+75] += hiv_inc*X[i6+50]
        i7 = np.arange(4)                           # inappropriate tx for TB
        chg_arr[i7,i7+50] += prop_cough*(1-invspec(intopt,0))*X[i7]
        chg_arr[i7+25,i7+75] += prop_cough*(1-invspec(intopt,1))*X[i7+25]
        i8 = np.arange(50)                         # mortality
        chg_arr[i6,0] += mortality(0,0)*X[i6]       # background mortality
        chg_arr[i6+50,0] += mortality(0,0)*X[i6+50]
        chg_arr[i6+25,0] += mortality(0,1)*X[i6+25] # HIV deaths
        chg_arr[i6+75,0] += mortality(0,1)*X[i6+75]
        i9 = np.array([7,9,11,13,15,17])  # smear-pos
        i10 = np.array([4,5,6,8,10,12,14,16,18,19,20,21,22,23,24])  # smear-neg
        chg_arr[i9,0] += mortality(1,0)*X[i9]              # smear-positive TB deaths
        chg_arr[i9+25,0] += mortality(1,1)*X[i9+25]
        chg_arr[i9+50,0] += mortality(1,0)*X[i9+50]
        chg_arr[i9+75,0] += mortality(1,1)*X[i9+75]
        chg_arr[i10,0] += mortality(2,0)*X[i10]              # smear-negative TB deaths
        chg_arr[i10+25,0] += mortality(2,1)*X[i10+25]
        chg_arr[i10+50,0] += mortality(2,0)*X[i10+50]
        chg_arr[i10+75,0] += mortality(2,1)*X[i10+75]
        chg_arr[i1+7,i1+1] += cure_sp*X[i1+7]              # smear-positive self-cure (no HIV)
        chg_arr[i1+9,i1+2] += cure_sp*X[i1+9]
        chg_arr[i1+11,i1+3] += cure_sp*X[i1+11]
        chg_arr[i1+13,i1+1] += cure_sp*X[i1+13]
        chg_arr[i1+15,i1+2] += cure_sp*X[i1+15]
        chg_arr[i1+17,i1+3] += cure_sp*X[i1+17]
        chg_arr[i1+8,i1+1] += cure_sn*X[i1+8]              # smear-negative self-cure (no HIV)
        chg_arr[i1+10,i1+2] += cure_sn*X[i1+10]
        chg_arr[i1+12,i1+3] += cure_sn*X[i1+12]
        chg_arr[i1+14,i1+1] += cure_sn*X[i1+14]
        chg_arr[i1+16,i1+2] += cure_sn*X[i1+16]
        chg_arr[i1+18,i1+3] += cure_sn*X[i1+18]
        chg_arr[i1+4,i1+1] += cure_sn*X[i1+4]
        chg_arr[i1+5,i1+2] += cure_sn*X[i1+5]
        chg_arr[i1+6,i1+3] += cure_sn*X[i1+6]
        chg_arr[i1+19,i1+1] += cure_sn*X[i1+19]
        chg_arr[i1+21,i1+2] += cure_sn*X[i1+21]
        chg_arr[i1+23,i1+3] += cure_sn*X[i1+23]
        chg_arr[i1+20,i1+1] += cure_sn*X[i1+20]
        chg_arr[i1+22,i1+2] += cure_sn*X[i1+22]
        chg_arr[i1+24,i1+3] += cure_sn*X[i1+24]

        return chg_arr
    
    # This function defines the differential equations that are later solved.

    def diffeq(population,time):
        dxdt = np.zeros(tb_num)  # create the vector of ODEs
        X = population

        # The following equations add up all infectious compartments to calculate the force of infection
        active_s = np.array([7,13])              # smear-pos, DS-TB compartments
        active_s1 = active_s + 25
        active_s = np.concatenate((active_s,active_s1))
        active_s1 = active_s + 50
        active_s = np.concatenate((active_s,active_s1))

        smneg_s = np.array([4,8,14,19,20])       # smear-neg, DS-TB compartments
        smneg_s1 = smneg_s + 25
        smneg_s = np.concatenate((smneg_s,smneg_s1))
        smneg_s1 = smneg_s + 50
        smneg_s = np.concatenate((smneg_s,smneg_s1))    

        active_i = np.array([9,15])              # smear-pos, INH-m compartments
        active_i1 = active_i + 25
        active_i = np.concatenate((active_i,active_i1))
        active_i1 = active_i + 50
        active_i = np.concatenate((active_i,active_i1))

        smneg_i = np.array([5,10,16,21,22])       # smear-neg, INH-m compartments
        smneg_i1 = smneg_i + 25
        smneg_i = np.concatenate((smneg_i,smneg_i1))
        smneg_i1 = smneg_i + 50
        smneg_i = np.concatenate((smneg_i,smneg_i1))

        active_m = np.array([11,17])              # smear-pos, MDR-TB compartments
        active_m1 = active_m + 25
        active_m = np.concatenate((active_m,active_m1))
        active_m1 = active_m + 50
        active_m = np.concatenate((active_m,active_m1))

        smneg_m = np.array([6,12,18,23,24])       # smear-neg, MDR-TB compartments
        smneg_m1 = smneg_m + 25
        smneg_m = np.concatenate((smneg_m,smneg_m1))
        smneg_m1 = smneg_m + 50
        smneg_m = np.concatenate((smneg_m,smneg_m1))

        # use the quantities above to calculate forces of infection:
        force_s = beta_s*(X[active_s].sum() + relbeta_sn*X[smneg_s].sum())/population.sum()
        force_i = beta_s*fit_i*(X[active_i].sum() + relbeta_sn*X[smneg_i].sum())/population.sum()
        force_m = beta_s*fit_m*(X[active_m].sum() + relbeta_sn*X[smneg_m].sum())/population.sum()
        force_t = force_s + force_i + force_m

        # # create a 100x100 matrix of flow rates:
        # # each element is the flow from row to column
        # # thus, chg_arr[0,1] is the flow from compartment 0 (uninfected) to
        # # compartment 1 (DS-latent)
        chg_arr = getcharray(force_s, force_i, force_m, X)

        # differential equations that are the output of the function:
        # each equation is "sum of all inputs minus sum of all outputs"
        dxdt = chg_arr.sum(axis=0) - chg_arr.sum(axis=1)

        return dxdt
    ###########################################################


    ##########################################################
    # 7. SOLVE FOR EQUILIBRIUM
    ##########################################################

    # The following equation replicates the function in
    #      6 above (differential equation function).
    # However, it substitutes 3 quantities: rec25, rec75, and rec50.
    # These quantities are used as placeholders for the given populations,
    #      X[25], X[50], and X[75].
    # Thus, rec25 denotes the population of the 25th compartment,
    #      rec50 denotes the population of the 50th compartment,
    #      rec75 denotes the population of the 75th compartment.
    # The "initial guess" for these populations is that they are equal in size.
    # Concomitantly, X[25] denotes hiv incidence,
    #       X[50] denotes the transmission parameter beta,
    #       X[75] denotes the reduction in transmission rate for MDR-TB.
    # The rates of change in X[25], X[50], and X[75] are now set to
    #       the difference between calculated and target HIV prevalence,
    #       between calculated and target TB incidence, and
    #       between calculated and target MDR-TB incidence, respectively.
    # Thus, at the roots of these equations (dx/dt = 0), the population
    #      has achieved the target HIV prevalence, TB incidence, and MDR-TB incidence.
    # Equations 100 and 101 are then used to maintain a constant population,
    #      ensuring that rec25 and rec50 are also not changing at steady-state.

    def solvebeta(population):
        dxdt = np.zeros(tb_num+2)  # create the vector of ODEs
        X = population
        rec1 = 100000 - np.sum(X) + X[25] + X[50] + X[75] + X[100] + X[101]
        rec25 = (rec1+X[100]+X[101])/3.   
        rec75 = rec25-X[100] # X[100] is the difference b/w X[25] & X[75]
        rec50 = rec25-X[101] # X[101] is the difference b/w X[50] & X[25]

        active_s = np.array([7,13])              # smear-pos, DS-TB compartments
        active_s1 = active_s + 25
        active_s = np.concatenate((active_s,active_s1))
        active_s1 = active_s + 50
        active_s = np.concatenate((active_s,active_s1))

        smneg_s = np.array([4,8,14,19,20])       # smear-neg, DS-TB compartments
        smneg_s1 = smneg_s + 25
        smneg_s = np.concatenate((smneg_s,smneg_s1))
        smneg_s1 = smneg_s + 50
        smneg_s = np.concatenate((smneg_s,smneg_s1))    

        active_i = np.array([9,15])              # smear-pos, INH-m compartments
        active_i1 = active_i + 25
        active_i = np.concatenate((active_i,active_i1))
        active_i1 = active_i + 50
        active_i = np.concatenate((active_i,active_i1))

        smneg_i = np.array([5,10,16,21,22])       # smear-neg, INH-m compartments
        smneg_i1 = smneg_i + 25
        smneg_i = np.concatenate((smneg_i,smneg_i1))
        smneg_i1 = smneg_i + 50
        smneg_i = np.concatenate((smneg_i,smneg_i1))

        active_m = np.array([11,17])              # smear-pos, MDR-TB compartments
        active_m1 = active_m + 25
        active_m = np.concatenate((active_m,active_m1))
        active_m1 = active_m + 50
        active_m = np.concatenate((active_m,active_m1))

        smneg_m = np.array([6,12,18,23,24])       # smear-neg, MDR-TB compartments
        smneg_m1 = smneg_m + 25
        smneg_m = np.concatenate((smneg_m,smneg_m1))
        smneg_m1 = smneg_m + 50
        smneg_m = np.concatenate((smneg_m,smneg_m1))

        # forces of infection:
        force_s = X[50]*(X[active_s].sum() + relbeta_sn*X[smneg_s].sum())/100000
        force_i = X[50]*(1.-X[75]/4.)*(X[active_i].sum() + relbeta_sn*X[smneg_i].sum())/100000
        force_m = X[50]*(1.-X[75])*(X[active_m].sum() + relbeta_sn*X[smneg_m].sum())/100000
        force_t = force_s + force_i + force_m

        # not quite getcharray
        # 100x100 matrix of flow rates:
        chg_arr = np.zeros((tb_num,tb_num))
        i1 = np.array([0,50]) # HIV-neg only
        i2 = np.array([25,75]) # HIV-pos only
        i3 = np.array([0,25,50,75]) # all
        i4 = np.array([0,25]) # new dx
        i5 = np.array([50,75]) # retx
        chg_arr[0,1] = force_s*(1-rapid)*X[0] # infection that becomes latent
        chg_arr[0,2] = force_i*(1-rapid)*X[0]
        chg_arr[0,3] = force_m*(1-rapid)*X[0]
        chg_arr[50,51] = force_s*(1-rapid)*rec50 # infection that becomes latent
        chg_arr[50,52] = force_i*(1-rapid)*rec50
        chg_arr[50,53] = force_m*(1-rapid)*rec50
        chg_arr[25,26] = force_s*(1-rapid_h)*rec25
        chg_arr[25,27] = force_i*(1-rapid_h)*rec25
        chg_arr[25,28] = force_m*(1-rapid_h)*rec25
        chg_arr[75,76] = force_s*(1-rapid_h)*rec75
        chg_arr[75,77] = force_i*(1-rapid_h)*rec75
        chg_arr[75,78] = force_m*(1-rapid_h)*rec75
        chg_arr[0,4] = force_s*rapid*X[0]     # primary progressive TB
        chg_arr[0,5] = force_i*rapid*X[0]
        chg_arr[0,6] = force_m*rapid*X[0]
        chg_arr[50,54] = force_s*rapid*rec50     # primary progressive TB
        chg_arr[50,55] = force_i*rapid*rec50
        chg_arr[50,56] = force_m*rapid*rec50
        chg_arr[25,29] = force_s*rapid_h*rec25
        chg_arr[25,30] = force_i*rapid_h*rec25
        chg_arr[25,31] = force_m*rapid_h*rec25
        chg_arr[75,79] = force_s*rapid_h*rec75
        chg_arr[75,80] = force_i*rapid_h*rec75
        chg_arr[75,81] = force_m*rapid_h*rec75
        chg_arr[i1+1,i1+4] += react*X[i1+1]         # reactivation
        chg_arr[i1+2,i1+5] += react*X[i1+2]
        chg_arr[i1+3,i1+6] += react*X[i1+3]
        chg_arr[i2+1,i2+4] += react_h*X[i2+1]
        chg_arr[i2+2,i2+5] += react_h*X[i2+2]
        chg_arr[i2+3,i2+6] += react_h*X[i2+3]
        chg_arr[i1+1,i1+4] += force_s*rapid*(1-prot)*X[i1+1]     # reinfection to active
        chg_arr[i1+1,i1+5] += force_i*rapid*(1-prot)*X[i1+1]
        chg_arr[i1+1,i1+6] += force_m*rapid*(1-prot)*X[i1+1]
        chg_arr[i1+2,i1+4] += force_s*rapid*(1-prot)*X[i1+2]
        chg_arr[i1+2,i1+5] += force_i*rapid*(1-prot)*X[i1+2]
        chg_arr[i1+2,i1+6] += force_m*rapid*(1-prot)*X[i1+2]
        chg_arr[i1+3,i1+4] += force_s*rapid*(1-prot)*X[i1+3]
        chg_arr[i1+3,i1+5] += force_i*rapid*(1-prot)*X[i1+3]
        chg_arr[i1+3,i1+6] += force_m*rapid*(1-prot)*X[i1+3]
        chg_arr[i1+1,i1+2] += force_i*(1-rapid*(1-prot))*X[i1+1] # reinfection to latent
        chg_arr[i1+1,i1+3] += force_m*(1-rapid*(1-prot))*X[i1+1]
        chg_arr[i1+2,i1+1] += force_s*(1-rapid*(1-prot))*X[i1+2]
        chg_arr[i1+2,i1+3] += force_m*(1-rapid*(1-prot))*X[i1+2]
        chg_arr[i1+3,i1+1] += force_s*(1-rapid*(1-prot))*X[i1+3]
        chg_arr[i1+3,i1+2] += force_i*(1-rapid*(1-prot))*X[i1+3]
        chg_arr[i2+1,i2+4] += force_s*rapid_h*X[i2+1]     # reinfection to active (HIV)
        chg_arr[i2+1,i2+5] += force_i*rapid_h*X[i2+1]
        chg_arr[i2+1,i2+6] += force_m*rapid_h*X[i2+1]
        chg_arr[i2+2,i2+4] += force_s*rapid_h*X[i2+2]
        chg_arr[i2+2,i2+5] += force_i*rapid_h*X[i2+2]
        chg_arr[i2+2,i2+6] += force_m*rapid_h*X[i2+2]
        chg_arr[i2+3,i2+4] += force_s*rapid_h*X[i2+3]
        chg_arr[i2+3,i2+5] += force_i*rapid_h*X[i2+3]
        chg_arr[i2+3,i2+6] += force_m*rapid_h*X[i2+3]
        chg_arr[i1+4,i1+5] += force_i*(1-rapid_h)*X[i1+4]  # reinfection to latent (HIV)
        chg_arr[i1+4,i1+6] += force_m*(1-rapid_h)*X[i1+4]
        chg_arr[i1+5,i1+4] += force_s*(1-rapid_h)*X[i1+5]
        chg_arr[i1+5,i1+6] += force_m*(1-rapid_h)*X[i1+5]
        chg_arr[i1+6,i1+4] += force_s*(1-rapid_h)*X[i1+6]
        chg_arr[i1+6,i1+5] += force_i*(1-rapid_h)*X[i1+6]
        chg_arr[i1+4,i1+7] += (1/predx_dur-cure_sn)*prop_inf*X[i1+4]  # progression to dx-seeking
        chg_arr[i1+4,i1+8] += (1/predx_dur-cure_sn)*(1-prop_inf)*X[i1+4]
        chg_arr[i1+5,i1+9] += (1/predx_dur-cure_sn)*prop_inf*X[i1+5]
        chg_arr[i1+5,i1+10] += (1/predx_dur-cure_sn)*(1-prop_inf)*X[i1+5]
        chg_arr[i1+6,i1+11] += (1/predx_dur-cure_sn)*prop_inf*X[i1+6]
        chg_arr[i1+6,i1+12] += (1/predx_dur-cure_sn)*(1-prop_inf)*X[i1+6]
        chg_arr[i2+4,i2+7] += (1/predx_dur_h)*prop_inf_h*X[i2+4]  # progress to dx-seek, HIV
        chg_arr[i2+4,i2+8] += (1/predx_dur_h)*(1-prop_inf_h)*X[i2+4]
        chg_arr[i2+5,i2+9] += (1/predx_dur_h)*prop_inf_h*X[i2+5]
        chg_arr[i2+5,i2+10] += (1/predx_dur_h)*(1-prop_inf_h)*X[i2+5]
        chg_arr[i2+6,i2+11] += (1/predx_dur_h)*prop_inf_h*X[i2+6]
        chg_arr[i2+6,i2+12] += (1/predx_dur_h)*(1-prop_inf_h)*X[i2+6]

        chg_arr[7,13] += txrate(intopt,1,0,0,0)[0]*X[7] # progress to "dx in progress"
        chg_arr[8,14] += txrate(intopt,0,0,0,0)[0]*X[8] # new
        chg_arr[9,15] += txrate(intopt,1,1,0,0)[0]*X[9]
        chg_arr[10,16] += txrate(intopt,0,1,0,0)[0]*X[10]
        chg_arr[11,17] += txrate(intopt,1,2,0,0)[0]*X[11]
        chg_arr[12,18] += txrate(intopt,0,2,0,0)[0]*X[12]
        chg_arr[57,63] += txrate(intopt,1,0,1,0)[0]*X[57] # progress to "dx in progress"
        chg_arr[58,64] += txrate(intopt,0,0,1,0)[0]*X[58] # retreat
        chg_arr[59,65] += txrate(intopt,1,1,1,0)[0]*X[59]
        chg_arr[60,66] += txrate(intopt,0,1,1,0)[0]*X[60]
        chg_arr[61,67] += txrate(intopt,1,2,1,0)[0]*X[61]
        chg_arr[62,68] += txrate(intopt,0,2,1,0)[0]*X[62]
        chg_arr[7,19] += txrate(intopt,1,0,0,0)[2]*(1-acq_s)*X[7] # progress to "inapp tx"
        chg_arr[8,20] += txrate(intopt,0,0,0,0)[2]*(1-acq_s)*X[8] # new
        chg_arr[9,21] += txrate(intopt,1,1,0,0)[2]*(1-acq_i1)*X[9]
        chg_arr[10,22] += txrate(intopt,0,1,0,0)[2]*(1-acq_i1)*X[10]
        chg_arr[11,23] += txrate(intopt,1,2,0,0)[2]*X[11]
        chg_arr[12,24] += txrate(intopt,0,2,0,0)[2]*X[12]
        chg_arr[7,21] += txrate(intopt,1,0,0,0)[2]*(acq_s)*(1-acq_mdr)*X[7] # new resistance
        chg_arr[8,22] += txrate(intopt,0,0,0,0)[2]*(acq_s)*(1-acq_mdr)*X[8]
        chg_arr[7,23] += txrate(intopt,1,0,0,0)[2]*(acq_s)*(acq_mdr)*X[7] # new resistance
        chg_arr[8,24] += txrate(intopt,0,0,0,0)[2]*(acq_s)*(acq_mdr)*X[8]
        chg_arr[9,23] += txrate(intopt,1,1,0,0)[2]*(acq_i1)*X[9]
        chg_arr[10,24] += txrate(intopt,0,1,0,0)[2]*(acq_i1)*X[10]
        chg_arr[57,69] += txrate(intopt,1,0,1,0)[2]*(1-acq_s)*X[57] # progress to "inapp tx"
        chg_arr[58,70] += txrate(intopt,0,0,1,0)[2]*(1-acq_s)*X[58] # retreat
        chg_arr[59,71] += txrate(intopt,1,1,1,0)[2]*(1-acq_i2)*X[59]
        chg_arr[60,72] += txrate(intopt,0,1,1,0)[2]*(1-acq_i2)*X[60]
        chg_arr[61,73] += txrate(intopt,1,2,1,0)[2]*X[61]
        chg_arr[62,74] += txrate(intopt,0,2,1,0)[2]*X[62]
        chg_arr[57,71] += txrate(intopt,1,0,1,0)[2]*(acq_s)*(1-acq_mdr)*X[57] # new resistance
        chg_arr[58,72] += txrate(intopt,0,0,1,0)[2]*(acq_s)*(1-acq_mdr)*X[58]
        chg_arr[57,73] += txrate(intopt,1,0,1,0)[2]*(acq_s)*(acq_mdr)*X[57] # new resistance
        chg_arr[58,74] += txrate(intopt,0,0,1,0)[2]*(acq_s)*(acq_mdr)*X[58]
        chg_arr[59,73] += txrate(intopt,1,1,1,0)[2]*(acq_i2)*X[59]
        chg_arr[60,74] += txrate(intopt,0,1,1,0)[2]*(acq_i2)*X[60]
        chg_arr[13,51] += txrate(intopt,1,0,0,0)[1]*X[13] # progress to treated
        chg_arr[14,51] += txrate(intopt,0,0,0,0)[1]*X[14] # new
        chg_arr[15,52] += txrate(intopt,1,1,0,0)[1]*X[15]
        chg_arr[16,52] += txrate(intopt,0,1,0,0)[1]*X[16]
        chg_arr[17,53] += txrate(intopt,1,2,0,0)[1]*X[17]
        chg_arr[18,53] += txrate(intopt,0,2,0,0)[1]*X[18]
        chg_arr[63,51] += txrate(intopt,1,0,1,0)[1]*X[63] # progress to treated
        chg_arr[64,51] += txrate(intopt,0,0,1,0)[1]*X[64] # retreat
        chg_arr[65,52] += txrate(intopt,1,1,1,0)[1]*X[65]
        chg_arr[66,52] += txrate(intopt,0,1,1,0)[1]*X[66]
        chg_arr[67,53] += txrate(intopt,1,2,1,0)[1]*X[67]
        chg_arr[68,53] += txrate(intopt,0,2,1,0)[1]*X[68]

        chg_arr[7+25,13+25] += txrate(intopt,1,0,0,1)[0]*X[7+25] # progress to "dx in progress"
        chg_arr[8+25,14+25] += txrate(intopt,0,0,0,1)[0]*X[8+25] # new
        chg_arr[9+25,15+25] += txrate(intopt,1,1,0,1)[0]*X[9+25]
        chg_arr[10+25,16+25] += txrate(intopt,0,1,0,1)[0]*X[10+25]
        chg_arr[11+25,17+25] += txrate(intopt,1,2,0,1)[0]*X[11+25]
        chg_arr[12+25,18+25] += txrate(intopt,0,2,0,1)[0]*X[12+25]
        chg_arr[57+25,63+25] += txrate(intopt,1,0,1,1)[0]*X[57+25] # progress to "dx in progress"
        chg_arr[58+25,64+25] += txrate(intopt,0,0,1,1)[0]*X[58+25] # retreat
        chg_arr[59+25,65+25] += txrate(intopt,1,1,1,1)[0]*X[59+25]
        chg_arr[60+25,66+25] += txrate(intopt,0,1,1,1)[0]*X[60+25]
        chg_arr[61+25,67+25] += txrate(intopt,1,2,1,1)[0]*X[61+25]
        chg_arr[62+25,68+25] += txrate(intopt,0,2,1,1)[0]*X[62+25]
        chg_arr[7+25,19+25] += txrate(intopt,1,0,0,1)[2]*(1-acq_s)*X[7+25] # progress to "inapp tx"
        chg_arr[8+25,20+25] += txrate(intopt,0,0,0,1)[2]*(1-acq_s)*X[8+25] # new
        chg_arr[9+25,21+25] += txrate(intopt,1,1,0,1)[2]*(1-acq_i1)*X[9+25]
        chg_arr[10+25,22+25] += txrate(intopt,0,1,0,1)[2]*(1-acq_i1)*X[10+25]
        chg_arr[11+25,23+25] += txrate(intopt,1,2,0,1)[2]*X[11+25]
        chg_arr[12+25,24+25] += txrate(intopt,0,2,0,1)[2]*X[12+25]
        chg_arr[7+25,21+25] += txrate(intopt,1,0,0,1)[2]*(acq_s)*(1-acq_mdr)*X[7+25] # new resistance
        chg_arr[8+25,22+25] += txrate(intopt,0,0,0,1)[2]*(acq_s)*(1-acq_mdr)*X[8+25]
        chg_arr[7+25,23+25] += txrate(intopt,1,0,0,1)[2]*(acq_s)*(acq_mdr)*X[7+25] # new resistance
        chg_arr[8+25,24+25] += txrate(intopt,0,0,0,1)[2]*(acq_s)*(acq_mdr)*X[8+25]
        chg_arr[9+25,23+25] += txrate(intopt,1,1,0,1)[2]*(acq_i1)*X[9+25]
        chg_arr[10+25,24+25] += txrate(intopt,0,1,0,1)[2]*(acq_i1)*X[10+25]
        chg_arr[57+25,69+25] += txrate(intopt,1,0,1,1)[2]*(1-acq_s)*X[57+25] # progress to "inapp tx"
        chg_arr[58+25,70+25] += txrate(intopt,0,0,1,1)[2]*(1-acq_s)*X[58+25] # retreat
        chg_arr[59+25,71+25] += txrate(intopt,1,1,1,1)[2]*(1-acq_i2)*X[59+25]
        chg_arr[60+25,72+25] += txrate(intopt,0,1,1,1)[2]*(1-acq_i2)*X[60+25]
        chg_arr[61+25,73+25] += txrate(intopt,1,2,1,1)[2]*X[61+25]
        chg_arr[62+25,74+25] += txrate(intopt,0,2,1,1)[2]*X[62+25]
        chg_arr[57+25,71+25] += txrate(intopt,1,0,1,1)[2]*(acq_s)*(1-acq_mdr)*X[57+25] # new resistance
        chg_arr[58+25,72+25] += txrate(intopt,0,0,1,1)[2]*(acq_s)*(1-acq_mdr)*X[58+25]
        chg_arr[57+25,73+25] += txrate(intopt,1,0,1,1)[2]*(acq_s)*(acq_mdr)*X[57+25] # new resistance
        chg_arr[58+25,74+25] += txrate(intopt,0,0,1,1)[2]*(acq_s)*(acq_mdr)*X[58+25]
        chg_arr[59+25,73+25] += txrate(intopt,1,1,1,1)[2]*(acq_i2)*X[59+25]
        chg_arr[60+25,74+25] += txrate(intopt,0,1,1,1)[2]*(acq_i2)*X[60+25]
        chg_arr[13+25,51+25] += txrate(intopt,1,0,0,1)[1]*X[13+25] # progress to treated
        chg_arr[14+25,51+25] += txrate(intopt,0,0,0,1)[1]*X[14+25] # new
        chg_arr[15+25,52+25] += txrate(intopt,1,1,0,1)[1]*X[15+25]
        chg_arr[16+25,52+25] += txrate(intopt,0,1,0,1)[1]*X[16+25]
        chg_arr[17+25,53+25] += txrate(intopt,1,2,0,1)[1]*X[17+25]
        chg_arr[18+25,53+25] += txrate(intopt,0,2,0,1)[1]*X[18+25]
        chg_arr[63+25,51+25] += txrate(intopt,1,0,1,1)[1]*X[63+25] # progress to treated
        chg_arr[64+25,51+25] += txrate(intopt,0,0,1,1)[1]*X[64+25] # retreat
        chg_arr[65+25,52+25] += txrate(intopt,1,1,1,1)[1]*X[65+25]
        chg_arr[66+25,52+25] += txrate(intopt,0,1,1,1)[1]*X[66+25]
        chg_arr[67+25,53+25] += txrate(intopt,1,2,1,1)[1]*X[67+25]
        chg_arr[68+25,53+25] += txrate(intopt,0,2,1,1)[1]*X[68+25]

        chg_arr[i4+19,i4+57] += (12./dur_fail)*defvfail_s*X[i4+19] # default from "inapp tx" (new cases)   
        chg_arr[i4+20,i4+58] += (12./dur_fail)*defvfail_s*X[i4+20] # assume zero cost b/c 1/2 of default is cured
        chg_arr[i4+21,i4+59] += (12./dur_fail)*defvfail_i*X[i4+21] 
        chg_arr[i4+22,i4+60] += (12./dur_fail)*defvfail_i*X[i4+22]    
        chg_arr[i4+23,i4+61] += (12./dur_fail)*defvfail_m*X[i4+23]
        chg_arr[i4+24,i4+62] += (12./dur_fail)*defvfail_m*X[i4+24]    
        chg_arr[i5+19,i5+7] += (12./dur_fail)*defvfail_s*X[i5+19] # default from "inapp tx" (retx cases)
        chg_arr[i5+20,i5+8] += (12./dur_fail)*defvfail_s*X[i5+20] # assume zero cost b/c 1/2 of default is cured
        chg_arr[i5+21,i5+9] += (12./dur_fail)*defvfail_i*X[i5+21]
        chg_arr[i5+22,i5+10] += (12./dur_fail)*defvfail_i*X[i5+22]    
        chg_arr[i5+23,i5+11] += (12./dur_fail)*defvfail_m*X[i5+23]
        chg_arr[i5+24,i5+12] += (12./dur_fail)*defvfail_m*X[i5+24] 
        chg_arr[i4+19,i4+51] += (12./dur_fail)*(1-defvfail_s)*X[i4+19]*(1-fail_rt) # failure from "inapp tx" (new TB cases)    
        chg_arr[i4+20,i4+51] += (12./dur_fail)*(1-defvfail_s)*X[i4+20]*(1-fail_rt) # these are the pts that are then successfully treated 
        chg_arr[i4+21,i4+52] += (12./dur_fail)*(1-defvfail_i)*X[i4+21]*(1-fail_i2) # the ones below re-fail
        chg_arr[i4+22,i4+52] += (12./dur_fail)*(1-defvfail_i)*X[i4+22]*(1-fail_i2)   
        chg_arr[i4+23,i4+53] += (12./dur_fail)*(1-defvfail_m)*X[i4+23]*(1-fail_m2)
        chg_arr[i4+24,i4+53] += (12./dur_fail)*(1-defvfail_m)*X[i4+24]*(1-fail_m2)
        chg_arr[i4+19,i4+69] += (12./dur_fail)*(1-defvfail_s)*X[i4+19]*(fail_rt) # re-failure from "inapp tx" -> now classified as retx    
        chg_arr[i4+20,i4+70] += (12./dur_fail)*(1-defvfail_s)*X[i4+20]*(fail_rt) 
        chg_arr[i4+21,i4+71] += (12./dur_fail)*(1-defvfail_i)*X[i4+21]*(fail_i2) 
        chg_arr[i4+22,i4+72] += (12./dur_fail)*(1-defvfail_i)*X[i4+22]*(fail_i2)   
        chg_arr[i4+23,i4+73] += (12./dur_fail)*(1-defvfail_m)*X[i4+23]*(fail_m2)
        chg_arr[i4+24,i4+74] += (12./dur_fail)*(1-defvfail_m)*X[i4+24]*(fail_m2)
        chg_arr[i5+19,i5+1] += (12./dur_fail)*(1-defvfail_s)*X[i5+19]*(1-fail_rt) # failure from "inapp tx"    
        chg_arr[i5+20,i5+1] += (12./dur_fail)*(1-defvfail_s)*X[i5+20]*(1-fail_rt) # retx pts, so "re-fail" stays in the same box
        chg_arr[i5+21,i5+2] += (12./dur_fail)*(1-defvfail_i)*X[i5+21]*(1-fail_i2)  
        chg_arr[i5+22,i5+2] += (12./dur_fail)*(1-defvfail_i)*X[i5+22]*(1-fail_i2)   
        chg_arr[i5+23,i5+3] += (12./dur_fail)*(1-defvfail_m)*X[i5+23]*(1-fail_m2)
        chg_arr[i5+24,i5+3] += (12./dur_fail)*(1-defvfail_m)*X[i5+24]*(1-fail_m2)
        i6 = np.arange(25)                          # HIV infection
        chg_arr[i6,i6+25] += X[25]*X[i6]
        chg_arr[i6+50,i6+75] += X[25]*X[i6+50]
        chg_arr[50,75]-=X[25]*X[50]
        chg_arr[50,75]+=X[25]*rec50
        i7 = np.arange(4)                           # inappropriate tx for TB
        chg_arr[i7,i7+50] += prop_cough*(1-invspec(intopt,0))*X[i7]
        chg_arr[i7+25,i7+75] += prop_cough*(1-invspec(intopt,1))*X[i7+25]
        chg_arr[25,75] -= prop_cough*(1-invspec(intopt,1))*X[25]
        chg_arr[25,75] += prop_cough*(1-invspec(intopt,1))*rec25
        i8 = np.arange(50)                         # mortality
        chg_arr[i6,0] += mortality(0,0)*X[i6]       # background mortality
        chg_arr[i6+50,0] += mortality(0,0)*X[i6+50]
        chg_arr[50,0]-=mortality(0,0)*X[50]
        chg_arr[50,0]+=mortality(0,0)*rec50
        chg_arr[i6+25,0] += mortality(0,1)*X[i6+25] # HIV deaths
        chg_arr[i6+75,0] += mortality(0,1)*X[i6+75]
        chg_arr[75,0]-=mortality(0,1)*X[75]
        chg_arr[75,0]+=mortality(0,1)*rec75
        chg_arr[25,0]-=mortality(0,1)*X[25]
        chg_arr[25,0]+=mortality(0,1)*rec25
        i9 = np.array([7,9,11,13,15,17])  # smear-pos
        i10 = np.array([4,5,6,8,10,12,14,16,18,19,20,21,22,23,24])  # smear-neg
        chg_arr[i9,0] += mortality(1,0)*X[i9]              # smear-positive TB deaths
        chg_arr[i9+25,0] += mortality(1,1)*X[i9+25]
        chg_arr[i9+50,0] += mortality(1,0)*X[i9+50]
        chg_arr[i9+75,0] += mortality(1,1)*X[i9+75]
        chg_arr[i10,0] += mortality(2,0)*X[i10]              # smear-negative TB deaths
        chg_arr[i10+25,0] += mortality(2,1)*X[i10+25]
        chg_arr[i10+50,0] += mortality(2,0)*X[i10+50]
        chg_arr[i10+75,0] += mortality(2,1)*X[i10+75]
        chg_arr[i1+7,i1+1] += cure_sp*X[i1+7]              # smear-positive self-cure (no HIV)
        chg_arr[i1+9,i1+2] += cure_sp*X[i1+9]
        chg_arr[i1+11,i1+3] += cure_sp*X[i1+11]
        chg_arr[i1+13,i1+1] += cure_sp*X[i1+13]
        chg_arr[i1+15,i1+2] += cure_sp*X[i1+15]
        chg_arr[i1+17,i1+3] += cure_sp*X[i1+17]
        chg_arr[i1+8,i1+1] += cure_sn*X[i1+8]              # smear-negative self-cure (no HIV)
        chg_arr[i1+10,i1+2] += cure_sn*X[i1+10]
        chg_arr[i1+12,i1+3] += cure_sn*X[i1+12]
        chg_arr[i1+14,i1+1] += cure_sn*X[i1+14]
        chg_arr[i1+16,i1+2] += cure_sn*X[i1+16]
        chg_arr[i1+18,i1+3] += cure_sn*X[i1+18]
        chg_arr[i1+4,i1+1] += cure_sn*X[i1+4]
        chg_arr[i1+5,i1+2] += cure_sn*X[i1+5]
        chg_arr[i1+6,i1+3] += cure_sn*X[i1+6]
        chg_arr[i1+19,i1+1] += cure_sn*X[i1+19]
        chg_arr[i1+21,i1+2] += cure_sn*X[i1+21]
        chg_arr[i1+23,i1+3] += cure_sn*X[i1+23]
        chg_arr[i1+20,i1+1] += cure_sn*X[i1+20]
        chg_arr[i1+22,i1+2] += cure_sn*X[i1+22]
        chg_arr[i1+24,i1+3] += cure_sn*X[i1+24]

        # calculate incidence (to ensure that it reaches the target)
        incidence = np.sum(chg_arr[i4,i4+4]) + np.sum(chg_arr[i4,i4+5]) + np.sum(chg_arr[i4,i4+6]) + \
                 np.sum(chg_arr[i4+1,i4+4]) + np.sum(chg_arr[i4+1,i4+5]) + np.sum(chg_arr[i4+1,i4+6]) + \
                 np.sum(chg_arr[i4+2,i4+4]) + np.sum(chg_arr[i4+2,i4+5]) + np.sum(chg_arr[i4+2,i4+6]) + \
                 np.sum(chg_arr[i4+3,i4+4]) + np.sum(chg_arr[i4+3,i4+5]) + np.sum(chg_arr[i4+3,i4+6]) + \
                 np.sum(chg_arr[i5,i5+4]) + np.sum(chg_arr[i5,i5+5]) + np.sum(chg_arr[i5,i5+6]) + \
                 np.sum(chg_arr[i5+1,i5+4]) + np.sum(chg_arr[i5+1,i5+5]) + np.sum(chg_arr[i5+1,i5+6]) + \
                 np.sum(chg_arr[i5+2,i5+4]) + np.sum(chg_arr[i5+2,i5+5]) + np.sum(chg_arr[i5+2,i5+6]) + \
                 np.sum(chg_arr[i5+3,i5+4]) + np.sum(chg_arr[i5+3,i5+5]) + np.sum(chg_arr[i5+3,i5+6]) + \
                 np.sum(chg_arr[i4+19,i4+57]) + np.sum(chg_arr[i4+20,i4+58]) + np.sum(chg_arr[i4+21,i4+59]) + \
                 np.sum(chg_arr[i4+22,i4+60]) + np.sum(chg_arr[i4+23,i4+61]) + np.sum(chg_arr[i4+24,i4+62]) + \
                 np.sum(chg_arr[i5+19,i5+7]) + np.sum(chg_arr[i5+20,i5+8]) + np.sum(chg_arr[i5+21,i5+9]) + \
                 np.sum(chg_arr[i5+22,i5+10]) + np.sum(chg_arr[i5+23,i5+11]) + np.sum(chg_arr[i5+24,i5+12]) + \
                 np.sum((12./dur_fail)*(1-defvfail_s)*X[i3+19]) + np.sum((12./dur_fail)*(1-defvfail_s)*X[i3+20]) + \
                 np.sum((12./dur_fail)*(1-defvfail_i)*X[i3+21]) + np.sum((12./dur_fail)*(1-defvfail_i)*X[i3+22]) + \
                 np.sum((12./dur_fail)*(1-defvfail_m)*X[i3+23]) + np.sum((12./dur_fail)*(1-defvfail_m)*X[i3+24])


        # calculate incidence of non-retreatment TB (to match MDR-TB incidence to target)
        incnew = np.sum(chg_arr[i4,i4+4]) + np.sum(chg_arr[i4,i4+5]) + np.sum(chg_arr[i4,i4+6]) + \
                 np.sum(chg_arr[i4+1,i4+4]) + np.sum(chg_arr[i4+1,i4+5]) + np.sum(chg_arr[i4+1,i4+6]) + \
                 np.sum(chg_arr[i4+2,i4+4]) + np.sum(chg_arr[i4+2,i4+5]) + np.sum(chg_arr[i4+2,i4+6]) + \
                 np.sum(chg_arr[i4+3,i4+4]) + np.sum(chg_arr[i4+3,i4+5]) + np.sum(chg_arr[i4+3,i4+6])

        # calculate incidence of new MDR-TB
        incmdrnew = np.sum(chg_arr[i4,i4+6]) + np.sum(chg_arr[i4+1,i4+6]) + np.sum(chg_arr[i4+2,i4+6]) + np.sum(chg_arr[i4+3,i4+6])

        # calculate HIV prevalence (to match to target)
        i12 = np.concatenate((np.arange(25,50), np.arange(75,100)))
        HIVprev = (sum(X[i12])-X[25]-X[75]+rec25+rec75)/1000

        # differential equations:
        dxdt[0] = np.sum(chg_arr[:,0])-np.sum(chg_arr[0,:])
        dxdt[1] = np.sum(chg_arr[:,1])-np.sum(chg_arr[1,:])
        dxdt[2] = np.sum(chg_arr[:,2])-np.sum(chg_arr[2,:])
        dxdt[3] = np.sum(chg_arr[:,3])-np.sum(chg_arr[3,:])
        dxdt[4] = np.sum(chg_arr[:,4])-np.sum(chg_arr[4,:])
        dxdt[5] = np.sum(chg_arr[:,5])-np.sum(chg_arr[5,:])
        dxdt[6] = np.sum(chg_arr[:,6])-np.sum(chg_arr[6,:])
        dxdt[7] = np.sum(chg_arr[:,7])-np.sum(chg_arr[7,:])
        dxdt[8] = np.sum(chg_arr[:,8])-np.sum(chg_arr[8,:])
        dxdt[9] = np.sum(chg_arr[:,9])-np.sum(chg_arr[9,:])
        dxdt[10] = np.sum(chg_arr[:,10])-np.sum(chg_arr[10,:])
        dxdt[11] = np.sum(chg_arr[:,11])-np.sum(chg_arr[11,:])
        dxdt[12] = np.sum(chg_arr[:,12])-np.sum(chg_arr[12,:])
        dxdt[13] = np.sum(chg_arr[:,13])-np.sum(chg_arr[13,:])
        dxdt[14] = np.sum(chg_arr[:,14])-np.sum(chg_arr[14,:])
        dxdt[15] = np.sum(chg_arr[:,15])-np.sum(chg_arr[15,:])
        dxdt[16] = np.sum(chg_arr[:,16])-np.sum(chg_arr[16,:])
        dxdt[17] = np.sum(chg_arr[:,17])-np.sum(chg_arr[17,:])
        dxdt[18] = np.sum(chg_arr[:,18])-np.sum(chg_arr[18,:])
        dxdt[19] = np.sum(chg_arr[:,19])-np.sum(chg_arr[19,:])
        dxdt[20] = np.sum(chg_arr[:,20])-np.sum(chg_arr[20,:])
        dxdt[21] = np.sum(chg_arr[:,21])-np.sum(chg_arr[21,:])
        dxdt[22] = np.sum(chg_arr[:,22])-np.sum(chg_arr[22,:])
        dxdt[23] = np.sum(chg_arr[:,23])-np.sum(chg_arr[23,:])
        dxdt[24] = np.sum(chg_arr[:,24])-np.sum(chg_arr[24,:])
        dxdt[25] = target_hiv - HIVprev # equals zero when HIV prevalence is at target
        dxdt[26] = np.sum(chg_arr[:,26])-np.sum(chg_arr[26,:])
        dxdt[27] = np.sum(chg_arr[:,27])-np.sum(chg_arr[27,:])
        dxdt[28] = np.sum(chg_arr[:,28])-np.sum(chg_arr[28,:])
        dxdt[29] = np.sum(chg_arr[:,29])-np.sum(chg_arr[29,:])
        dxdt[30] = np.sum(chg_arr[:,30])-np.sum(chg_arr[30,:])
        dxdt[31] = np.sum(chg_arr[:,31])-np.sum(chg_arr[31,:])
        dxdt[32] = np.sum(chg_arr[:,32])-np.sum(chg_arr[32,:])
        dxdt[33] = np.sum(chg_arr[:,33])-np.sum(chg_arr[33,:])
        dxdt[34] = np.sum(chg_arr[:,34])-np.sum(chg_arr[34,:])
        dxdt[35] = np.sum(chg_arr[:,35])-np.sum(chg_arr[35,:])
        dxdt[36] = np.sum(chg_arr[:,36])-np.sum(chg_arr[36,:])
        dxdt[37] = np.sum(chg_arr[:,37])-np.sum(chg_arr[37,:])
        dxdt[38] = np.sum(chg_arr[:,38])-np.sum(chg_arr[38,:])
        dxdt[39] = np.sum(chg_arr[:,39])-np.sum(chg_arr[39,:])
        dxdt[40] = np.sum(chg_arr[:,40])-np.sum(chg_arr[40,:])
        dxdt[41] = np.sum(chg_arr[:,41])-np.sum(chg_arr[41,:])
        dxdt[42] = np.sum(chg_arr[:,42])-np.sum(chg_arr[42,:])
        dxdt[43] = np.sum(chg_arr[:,43])-np.sum(chg_arr[43,:])
        dxdt[44] = np.sum(chg_arr[:,44])-np.sum(chg_arr[44,:])
        dxdt[45] = np.sum(chg_arr[:,45])-np.sum(chg_arr[45,:])
        dxdt[46] = np.sum(chg_arr[:,46])-np.sum(chg_arr[46,:])
        dxdt[47] = np.sum(chg_arr[:,47])-np.sum(chg_arr[47,:])
        dxdt[48] = np.sum(chg_arr[:,48])-np.sum(chg_arr[48,:])
        dxdt[49] = np.sum(chg_arr[:,49])-np.sum(chg_arr[49,:])
        dxdt[50] = target_inc-incidence # equals zero when TB incidence is at target
        dxdt[51] = np.sum(chg_arr[:,51])-np.sum(chg_arr[51,:])
        dxdt[52] = np.sum(chg_arr[:,52])-np.sum(chg_arr[52,:])
        dxdt[53] = np.sum(chg_arr[:,53])-np.sum(chg_arr[53,:])
        dxdt[54] = np.sum(chg_arr[:,54])-np.sum(chg_arr[54,:])
        dxdt[55] = np.sum(chg_arr[:,55])-np.sum(chg_arr[55,:])
        dxdt[56] = np.sum(chg_arr[:,56])-np.sum(chg_arr[56,:])
        dxdt[57] = np.sum(chg_arr[:,57])-np.sum(chg_arr[57,:])
        dxdt[58] = np.sum(chg_arr[:,58])-np.sum(chg_arr[58,:])
        dxdt[59] = np.sum(chg_arr[:,59])-np.sum(chg_arr[59,:])
        dxdt[60] = np.sum(chg_arr[:,60])-np.sum(chg_arr[60,:])
        dxdt[61] = np.sum(chg_arr[:,61])-np.sum(chg_arr[61,:])
        dxdt[62] = np.sum(chg_arr[:,62])-np.sum(chg_arr[62,:])
        dxdt[63] = np.sum(chg_arr[:,63])-np.sum(chg_arr[63,:])
        dxdt[64] = np.sum(chg_arr[:,64])-np.sum(chg_arr[64,:])
        dxdt[65] = np.sum(chg_arr[:,65])-np.sum(chg_arr[65,:])
        dxdt[66] = np.sum(chg_arr[:,66])-np.sum(chg_arr[66,:])
        dxdt[67] = np.sum(chg_arr[:,67])-np.sum(chg_arr[67,:])
        dxdt[68] = np.sum(chg_arr[:,68])-np.sum(chg_arr[68,:])
        dxdt[69] = np.sum(chg_arr[:,69])-np.sum(chg_arr[69,:])
        dxdt[70] = np.sum(chg_arr[:,70])-np.sum(chg_arr[70,:])
        dxdt[71] = np.sum(chg_arr[:,71])-np.sum(chg_arr[71,:])
        dxdt[72] = np.sum(chg_arr[:,72])-np.sum(chg_arr[72,:])
        dxdt[73] = np.sum(chg_arr[:,73])-np.sum(chg_arr[73,:])
        dxdt[74] = np.sum(chg_arr[:,74])-np.sum(chg_arr[74,:])
        dxdt[75] = target_mdr-incmdrnew/incnew # equals zero when new MDR-TB prevalence is at target
        dxdt[76] = np.sum(chg_arr[:,76])-np.sum(chg_arr[76,:])
        dxdt[77] = np.sum(chg_arr[:,77])-np.sum(chg_arr[77,:])
        dxdt[78] = np.sum(chg_arr[:,78])-np.sum(chg_arr[78,:])
        dxdt[79] = np.sum(chg_arr[:,79])-np.sum(chg_arr[79,:])
        dxdt[80] = np.sum(chg_arr[:,80])-np.sum(chg_arr[80,:])
        dxdt[81] = np.sum(chg_arr[:,81])-np.sum(chg_arr[81,:])
        dxdt[82] = np.sum(chg_arr[:,82])-np.sum(chg_arr[82,:])
        dxdt[83] = np.sum(chg_arr[:,83])-np.sum(chg_arr[83,:])
        dxdt[84] = np.sum(chg_arr[:,84])-np.sum(chg_arr[84,:])
        dxdt[85] = np.sum(chg_arr[:,85])-np.sum(chg_arr[85,:])
        dxdt[86] = np.sum(chg_arr[:,86])-np.sum(chg_arr[86,:])
        dxdt[87] = np.sum(chg_arr[:,87])-np.sum(chg_arr[87,:])
        dxdt[88] = np.sum(chg_arr[:,88])-np.sum(chg_arr[88,:])
        dxdt[89] = np.sum(chg_arr[:,89])-np.sum(chg_arr[89,:])
        dxdt[90] = np.sum(chg_arr[:,90])-np.sum(chg_arr[90,:])
        dxdt[91] = np.sum(chg_arr[:,91])-np.sum(chg_arr[91,:])
        dxdt[92] = np.sum(chg_arr[:,92])-np.sum(chg_arr[92,:])
        dxdt[93] = np.sum(chg_arr[:,93])-np.sum(chg_arr[93,:])
        dxdt[94] = np.sum(chg_arr[:,94])-np.sum(chg_arr[94,:])
        dxdt[95] = np.sum(chg_arr[:,95])-np.sum(chg_arr[95,:])
        dxdt[96] = np.sum(chg_arr[:,96])-np.sum(chg_arr[96,:])
        dxdt[97] = np.sum(chg_arr[:,97])-np.sum(chg_arr[97,:])
        dxdt[98] = np.sum(chg_arr[:,98])-np.sum(chg_arr[98,:])
        dxdt[99] = np.sum(chg_arr[:,99])-np.sum(chg_arr[99,:])
        dxdt[100] = np.sum(chg_arr[:,50])-np.sum(chg_arr[50,:])
        dxdt[101] = np.sum(chg_arr[:,25])-np.sum(chg_arr[25,:])
        return dxdt

    # The initial population for the "solvebeta" function requires initial
    #    estimates for the populations of compartments 100 and 101.
    I2 = np.append(I,[I[25]-I[75]])
    I3 = np.append(I2,[I[25]-I[50]])
    # As above, substitute compartments 25, 50, and 75 with the quantities
    #     of interest
    I3[25] = hiv_inc
    I3[50] = beta_s
    I3[75] = fit_m

    # solve for the roots of the system of equations.
    # compartment 25 is now hiv_inc, 50 beta_s, and 75 fit_m at steady-state.
    equipop_pre = fsolve(solvebeta,I3)
    beta_s = equipop_pre[50]
    fit_m = 1.-equipop_pre[75]
    fit_i = 1.-equipop_pre[75]*0.25
    hiv_inc = equipop_pre[25]

    # now, add back in the actual population sizes to compartments 25, 50, and 75.
    equipop_pre[25] = (100000-sum(equipop_pre)+equipop_pre[25]+equipop_pre[50] + \
                             equipop_pre[75]+2.*equipop_pre[100]+2.*equipop_pre[101])/3.
    equipop_pre[75] = equipop_pre[25]-equipop_pre[100]
    equipop_pre[50] = equipop_pre[25]-equipop_pre[101]

    # "equipop" is now the equilibrium population that fits the user-specified values
    # of TB incidence, MDR-TB incidence, and HIV prevalence.
    equipop = np.zeros(tb_num)
    equipop[:] = equipop_pre[0:100]

    ###########################################################

    # Check to make sure no negative cells - will stop running if they are detected:
    for xyz in range(100):
        if equipop[xyz]<0:
            #print "negative population - recalibrating..."
            I3[4:25] +=10.
            I3[29:50] +=10.
            I3[54:75]+=10.
            I3[79:100]+=10.
            I3[100] = I[25]-I[75]
            I3[101] = I[25]-I[50]
            I3[0] = 100000 - np.sum(I3[1:100])
            equipop_pre = fsolve(solvebeta,I3)
            # print equipop
            #print "beta", equipop_pre[50]
            #print "MDR fitness", 1.-equipop_pre[75]
            #print "INH fitness", 1.-equipop_pre[75]*0.25
            #print "HIV incidence", equipop_pre[25]
            beta_s = equipop_pre[50]
            fit_m = 1.-equipop_pre[75]
            fit_i = 1.-equipop_pre[75]*0.25
            hiv_inc = equipop_pre[25]
            # i10 = np.array([17,18,42,43,68,93])
            # equipop_pre[i10] = 0.
            equipop_pre[25] = (100000-np.sum(equipop_pre)+equipop_pre[25]+equipop_pre[50] + \
                                     equipop_pre[75]+2.*equipop_pre[100]+2.*equipop_pre[101])/3.
            equipop_pre[75] = equipop_pre[25]-equipop_pre[100]
            equipop_pre[50] = equipop_pre[25]-equipop_pre[101]
            equipop = np.zeros(tb_num)
            equipop[:] = equipop_pre[0:100]
            #print "total pop (check for 100,000)", sum(equipop[:])
            #print " "

    for xyz2 in range(100):
        if equipop[xyz2]<0:
            print "negative population - failed run"
            exit(0)


    ###########################################################
    # 8. CALCULATE INCIDENCE, PREVALENCE, MORTALITY, COST
    ###########################################################
    # This function takes the same equations as in 6 and 7 above
    # and uses them to calculate incidence (according to treatment & HIV status),
    # as well as TB and HIV prevalence, TB mortality, and total costs.

    def incprevmort(pop1):
        dxdt = np.zeros(tb_num)  # create the vector of ODEs
        X = pop1

        active_s = np.array([7,13])              # smear-pos, DS-TB compartments
        active_s1 = active_s + 25
        active_s = np.concatenate((active_s,active_s1))
        active_s1 = active_s + 50
        active_s = np.concatenate((active_s,active_s1))

        smneg_s = np.array([4,8,14,19,20])       # smear-neg, DS-TB compartments
        smneg_s1 = smneg_s + 25
        smneg_s = np.concatenate((smneg_s,smneg_s1))
        smneg_s1 = smneg_s + 50
        smneg_s = np.concatenate((smneg_s,smneg_s1))    

        active_i = np.array([9,15])              # smear-pos, INH-m compartments
        active_i1 = active_i + 25
        active_i = np.concatenate((active_i,active_i1))
        active_i1 = active_i + 50
        active_i = np.concatenate((active_i,active_i1))

        smneg_i = np.array([5,10,16,21,22])       # smear-neg, INH-m compartments
        smneg_i1 = smneg_i + 25
        smneg_i = np.concatenate((smneg_i,smneg_i1))
        smneg_i1 = smneg_i + 50
        smneg_i = np.concatenate((smneg_i,smneg_i1))

        active_m = np.array([11,17])              # smear-pos, MDR-TB compartments
        active_m1 = active_m + 25
        active_m = np.concatenate((active_m,active_m1))
        active_m1 = active_m + 50
        active_m = np.concatenate((active_m,active_m1))

        smneg_m = np.array([6,12,18,23,24])       # smear-neg, MDR-TB compartments
        smneg_m1 = smneg_m + 25
        smneg_m = np.concatenate((smneg_m,smneg_m1))
        smneg_m1 = smneg_m + 50
        smneg_m = np.concatenate((smneg_m,smneg_m1))

        # forces of infection:
        force_s = beta_s*(X[active_s].sum() + relbeta_sn*X[smneg_s].sum())/pop1.sum()
        force_i = beta_s*fit_i*(X[active_i].sum() + relbeta_sn*X[smneg_i].sum())/pop1.sum()
        force_m = beta_s*fit_m*(X[active_m].sum() + relbeta_sn*X[smneg_m].sum())/pop1.sum()

        # 100x100 matrix of flow rates (cut-and-pasted from above):
        chg_arr = getcharray(force_s, force_i, force_m, X)
        # chg_arr = np.zeros((tb_num,tb_num))
        i1 = np.array([0,50]) # HIV-neg only
        i2 = np.array([25,75]) # HIV-pos only
        i3 = np.array([0,25,50,75]) # all
        i4 = np.array([0,25]) # new dx
        i5 = np.array([50,75]) # retx 
        i9 = np.array([7,9,11,13,15,17])  # smear-pos
        i10 = np.array([4,5,6,8,10,12,14,16,18,19,20,21,22,23,24])  # smear-

        # 0 = incidence of new (non-retreatment) TB:
        incnew = np.sum(chg_arr[i4,i4+4]) + np.sum(chg_arr[i4,i4+5]) + np.sum(chg_arr[i4,i4+6]) + \
                 np.sum(chg_arr[i4+1,i4+4]) + np.sum(chg_arr[i4+1,i4+5]) + np.sum(chg_arr[i4+1,i4+6]) + \
                 np.sum(chg_arr[i4+2,i4+4]) + np.sum(chg_arr[i4+2,i4+5]) + np.sum(chg_arr[i4+2,i4+6]) + \
                 np.sum(chg_arr[i4+3,i4+4]) + np.sum(chg_arr[i4+3,i4+5]) + np.sum(chg_arr[i4+3,i4+6])
        # 1 = incidence of retreatment TB:
        incretx = np.sum(chg_arr[i5,i5+4]) + np.sum(chg_arr[i5,i5+5]) + np.sum(chg_arr[i5,i5+6]) + \
                 np.sum(chg_arr[i5+1,i5+4]) + np.sum(chg_arr[i5+1,i5+5]) + np.sum(chg_arr[i5+1,i5+6]) + \
                 np.sum(chg_arr[i5+2,i5+4]) + np.sum(chg_arr[i5+2,i5+5]) + np.sum(chg_arr[i5+2,i5+6]) + \
                 np.sum(chg_arr[i5+3,i5+4]) + np.sum(chg_arr[i5+3,i5+5]) + np.sum(chg_arr[i5+3,i5+6]) + \
                 np.sum(chg_arr[i4+19,i4+57]) + np.sum(chg_arr[i4+20,i4+58]) + np.sum(chg_arr[i4+21,i4+59]) + \
                 np.sum(chg_arr[i4+22,i4+60]) + np.sum(chg_arr[i4+23,i4+61]) + np.sum(chg_arr[i4+24,i4+62]) + \
                 np.sum(chg_arr[i5+19,i5+7]) + np.sum(chg_arr[i5+20,i5+8]) + np.sum(chg_arr[i5+21,i5+9]) + \
                 np.sum(chg_arr[i5+22,i5+10]) + np.sum(chg_arr[i5+23,i5+11]) + np.sum(chg_arr[i5+24,i5+12]) + \
                 np.sum((12./dur_fail)*(1-defvfail_s)*X[i3+19]) + np.sum((12./dur_fail)*(1-defvfail_s)*X[i3+20]) + \
                 np.sum((12./dur_fail)*(1-defvfail_i)*X[i3+21]) + np.sum((12./dur_fail)*(1-defvfail_i)*X[i3+22]) + \
                 np.sum((12./dur_fail)*(1-defvfail_m)*X[i3+23]) + np.sum((12./dur_fail)*(1-defvfail_m)*X[i3+24])
        # 2 = incidence of new, INH-resistant TB:
        incinhnew = np.sum(chg_arr[i4,i4+5]) + np.sum(chg_arr[i4+1,i4+5])+ + np.sum(chg_arr[i4+2,i4+5]) + \
                    np.sum(chg_arr[i4+3,i4+5])
        # 3 = incidence of retreatment, INH-resistant TB:
        incinhretx = np.sum(chg_arr[i5,i5+5]) + np.sum(chg_arr[i5+1,i5+5]) + np.sum(chg_arr[i5+2,i5+5]) + \
                     np.sum(chg_arr[i5+3,i5+5]) + np.sum(chg_arr[i4+21,i4+59]) + \
                     np.sum(chg_arr[i4+22,i4+60]) + np.sum(chg_arr[i5+21,i5+9]) + np.sum(chg_arr[i5+22,i5+10]) + \
                     np.sum((12./dur_fail)*(1-defvfail_i)*X[i3+21]) + np.sum((12./dur_fail)*(1-defvfail_i)*X[i3+22])
        # 4 = incidence of new, MDR-TB:
        incmdrnew = np.sum(chg_arr[i4,i4+6]) + np.sum(chg_arr[i4+1,i4+6])+ + np.sum(chg_arr[i4+2,i4+6]) + \
                    np.sum(chg_arr[i4+3,i4+6])
        # 5 = incidence of retreatment, MDR-TB:
        incmdrretx = np.sum(chg_arr[i5,i5+6]) + np.sum(chg_arr[i5+1,i5+6]) + np.sum(chg_arr[i5+2,i5+6]) + \
                     np.sum(chg_arr[i5+3,i5+6]) + np.sum(chg_arr[i4+23,i4+61]) + \
                     np.sum(chg_arr[i4+24,i4+62]) + np.sum(chg_arr[i5+23,i5+11]) + np.sum(chg_arr[i5+24,i5+12]) + \
                     np.sum((12./dur_fail)*(1-defvfail_m)*X[i3+23]) + np.sum((12./dur_fail)*(1-defvfail_m)*X[i3+24])
        # 6 = incidence of HIV-associated TB:
        inctbhiv = np.sum(chg_arr[i2,i2+4]) + np.sum(chg_arr[i2,i2+5]) + np.sum(chg_arr[i2,i2+6]) + \
          np.sum(chg_arr[i2+1,i2+4]) + np.sum(chg_arr[i2+1,i2+5]) + np.sum(chg_arr[i2+1,i2+6]) +\
           np.sum(chg_arr[i2+2,i2+4]) + np.sum(chg_arr[i2+2,i2+5]) + np.sum(chg_arr[i2+2,i2+6]) +\
            np.sum(chg_arr[i2+3,i2+4]) + np.sum(chg_arr[i2+3,i2+5]) + np.sum(chg_arr[i2+3,i2+6]) +\
             np.sum(chg_arr[25+19,25+57]) + np.sum(chg_arr[25+20,25+58]) + np.sum(chg_arr[25+21,25+59]) +\
              np.sum(chg_arr[25+22,25+60]) + np.sum(chg_arr[25+23,25+61]) + np.sum(chg_arr[25+24,25+62]) +\
               np.sum(chg_arr[75+19,75+7]) + np.sum(chg_arr[75+20,75+8]) + np.sum(chg_arr[75+21,75+9]) + np.sum(chg_arr[75+22,75+10]) +\
                np.sum(chg_arr[75+23,75+11]) + np.sum(chg_arr[75+24,75+12])  +\
                 np.sum((12./dur_fail)*(1-defvfail_s)*X[i2+19]) + np.sum((12./dur_fail)*(1-defvfail_s)*X[i2+20]) +\
                 np.sum((12./dur_fail)*(1-defvfail_i)*X[i2+21]) + np.sum((12./dur_fail)*(1-defvfail_i)*X[i2+22]) +\
                      np.sum((12./dur_fail)*(1-defvfail_m)*X[i2+23]) + np.sum((12./dur_fail)*(1-defvfail_m)*X[i2+24])
        # 7 = TB mortality
        TBmort = np.sum(mortality(1,0)*X[i9]) + np.sum(mortality(1,1)*X[i9+25]) + np.sum(mortality(1,0)*X[i9+50]) + \
                 np.sum(mortality(1,1)*X[i9+75]) + np.sum(mortality(2,0)*X[i10]) + np.sum(mortality(2,1)*X[i10+25]) + \
                 np.sum(mortality(2,0)*X[i10+50]) + np.sum(mortality(2,1)*X[i10+75])    

        i11 = np.arange(4,25)
        i11b = i11 + 25
        i11 = np.concatenate((i11,i11b))
        i11b = i11 + 50
        i11 = np.concatenate((i11,i11b))
        # 8 = TB prevalence
        # sum(X[i11]) includes all active cases
        # the next indent is people on 1st-line therapy (for 6 months)
        # the next indent is people on cat2 regimen (for 8 months)
        # and the final indent is people on MDR/2nd-line therapy (for 20 months)
        TBprev = np.sum(X[i11])+ 0.5*(np.sum(txrate(intopt,1,0,0,0)[1]*X[13]) + np.sum(txrate(intopt,0,0,0,0)[1]*X[14]) + \
                                   np.sum(txrate(intopt,1,1,0,0)[1]*X[15]) + np.sum(txrate(intopt,0,1,0,0)[1]*X[16]) + \
                                   np.sum(txrate(intopt,1,2,0,0)[1]*X[17]) + np.sum(txrate(intopt,0,2,0,0)[1]*X[18]) + \
                                   np.sum(txrate(intopt,1,0,0,1)[1]*X[25+13]) + np.sum(txrate(intopt,0,0,0,1)[1]*X[25+14]) + \
                                   np.sum(txrate(intopt,1,1,0,1)[1]*X[25+15]) + np.sum(txrate(intopt,0,1,0,1)[1]*X[25+16]) + \
                                   np.sum(txrate(intopt,1,2,0,1)[1]*X[25+17]) + np.sum(txrate(intopt,0,2,0,1)[1]*X[25+18])) + \
                                   (8./12.)*(np.sum(txrate(intopt,1,0,1,0)[1]*X[63]) + np.sum(txrate(intopt,0,0,1,0)[1]*X[64]) + \
                                             np.sum(txrate(intopt,1,1,1,0)[1]*X[65]) + np.sum(txrate(intopt,0,1,1,0)[1]*X[66]) + \
                                             np.sum(txrate(intopt,1,0,1,1)[1]*X[25+63]) + np.sum(txrate(intopt,0,0,1,1)[1]*X[25+64]) + \
                                             np.sum(txrate(intopt,1,1,1,1)[1]*X[25+65]) + np.sum(txrate(intopt,0,1,1,1)[1]*X[25+66]) + \
                                             np.sum((12./dur_fail)*0.5*X[i4+19]*(1-fail_rt)) + \
                                             np.sum((12./dur_fail)*0.5*X[i4+20]*(1-fail_rt)) + \
                                             np.sum((12./dur_fail)*0.5*X[i4+21]*(1-fail_i2)) + \
                                             np.sum((12./dur_fail)*0.5*X[i4+22]*(1-fail_i2)) + \
                                             np.sum((12./dur_fail)*0.5*X[i5+19]*(1-fail_rt)) + \
                                             np.sum((12./dur_fail)*0.5*X[i5+20]*(1-fail_rt)) + \
                                             np.sum((12./dur_fail)*0.5*X[i5+20]*(1-fail_rt)) + \
                                             np.sum((12./dur_fail)*0.5*X[i5+21]*(1-fail_i2)) + \
                                             np.sum((12./dur_fail)*0.5*X[i5+22]*(1-fail_i2))) + \
                                             (20./12.)*(np.sum(txrate(intopt,1,2,1,0)[1]*X[67]) + \
                                                        np.sum(txrate(intopt,0,2,1,0)[1]*X[68]) + \
                                                        np.sum(txrate(intopt,1,2,1,1)[1]*X[25+67]) + \
                                                        np.sum(txrate(intopt,0,2,1,1)[1]*X[25+68]) + \
                                                        np.sum((12./dur_fail)*0.5*X[i4+23]*(1-fail_m2)) + \
                                                        np.sum((12./dur_fail)*0.5*X[i4+24]*(1-fail_m2)) + \
                                                        np.sum((12./dur_fail)*0.5*X[i5+23]*(1-fail_m2)) + \
                                                        np.sum((12./dur_fail)*0.5*X[i5+24]*(1-fail_m2)))
        i12 = np.concatenate((np.arange(25,50), np.arange(75,100)))
        # 9 = HIV prevalence
        HIVprev = np.sum(X[i12])

        # 10 = Cost
        # First indent is diagnostic costs for people without TB.
        # Second indent is diagnostic costs.
        # Third indent is costs of inappropriate treatment (e.g., 1st-line for MDR-TB) after diagnosis
        # Fourth indent is costs of appropriate treatment
        # Fifth indent is inappropriate treatment that is made empirically
        # Sixth indent is failure: inappropriate treatment followed by appropriate treatment
        # Seventh through tenth indents are indents 3-6 but extended to people with HIV
        i13a = np.arange(4)
        i13b = i13a + 25
        i13 = np.concatenate((i13a,i13b),axis=1)
        i14a = np.arange(50,54)
        i14b = i14a + 25
        i14 = np.concatenate((i14a,i14b),axis=1)
        cost = np.sum(prop_cough*X[i13a]*dxcost(intopt,0,0))+np.sum(prop_cough*X[i14a]*dxcost(intopt,1,0)) + \
               np.sum(prop_cough*X[i13b]*dxcost(intopt,0,1))+np.sum(prop_cough*X[i14b]*dxcost(intopt,1,1)) + \
               dx_rate*(np.sum(txrate(intopt,1,0,0,0)[3]*X[7]) + np.sum(txrate(intopt,0,0,0,0)[3]*X[8]) + \
                        np.sum(txrate(intopt,1,1,0,0)[3]*X[9]) + np.sum(txrate(intopt,0,1,0,0)[3]*X[10]) + \
                        np.sum(txrate(intopt,1,2,0,0)[3]*X[11]) + np.sum(txrate(intopt,0,2,0,0)[3]*X[12]) + \
                        np.sum(txrate(intopt,1,0,1,0)[3]*X[57]) + np.sum(txrate(intopt,0,0,1,0)[3]*X[58]) + \
                        np.sum(txrate(intopt,1,1,1,0)[3]*X[59]) + np.sum(txrate(intopt,0,1,1,0)[3]*X[60]) + \
                        np.sum(txrate(intopt,1,2,1,0)[3]*X[61]) + np.sum(txrate(intopt,0,2,1,0)[3]*X[62]) + \
                        np.sum(txrate(intopt,1,0,0,1)[3]*X[25+7]) + np.sum(txrate(intopt,0,0,0,1)[3]*X[25+8]) + \
                        np.sum(txrate(intopt,1,1,0,1)[3]*X[25+9]) + np.sum(txrate(intopt,0,1,0,1)[3]*X[25+10]) + \
                        np.sum(txrate(intopt,1,2,0,1)[3]*X[25+11]) + np.sum(txrate(intopt,0,2,0,1)[3]*X[25+12]) + \
                        np.sum(txrate(intopt,1,0,1,1)[3]*X[25+57]) + np.sum(txrate(intopt,0,0,1,1)[3]*X[25+58]) + \
                        np.sum(txrate(intopt,1,1,1,1)[3]*X[25+59]) + np.sum(txrate(intopt,0,1,1,1)[3]*X[25+60]) + \
                        np.sum(txrate(intopt,1,2,1,1)[3]*X[25+61]) + np.sum(txrate(intopt,0,2,1,1)[3]*X[25+62])) + \
                    np.sum(txrate(intopt,1,0,0,0)[2]*txrate(intopt,1,0,0,0)[5]*X[7]) + \
                    np.sum(txrate(intopt,0,0,0,0)[2]*txrate(intopt,0,0,0,0)[5]*X[8]) + \
                    np.sum(txrate(intopt,1,1,0,0)[2]*txrate(intopt,1,1,0,0)[5]*X[9]) + \
                    np.sum(txrate(intopt,0,1,0,0)[2]*txrate(intopt,0,1,0,0)[5]*X[10]) + \
                    np.sum(txrate(intopt,1,2,0,0)[2]*txrate(intopt,1,2,0,0)[5]*X[11]) + \
                    np.sum(txrate(intopt,0,2,0,0)[2]*txrate(intopt,0,2,0,0)[5]*X[12]) + \
                    np.sum(txrate(intopt,1,0,1,0)[2]*txrate(intopt,1,0,1,0)[5]*X[57]) + \
                    np.sum(txrate(intopt,0,0,1,0)[2]*txrate(intopt,0,0,1,0)[5]*X[58]) + \
                    np.sum(txrate(intopt,1,1,1,0)[2]*txrate(intopt,1,1,1,0)[5]*X[59]) + \
                    np.sum(txrate(intopt,0,1,1,0)[2]*txrate(intopt,0,1,1,0)[5]*X[60]) + \
                    np.sum(txrate(intopt,1,2,1,0)[2]*txrate(intopt,1,2,1,0)[5]*X[61]) + \
                    np.sum(txrate(intopt,0,2,1,0)[2]*txrate(intopt,0,2,1,0)[5]*X[62]) + \
                        np.sum(txrate(intopt,1,0,0,0)[0]*txrate(intopt,1,0,0,0)[4]*X[7]) + \
                        np.sum(txrate(intopt,0,0,0,0)[0]*txrate(intopt,0,0,0,0)[4]*X[8]) + \
                        np.sum(txrate(intopt,1,1,0,0)[0]*txrate(intopt,1,1,0,0)[4]*X[9]) + \
                        np.sum(txrate(intopt,0,1,0,0)[0]*txrate(intopt,0,1,0,0)[4]*X[10]) + \
                        np.sum(txrate(intopt,1,2,0,0)[0]*txrate(intopt,1,2,0,0)[4]*X[11]) + \
                        np.sum(txrate(intopt,0,2,0,0)[0]*txrate(intopt,0,2,0,0)[4]*X[12]) + \
                        np.sum(txrate(intopt,1,0,1,0)[0]*txrate(intopt,1,0,1,0)[4]*X[57]) + \
                        np.sum(txrate(intopt,0,0,1,0)[0]*txrate(intopt,0,0,1,0)[4]*X[58]) + \
                        np.sum(txrate(intopt,1,1,1,0)[0]*txrate(intopt,1,1,1,0)[4]*X[59]) + \
                        np.sum(txrate(intopt,0,1,1,0)[0]*txrate(intopt,0,1,1,0)[4]*X[60]) + \
                        np.sum(txrate(intopt,1,2,1,0)[0]*txrate(intopt,1,2,1,0)[4]*X[61]) + \
                        np.sum(txrate(intopt,0,2,1,0)[0]*txrate(intopt,0,2,1,0)[4]*X[62]) - \
                    np.sum(chg_arr[19,57]*txrate(intopt,1,0,0,0)[5]) - \
                    np.sum(chg_arr[20,58]*txrate(intopt,0,0,0,0)[5]) - \
                    np.sum(chg_arr[21,59]*txrate(intopt,1,1,0,0)[5]) - \
                    np.sum(chg_arr[22,60]*txrate(intopt,0,1,0,0)[5]) - \
                    np.sum(chg_arr[23,61]*txrate(intopt,1,2,0,0)[5]) - \
                    np.sum(chg_arr[24,62]*txrate(intopt,0,2,0,0)[5]) - \
                    np.sum(chg_arr[69,57]*txrate(intopt,1,0,1,0)[5]) - \
                    np.sum(chg_arr[70,58]*txrate(intopt,0,0,1,0)[5]) - \
                    np.sum(chg_arr[71,59]*txrate(intopt,1,1,1,0)[5]) - \
                    np.sum(chg_arr[72,60]*txrate(intopt,0,1,1,0)[5]) - \
                    np.sum(chg_arr[73,61]*txrate(intopt,1,2,1,0)[5]) - \
                    np.sum(chg_arr[74,62]*txrate(intopt,0,2,1,0)[5]) + \
                        np.sum((txrate(intopt,1,0,1,0)[4]+txrate(intopt,1,2,1,0)[3])*(12./dur_fail)*(1-defvfail_s)*X[i1+19]) + \
                        np.sum((txrate(intopt,0,0,1,0)[4]+txrate(intopt,1,2,1,0)[3])*(12./dur_fail)*(1-defvfail_s)*X[i1+20]) + \
                        np.sum((txrate(intopt,1,1,1,0)[4]+txrate(intopt,1,2,1,0)[3])*(12./dur_fail)*(1-defvfail_s)*X[i1+21]) + \
                        np.sum((txrate(intopt,0,1,1,0)[4]+txrate(intopt,1,2,1,0)[3])*(12./dur_fail)*(1-defvfail_s)*X[i1+22]) + \
                        np.sum((txrate(intopt,1,2,1,0)[4]+txrate(intopt,1,2,1,0)[3])*(12./dur_fail)*(1-defvfail_s)*X[i1+23]) + \
                        np.sum((txrate(intopt,0,2,1,0)[4]+txrate(intopt,1,2,1,0)[3])*(12./dur_fail)*(1-defvfail_s)*X[i1+24]) + \
                    np.sum(txrate(intopt,1,0,0,1)[2]*txrate(intopt,1,0,0,1)[5]*X[7+25]) + \
                    np.sum(txrate(intopt,0,0,0,1)[2]*txrate(intopt,0,0,0,1)[5]*X[8+25]) + \
                    np.sum(txrate(intopt,1,1,0,1)[2]*txrate(intopt,1,1,0,1)[5]*X[9+25]) + \
                    np.sum(txrate(intopt,0,1,0,1)[2]*txrate(intopt,0,1,0,1)[5]*X[10+25]) + \
                    np.sum(txrate(intopt,1,2,0,1)[2]*txrate(intopt,1,2,0,1)[5]*X[11+25]) + \
                    np.sum(txrate(intopt,0,2,0,1)[2]*txrate(intopt,0,2,0,1)[5]*X[12+25]) + \
                    np.sum(txrate(intopt,1,0,1,1)[2]*txrate(intopt,1,0,1,1)[5]*X[57+25]) + \
                    np.sum(txrate(intopt,0,0,1,1)[2]*txrate(intopt,0,0,1,1)[5]*X[58+25]) + \
                    np.sum(txrate(intopt,1,1,1,1)[2]*txrate(intopt,1,1,1,1)[5]*X[59+25]) + \
                    np.sum(txrate(intopt,0,1,1,1)[2]*txrate(intopt,0,1,1,1)[5]*X[60+25]) + \
                    np.sum(txrate(intopt,1,2,1,1)[2]*txrate(intopt,1,2,1,1)[5]*X[61+25]) + \
                    np.sum(txrate(intopt,0,2,1,1)[2]*txrate(intopt,0,2,1,1)[5]*X[62+25]) + \
                        np.sum(txrate(intopt,1,0,0,1)[0]*txrate(intopt,1,0,0,1)[4]*X[7+25]) + \
                        np.sum(txrate(intopt,0,0,0,1)[0]*txrate(intopt,0,0,0,1)[4]*X[8+25]) + \
                        np.sum(txrate(intopt,1,1,0,1)[0]*txrate(intopt,1,1,0,1)[4]*X[9+25]) + \
                        np.sum(txrate(intopt,0,1,0,1)[0]*txrate(intopt,0,1,0,1)[4]*X[10+25]) + \
                        np.sum(txrate(intopt,1,2,0,1)[0]*txrate(intopt,1,2,0,1)[4]*X[11+25]) + \
                        np.sum(txrate(intopt,0,2,0,1)[0]*txrate(intopt,0,2,0,1)[4]*X[12+25]) + \
                        np.sum(txrate(intopt,1,0,1,1)[0]*txrate(intopt,1,0,1,1)[4]*X[57+25]) + \
                        np.sum(txrate(intopt,0,0,1,1)[0]*txrate(intopt,0,0,1,1)[4]*X[58+25]) + \
                        np.sum(txrate(intopt,1,1,1,1)[0]*txrate(intopt,1,1,1,1)[4]*X[59+25]) + \
                        np.sum(txrate(intopt,0,1,1,1)[0]*txrate(intopt,0,1,1,1)[4]*X[60+25]) + \
                        np.sum(txrate(intopt,1,2,1,1)[0]*txrate(intopt,1,2,1,1)[4]*X[61+25]) + \
                        np.sum(txrate(intopt,0,2,1,1)[0]*txrate(intopt,0,2,1,1)[4]*X[62+25]) - \
                    np.sum(chg_arr[19+25,57+25]*txrate(intopt,1,0,0,1)[5]) - \
                    np.sum(chg_arr[20+25,58+25]*txrate(intopt,0,0,0,1)[5]) - \
                    np.sum(chg_arr[21+25,59+25]*txrate(intopt,1,1,0,1)[5]) - \
                    np.sum(chg_arr[22+25,60+25]*txrate(intopt,0,1,0,1)[5]) - \
                    np.sum(chg_arr[23+25,61+25]*txrate(intopt,1,2,0,1)[5]) - \
                    np.sum(chg_arr[24+25,62+25]*txrate(intopt,0,2,0,1)[5]) - \
                    np.sum(chg_arr[69+25,57+25]*txrate(intopt,1,0,1,1)[5]) - \
                    np.sum(chg_arr[70+25,58+25]*txrate(intopt,0,0,1,1)[5]) - \
                    np.sum(chg_arr[71+25,59+25]*txrate(intopt,1,1,1,1)[5]) - \
                    np.sum(chg_arr[72+25,60+25]*txrate(intopt,0,1,1,1)[5]) - \
                    np.sum(chg_arr[73+25,61+25]*txrate(intopt,1,2,1,1)[5]) - \
                    np.sum(chg_arr[74+25,62+25]*txrate(intopt,0,2,1,1)[5]) + \
                        np.sum((txrate(intopt,1,0,1,1)[4]+txrate(intopt,1,2,1,1)[3])*(12./dur_fail)*(1-defvfail_s)*X[i2+19]) + \
                        np.sum((txrate(intopt,0,0,1,1)[4]+txrate(intopt,1,2,1,1)[3])*(12./dur_fail)*(1-defvfail_s)*X[i2+20]) + \
                        np.sum((txrate(intopt,1,1,1,1)[4]+txrate(intopt,1,2,1,1)[3])*(12./dur_fail)*(1-defvfail_s)*X[i2+21]) + \
                        np.sum((txrate(intopt,0,1,1,1)[4]+txrate(intopt,1,2,1,1)[3])*(12./dur_fail)*(1-defvfail_s)*X[i2+22]) + \
                        np.sum((txrate(intopt,1,2,1,1)[4]+txrate(intopt,1,2,1,1)[3])*(12./dur_fail)*(1-defvfail_s)*X[i2+23]) + \
                        np.sum((txrate(intopt,0,2,1,1)[4]+txrate(intopt,1,2,1,1)[3])*(12./dur_fail)*(1-defvfail_s)*X[i2+24])  
        return incnew, incretx, incinhnew, incinhretx, incmdrnew, incmdrretx, inctbhiv, TBmort, TBprev, HIVprev, cost

    ###########################################################
    # 9. LOW INCIDENCE AND EMERGING MDR SCENARIO CALCULATOR
    ###########################################################

    # One challenge is that equilibrium scenarios result in about 80% of all incident TB being due to recent infection.
    # Low incidence settings have more TB due to reactivation and thus cannot be modeled at equilibrium,
    #   as the proportion of incident TB reflecting reactivation is a key driver of impact.
    # This section allows the model to generate low-incidence scenarios that appropriately have a lower proportion of
    #   TB due to recent infection, with a higher proportion due to reactivation.

    # If the target incidence is <50 per 100,000, the program generates an equilibrium 50 years in the past
    #    with an incidence of 50 per 100,000/yr.
    # It then reduces the transmission parameter by a set proportion at that time, searching until it finds the
    #    reduction that allows it to re-create the target incidence as specified by the user.
    # This allows the program to generate a situation in which the majority of incident TB is due to
    #    reactivation rather than recent infection.

    equil_round = 0.
    if target_inc2 < 50:
        beta_s *= 1. + (target_inc2 - 50.)/82.
        while equil_round<50:
            pre_time = 50.
            time_burnin = np.arange(0,pre_time+1.,1.)
            EQUI = odeint(diffeq,equipop,time_burnin)
            if abs(incprevmort(EQUI[50,:])[0] + incprevmort(EQUI[50,:])[1] -target_inc2)<0.05:
                break
            print "calculating low-incidence scenario..."
            print equil_round, beta_s, incprevmort(EQUI[50,:])[0] + incprevmort(EQUI[50,:])[1]
            beta_s += (target_inc2 - incprevmort(EQUI[50,:])[0] - incprevmort(EQUI[50,:])[1])/(0.25*target_inc2)
            equil_round +=1
    if target_inc2<50:
        equipop = EQUI[50,:]


    # Below also allows for an emerging MDR scenario, in which the final MDR-TB prevalence is
    #    higher at the end of 5 years than at the beginning.

    equil_round = 0.
    if target_mdr2 > target_mdr:
        fit_m *= 1. + ((target_mdr2 - target_mdr)/0.25)*(1./fit_m)
        while equil_round<50:
            pre_time2 = 5.
            time_burnin2 = np.arange(0,pre_time2+0.1,0.1)
            EQUI2 = odeint(diffeq,equipop,time_burnin2)
            if abs(incprevmort(EQUI2[50,:])[4]/incprevmort(EQUI2[50,:])[0]-target_mdr2)<0.0005:
                break
            print "calculating emerging MDR scenario..."
            print equil_round, fit_m, incprevmort(EQUI2[50,:])[4]/incprevmort(EQUI2[50,:])[0]
            fit_m += (target_mdr2 - incprevmort(EQUI2[50,:])[4]/incprevmort(EQUI2[50,:])[0])/(5.*target_mdr2)
            equil_round +=1


    ###########################################################
    # 10. SOLVE DIFFERENTIAL EQUATIONS AND REPORT RESULTS
    ###########################################################
    duration = 5.                        # 5 year time horizon
    timestep = 0.01                      # calculate equations every 0.01 yrs
    time_range = np.arange(0, duration+timestep, timestep) # vector for calculations
    OUT = odeint(diffeq,equipop,time_range)  # solve the ODE's
    incvect = incprevmort(OUT[500,:])    # calculate the incidence/prevalence/mortality/cost at the end of year 5
    #Baseline
    as_data['bl']['lines'].append("%.1f per 100,000" % (np.around(incvect[0],decimals=1)))
    as_data['bl']['lines'].append("%.1f per 100,000" % (np.around(incvect[1],decimals=1)))
    as_data['bl']['lines'].append("%.1f per 100,000" % (np.around(incvect[0] + incvect[1],decimals=1)))
    as_data['bl']['lines'].append("%.1f%s" % (np.around(incvect[2]/incvect[0]*100,decimals=1),"%"))
    as_data['bl']['lines'].append("%.1f%s" % (np.around(incvect[3]/incvect[1]*100,decimals=1),"%"))
    as_data['bl']['lines'].append("%.1f%s" % (np.around(incvect[4]/incvect[0]*100,decimals=2),"%"))
    as_data['bl']['lines'].append("%.2f%s" % (np.around(incvect[5]/incvect[1]*100,decimals=2),"%"))
    as_data['bl']['lines'].append("%.1f per 100,000" % (np.around(incvect[4]+incvect[5],decimals=2)))
    as_data['bl']['lines'].append("%.1f%s" % (np.around(incvect[6]/(incvect[0]+incvect[1])*100,decimals=1),"%"))
    as_data['bl']['lines'].append("%.1f per 100,000" % (np.around(incvect[7],decimals=1)))
    as_data['bl']['lines'].append("%.2f years" % (np.around(incvect[8]/(incvect[0]+incvect[1]),decimals=2)))
    as_data['bl']['lines'].append("%.1f%s" % (np.around(incvect[9]/1000,decimals=1),"%"))
    # PD MOD 2012/5/20 costvect = np.array( [ incprevmort(row)[10] for row in OUT[1:100,:] ] )
    #costvect = incprevmort(OUT[100,:])[10] #PD MOD 2012/5/20
    costvect = np.zeros((100)) #JP DD MOD 2013/8/13
    for x in range(100):
        costvect[x] = incprevmort(OUT[1+x,:])[10]
    # PD MOD 2012/5/20 print "<li>Cost in Year 1: $%.2f</li>" % (np.around(np.sum(costvect[:])/100.))
#    as_data['bl']['lines'].append("$%.0f" % (np.around(costvect))) #PD MOD 2012/5/20
#    as_data['bl']['lines'].append("$%.0f" % (np.around(incvect[10]))) #PD MOD 2012/5/20
# Added thousands separator JP (2013/08/05)
    as_data['bl']['lines'].append("${:0,}".format(np.around(np.sum(costvect[:])/100.))[:-2])
    as_data['bl']['lines'].append("${:0,}".format(np.around(incvect[10]))[:-2])

        
    as_data['status'] = 1
    as_writeJSON(tmp_path, as_data) #Write out baseline results to tmp file

    impactarray = np.zeros(126)
    #costvect3 = np.zeros(500) PD MOD 2012/5/20
    #incvect3 = np.zeros(500) PD MOD 2012/5/20
    incredn5 = np.zeros(9)
    mdrredn5 = np.zeros(9)
    cmore5 = np.zeros(9)
    #Values for 3rd graph
    g_points = []
    if int_select==9:
        #For multiple strats to display in the proper order
        mapping = [0,1,7,6,2,8,3,4,5]
        #will change all references to abc in the 'for' loop to mapping[abc] (via intopt)
        #indicate which lines are modified with #was abc
        ###################################################
        for abc in range(9):
            #print "<div id='strat%d'>" % (abc)
            intopt = mapping[abc] #was abc
            OUT = odeint(diffeq,equipop,time_range)  # solve the ODE's from the same equilibrium population
            #print "INTERVENTION", intopt
            #print "<h1>Selected Intervention:</h1>"
            #print "<span>%d. %s</span><ul>" % (disp_order[abc],txt_strat[abc])
            incvect2 = incprevmort(OUT[500,:])
            
            wh = 'strat%d' % (intopt) # was (abc)        

            as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[0],decimals=1)))
            as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[1],decimals=1)))
            as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[0] + incvect2[1],decimals=1)))
            incredn5[intopt] = (1-(incvect2[0] + incvect2[1])/(incvect[0] + incvect[1]))*100 #was abc
            as_data[wh]['lines'].append("%.1f%s reduction" % (np.around(incredn5[intopt],decimals=1),'%')) #was abc
           
            as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[2]/incvect2[0]*100,decimals=1),"%"))
            as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[3]/incvect2[1]*100,decimals=1),"%"))
            as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[4]/incvect2[0]*100,decimals=1),"%"))
            as_data[wh]['lines'].append("%.2f%s" % (np.around(incvect2[5]/incvect2[1]*100,decimals=2),"%"))
            as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[4]+incvect2[5],decimals=1)))
            mdrredn5[intopt]=(1-(incvect2[4] + incvect2[5])/(incvect[4] + incvect[5]))*100 #was abc
            as_data[wh]['lines'].append("%.1f%s reduction" % (np.around(mdrredn5[intopt],decimals=1),'%')) #was abc

            as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[6]/(incvect2[0]+incvect2[1])*100,decimals=1),"%"))
            as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[7],decimals=1)))
            as_data[wh]['lines'].append("%.1f%s reduction" % (np.around((1-(incvect2[7])/(incvect[7]))*100,decimals=1),'%'))

            as_data[wh]['lines'].append("%.2f years" % (np.around(incvect2[8]/(incvect2[0]+incvect2[1]),decimals=2)))
            as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[9]/1000,decimals=1),"%"))
  
            # now also calculate costs in year 1 (from timestep 0 to timestep 99)
            # PD MOD 2012/5/20 ipm = np.array( [incprevmort(row) for row in OUT] ) # calculate costs etc over all 500 times
            ipm = incprevmort(OUT[100,:]) # PD MOD 2012/5/20
            #costvect3 = ipm[10] #PD MOD 2012/5/20
            costvect3 = np.zeros((100))
            for x in range(100): #JP DD MOD 2013/8/13
                costvect3[x] = incprevmort(OUT[1+x,:])[10]
            incvect3 = ipm[0] + ipm[1] #PD MOD 2012/5/20
            #costvect2 = costvect3[1:100] # only timestep 0 to 99 #PD MOD 2012/5/20
            #as_data[wh]['lines'].append("${:0,}".format(np.around(costvect3))[:-2])
            as_data[wh]['lines'].append("${:0,}".format(np.around(np.sum(costvect3[:])/100.))[:-2])
            #Added thousands separator JP (2013/08/05)
            #as_data[wh]['lines'].append("%.1f%s increase" % (np.around(((costvect3-costvect)/(costvect))* 100,decimals=1),'%')) #PD MOD 2012/5/20
            as_data[wh]['lines'].append("%.1f%s increase" % (np.around(((np.sum(costvect3[:])/100)-(np.sum(costvect[:])/100.))/(np.sum(costvect[:])/100.)*100,decimals=1),'%')) #JP DD MOD 203/8/13
            as_data[wh]['lines'].append("${:0,}".format(np.around(incvect2[10]))[:-2])
            if incvect2[10]>incvect[10]:
                as_data[wh]['lines'].append("%.1f%s increase" % (np.around(((incvect2[10])/(incvect[10])-1)*100,decimals=1),'%'))
            if incvect2[10]<=incvect[10]:
                as_data[wh]['lines'].append("%.1f%s decrease" % (np.around((1-(incvect2[10])/(incvect[10]))*100,decimals=1),"%"))

            #write the new entry to the file
            as_data['status'] = as_data['status'] + 1
            as_writeJSON(tmp_path,as_data)

            #Graphing and csv
            cmore5[intopt] = ((incvect2[10])/(incvect[10])-1)*100 #was abc
            #impactarray[intopt*14:intopt*14+11] = incvect2[0:11] #was abc
            #impactarray[intopt*14+11] = costvect #PD MOD 2012/5/20 #was abc
            #impactarray[intopt*14+12] = costvect3 #PD MOD 2012/5/20 #was abc
            #impactarray[intopt*14+13] = incvect3 #PD MOD 2012/5/20 #was abc

            #3rd Graph
            g_points.append ( (round ((1-(incvect2[0] + incvect2[1]) / (incvect[0] + incvect[1])) *(-100), 1),
                               round ((1-(incvect2[4] + incvect2[5]) / (incvect[4] + incvect[5])) *(-100), 1),
                               round ((1-(incvect2[7]/incvect[7])) * (-100), 1),
                               np.around(((np.sum(costvect3[:])/100)-(np.sum(costvect[:])/100.))/(np.sum(costvect[:])/100.)*100,decimals=1), #PD MOD 2012/5/20 JP Changed back 2013/08/13
                               round (((incvect2[10])/(incvect[10])-1) * 100, 1) ) )
                
        #Graphing

        tmp_str = ''.join([ "&c%d=%.2f" % (i,cmore5[i]) for i in xrange(9)])
        inc_str= ''.join([ "&i%d=%.2f" % (i,incredn5[i]) for i in xrange(9)])
        mdr_str= ''.join([ "&m%d=%.2f" % (i,mdrredn5[i]) for i in xrange(9)])
        tmp_gpnts_str=''.join([ "&d%d%d=%s" % (x,y,str(g_points[x][y])) for x in xrange(9) for y in xrange(5)])
        cm_str = "?" + tmp_str[1:]
        gpnts_str = "?" + tmp_gpnts_str[1:]
        as_data['strat9']['graphs'].append("/dgraph1.png%s%s" % (cm_str,inc_str))
        as_data['strat9']['graphs'].append("/dgraph2.png%s%s" % (cm_str,mdr_str))
        as_data['strat9']['graphs'].append("/bargraph.png%s" % (gpnts_str))
        
        #Write the graph data to the file
        as_data['status'] = as_data['status'] + 1
        as_writeJSON(tmp_path, as_data)

    if int_select<9: 
        intopt = int_select                 # change the intervention to the user-defined intervention
        OUT = odeint(diffeq,equipop,time_range)  # solve the ODE's from the same equilibrium population
        #print "<h1>Selected Intervention:</h1>"
        #print "<span>%d. %s</span><ul>" % (disp_order[intopt],txt_strat[intopt])
        incvect2 = incprevmort(OUT[500,:])
        wh = 'strat%d' % (int_select)

        as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[0],decimals=1)))
        as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[1],decimals=1)))
        as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[0] + incvect2[1],decimals=1)))
        as_data[wh]['lines'].append("%.1f%s reduction" % (np.around((1-(incvect2[0] + incvect2[1])/(incvect[0] + incvect[1]))*100,decimals=1),'%'))
        
        as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[2]/incvect2[0]*100,decimals=1),"%"))
        as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[3]/incvect2[1]*100,decimals=1),"%"))
        as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[4]/incvect2[0]*100,decimals=1),"%"))
        as_data[wh]['lines'].append("%.2f%s" % (np.around(incvect2[5]/incvect2[1]*100,decimals=2),"%"))
        as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[4]+incvect2[5],decimals=1)))
        as_data[wh]['lines'].append("%.1f%s reduction" % (np.around((1-(incvect2[4] + incvect2[5])/(incvect[4] + incvect[5]))*100,decimals=1),'%'))

        as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[6]/(incvect2[0]+incvect2[1])*100,decimals=1),"%"))
        as_data[wh]['lines'].append("%.1f per 100,000" % (np.around(incvect2[7],decimals=1)))
        as_data[wh]['lines'].append("%.1f%s reduction" % (np.around((1-(incvect2[7])/(incvect[7]))*100,decimals=1),'%'))

        as_data[wh]['lines'].append("%.2f years" % (np.around(incvect2[8]/(incvect2[0]+incvect2[1]),decimals=2)))
        as_data[wh]['lines'].append("%.1f%s" % (np.around(incvect2[9]/1000,decimals=1),"%"))

        # now also calculate costs in year 1 (from timestep 0 to timestep 99)
        #costvect2 = np.array( [ incprevmort(row)[10] for row in OUT[1:100,:] ] ) PD MOD 2012/5/20
        costvect2 = np.zeros((100))
        for x in range(100): #JP DD MOD 2013/8/13
            costvect2[x] = incprevmort(OUT[1+x,:])[10]

        #costvect2 = incprevmort(OUT[100,:])[10] #PD MOD 2012/5/20
        # Added thousands separator JP (2013/08/05)
        #as_data[wh]['lines'].append("${:0,}".format(np.around(costvect2))[:-2])
        as_data[wh]['lines'].append("${:0,}".format(np.around(np.sum(costvect2[:])/100.))[:-2])
        #as_data[wh]['lines'].append("%.1f%s increase" % (np.around(((costvect2-costvect)/(costvect)) * 100,decimals=1),'%')) #PD MOD 2012/5/20
        as_data[wh]['lines'].append("%.1f%s increase" % (np.around(((np.sum(costvect2[:])/100)-(np.sum(costvect[:])/100.))/(np.sum(costvect[:])/100.)*100,decimals=1),'%'))
        as_data[wh]['lines'].append("${:0,}".format(np.around(incvect2[10]))[:-2])
        if incvect2[10]>incvect[10]:
            as_data[wh]['lines'].append("%.1f%s increase" % (np.around(((incvect2[10])/(incvect[10])-1)*100,decimals=1),'%'))
        if incvect2[10]<=incvect[10]:
            as_data[wh]['lines'].append("%.1f%s decrease" % (np.around((1-(incvect2[10])/(incvect[10]))*100,decimals=1),"%"))

        #write the new entry to the file
        as_data['status'] = as_data['status'] + 1
        as_writeJSON(tmp_path,as_data)

def as_writeJSON ( tmp_path, data ):

    fd = open(tmp_path, "w+b")
    s = json.dumps(data,separators=(',',':'))
    fd.write(s)
    fd.close()

if __name__=='__main__':

    if len(sys.argv) != 17:
        print "Incorrect number of args"
        exit(-1)

    as_tb_inc = float(sys.argv[1])
    as_hiv_inc = float(sys.argv[2])
    as_mdr_inc = float(sys.argv[3])
    as_cost = float(sys.argv[4])
    as_sm_cost = float(sys.argv[5])    #
    as_gxp_cost = float(sys.argv[6])   # 
    as_cx_cost = float(sys.argv[7])    #
    as_mods_cost = float(sys.argv[8])  #
    as_dst_cost = float(sys.argv[9])   #
    as_sd_cost = float(sys.argv[10])   #
    as_sdgxp_cost = float(sys.argv[11])#
    as_drug2_cost = float(sys.argv[12])#
    as_drug3_cost = float(sys.argv[13])#
    as_outpt_cost = float(sys.argv[14])#
    as_int_select = int(sys.argv[15])
    as_tmp_path = sys.argv[16]

    #print "Inc:%f Hiv:%f MDR:%f Cost:%f Run:%d" % (as_tb_inc,as_hiv_inc,as_mdr_inc,as_cost,as_int_select) 

    as_data = { 'status':0 , 'bl' : {'name':'Baseline (Smear)','lines': []},
                             'strat0' : {'name':'Smear', 'lines':[]},
                             'strat1' : {'name':'Culture for Retreatment','lines':[]},
                             'strat2' : {'name':'GeneXpert for all','lines':[]},
                             'strat3' : {'name':'MODS/TLA','lines':[]},
                             'strat4' : {'name':'Same-day smear microscopy','lines':[]},
                             'strat5' : {'name':'Same-day GeneXpert','lines':[]},
                             'strat6' : {'name':'GeneXpert for smear positive only','lines':[]},
                             'strat7' : {'name':'GeneXpert for HIV positive only','lines':[]},
                             'strat8' : {'name':'GeneXpert for all, culture confirmed','lines':[]},
                             'strat9' : {'graphs':[]},
                'baseline' : [ 'Incidence, new:','Incidence, retx:','Incidence, total:','Incidence, INH new:',
                               'Incidence, INH retx:','Incidence, MDR new','Incidence, MDR retx:',
                               'Incidence, MDR total:','Incidence, TB/HIV:','TB mortality:','TB duration:',
                               'HIV prevalence:','Cost per year at End of Year 1: ','Cost per year at End of Year 5: '],
                'strats' : [ 'Incidence, new:','Incidence, retx:','Incidence, total:','\t','Incidence, INH new:',
                             'Incidence, INH retx:','Incidence, MDR new','Incidence, MDR retx:',
                             'Incidence, MDR total:','\t','Incidence, TB/HIV:','TB mortality:','\t','TB duration:',
                             'HIV prevalence:','Cost per year at End of Year 1: ','\t','Cost per year at End of Year 5: ',
                             '\t' ],
                'run_type' : as_int_select }

    as_writeJSON(as_tmp_path, as_data) #Write initial framework anticipating JQuery calls.
    
    runmodel (as_tb_inc, 
              as_hiv_inc, 
              as_mdr_inc / 100, 
              as_cost, 
              as_sm_cost,
              as_gxp_cost,
              as_cx_cost,
              as_mods_cost,
              as_dst_cost,
              as_sd_cost,
              as_sdgxp_cost,
              as_drug2_cost,
              as_drug3_cost,
              as_outpt_cost,
              as_int_select, 
              as_data, as_tmp_path)


