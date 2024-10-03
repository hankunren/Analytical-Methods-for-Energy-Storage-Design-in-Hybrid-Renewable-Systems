# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:19:52 2023

@author: orie4058
"""

import gurobipy as gp
from gurobipy import GRB
import pandapower as pp
import numpy as np
import pandas as pd
import time

#Import data from excel and put them into constant variables.
DATA = pd.read_excel("Storage.xlsx")
WIND = pd.read_excel(r"Wind.xlsx", index_col=0)
LOAD = pd.read_excel(r"Load.xlsx", index_col=0)
CHARGE_EFFICIENCY = DATA["Charge Efficiency"][0]
DISCHARGE_EFFICIENCY = DATA["Discharge Efficiency"][0]
LEAKAGE_RATE = DATA["Leakage Rate"][0]
MAX_DOD = DATA["Maximum DoD"][0]
MIN_DOD = DATA["Minimum DoD"][0]
CHARGE_C_RATING = DATA["Charge C Rating"][0]
DISCHARGE_C_RATING = DATA["Discharge C Rating"][0]
SIGNIFICANT_FIGURES = DATA["Significant Figures"][0]
MULTIPLIER = 0.5

def create_network():
    #Create the IEEE 24 bus reliability test system.
    net = pp.networks.case24_ieee_rts()
    
    #Add storage at each bus.
    for new_bus in net.bus.index:
        new_storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 0, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)

    #Add wind farms at bus 3, 5, 7, 16, 21 and 23.
    pp.create_sgen(net, 2, type='wind', p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 4, type='wind', p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 6, type='wind', p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 15, type='wind',p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 20, type='wind',p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 22, type='wind',p_mw=0., q_mvar=0, controllable=True)
    
    #net.ext_grid['max_p_mw']=float('inf')
    net.ext_grid['min_p_mw']= -197
    net.poly_cost.loc[3,'cp0_eur']=0
    net.poly_cost.loc[3,'cp1_eur_per_mw']=0
    net.poly_cost.loc[3,'cp2_eur_per_mw2'] = 1

    return net


def run_optimal_power_flow (starting_storage_level, max_charge, max_discharge, max_storage_constraint, min_storage_constraint):

    for storage_index in net.storage.index:
        
        for time_step in time_step_index:
            #Minimum storage level <= Storage level <= maximum storage level.
            upper_storage_constraint[storage_index][time_step].rhs = max_storage_constraint[storage_index]
            lower_storage_constraint[storage_index][time_step].rhs = min_storage_constraint[storage_index]
        
        for time_step in time_period_index:
            #-(maximum discharge limit) <= storage power <= maximum charge limit.
            max_charge_constraint[storage_index][time_step].rhs = max_charge[storage_index]
            max_discharge_constraint[storage_index][time_step].rhs = max_discharge[storage_index]
    
        #Starting storage level.
        starting_storage_level_constraint[storage_index].rhs = starting_storage_level[storage_index]
        
    model.update() 
    model.read(r"solution.sol")
    model.optimize()
            
    storage_profile_result = np.zeros((len(net.storage.index),len(time_step_index)))
    for i in net.storage.index:
        for j in time_step_index:
            storage_profile_result[i,j] = storage_profile[i, j].X
    
    return model.ObjVal, storage_profile_result


def size_storage (storage_profile, difference):
    #Find critical points by comparing with neighbouring points, np.logical_or returns a boolean array, np.where return a tuple of indexes where boolean is true, add [0] to return an array, add 1 to correct the index.
    critical_point=np.where(np.logical_or(
        (storage_profile[0:-2] <= storage_profile[1:-1]) * (storage_profile[1:-1] > storage_profile[2:]),
        (storage_profile[0:-2] >= storage_profile[1:-1]) * (storage_profile[1:-1] < storage_profile[2:])
        ))[0] + 1
    #Number of critical points.
    number_of_critical = critical_point.shape[0]
    #Obatin critical storage levels from the profile, convert 1D array into 2D array with both row and column dimensions.
    critical_storage_level = storage_profile[critical_point].reshape(number_of_critical, 1)
    #Convert vertical array into horizontal array.
    critical_storage_level_horizontal = critical_storage_level.reshape(1, number_of_critical)
    #Create lower triangular matrix with the difference value between start and end storage levels.
    lower_triangular_matrix = np.tril(np.full((number_of_critical, number_of_critical), difference), -1)
    #Difference matrix = lower triangular matrix + critical storage level matrix - transpose of critical storage level matrix.
    difference_matrix = lower_triangular_matrix + np.repeat(critical_storage_level_horizontal, number_of_critical, axis=0) - critical_storage_level
    #When there is no critical point, set matrix equal to zero.
    if number_of_critical == 0: difference_matrix = [0]
    
    #Increasing scenario.
    if difference > 0:
        #Storage size is equal to absolute of the difference matrix's minimum, or the largest discharge the storage can experience.
        storage_size = np.abs(np.amin(difference_matrix))
    #Decreasing scenario.
    elif difference < 0:
        #Storage size is equal to maximum of the difference matrix, or the largest charge the storage can experience.
        storage_size = np.amax(difference_matrix)
    #Neither increasing nor decreasing.
    else:
        #Storage size is equal to the maximum of the difference matrix, or the largest energy from either charge or discharge.
        storage_size = np.amax(np.abs(difference_matrix))
    
    return storage_size
    

def storage_parameter (profile):
    #Calculate the difference between end and start storage levels.
    difference = profile[-2] - profile[0]
    
    #Size energy storage, note in numpy, calling a row will give a column vector of that row.
    storage_size = size_storage(profile, difference)
    #Storage capacity is storage size divided by the depth of discharge.
    storage_capacity = storage_size / (MAX_DOD - MIN_DOD)
    
    #Maximum storage limit of the usable storage.
    max_storage_limit = storage_capacity * (1 - MIN_DOD)
    #Minimum storage limit of the usable storage.
    min_storage_limit = storage_capacity * (1 - MAX_DOD)
    
    #Maximum storage level constraint.
    max_storage_constraint = max_storage_limit + MULTIPLIER * np.abs(difference)
    #Minimum storage level constraint.
    min_storage_constraint = min_storage_limit - MULTIPLIER * np.abs(difference)

    #Maximum charge rate is related to storage energy capacity via the charge C-rating.
    max_charge = storage_capacity * CHARGE_C_RATING + MULTIPLIER * np.abs(difference)
    #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
    max_discharge = -1 * storage_capacity * DISCHARGE_C_RATING - MULTIPLIER * np.abs(difference)  

    #Decreasing scenario.        
    if difference > 0:
        #Sustainable starting storage = ending storage level - maximum storage level + maximum storage limit.
        starting_storage_level = profile[-2] - np.amax(profile[0:-1]) + max_storage_limit
    #Increasing scenario.        
    else:
        #Sustainable starting storage level = ending storage level - minimum storage level + minimum storage limit.
        starting_storage_level = profile[-2] - np.amin(profile[0:-1]) + min_storage_limit
    
    return [difference, storage_size, max_charge, max_discharge, max_storage_constraint, min_storage_constraint, starting_storage_level]


#Create electrical network.
net = create_network()

###Obtain data###
#Run powerflow to get the admittance matrix.
pp.rundcpp(net)

#Obatin base MVA for per unit system.
base_MVA = net['_ppc']['baseMVA']
#Obatin susceptance matrix from pandapower.
B_bus = net['_ppc']['internal']['Bbus'].toarray()
#Get susceptance value from real part of the susceptance matrix.
B_bus = pd.DataFrame(np.real(B_bus))
#Obatin from bus susceptance matrix from pandapower.
Bf = net['_ppc']['internal']['Bf'].toarray()
#Get susceptance matrix from the admittance matrix
Bf = pd.DataFrame(np.real(Bf))

#Obatin bus power injection due to phase shift from pandapower.
Pbus_inj = net['_ppc']['internal']['Pbusinj']
#Get susceptance matrix from the admittance matrix
Pbus_inj = pd.DataFrame(np.real(Pbus_inj))

#Obatin from bus susceptance matrix from pandapower.
Pf_inj = net['_ppc']['internal']['Pfinj']
#Get susceptance matrix from the admittance matrix
Pf_inj = pd.DataFrame(np.real(Pf_inj))

#Obtain generator cost.
gen_cost = net.poly_cost.loc[net.poly_cost['et']=='gen']
gen_cost.set_index('element', inplace=True)
gen_cost.sort_index(inplace=True)
#Obatin static generator cost.
sgen_cost = net.poly_cost.loc[net.poly_cost['et']=='sgen']
sgen_cost.set_index('element', inplace=True)
sgen_cost.sort_index(inplace=True)
#Obatin external grid cost.
ext_grid_cost = net.poly_cost.loc[net.poly_cost['et']=='ext_grid']
ext_grid_cost.set_index('element', inplace=True)
ext_grid_cost.sort_index(inplace=True)
#Obatin external grid cost.
storage_cost = net.poly_cost.loc[net.poly_cost['et']=='storage']
storage_cost.set_index('element', inplace=True)
storage_cost.sort_index(inplace=True)

#Range of time periods between time steps.
time_period_index = pd.RangeIndex(0,len(LOAD))
#Range of time steps.
time_step_index = pd.RangeIndex(0,len(LOAD)+1)

#Number of storage sites on the grid.
number_of_storage = len(net.storage)
#Range of storage number for iteration.
storage_index = net.storage.index

upper_storage_constraint = {}
lower_storage_constraint = {}
max_charge_constraint = {}
max_discharge_constraint = {}
starting_storage_level_constraint = {}

#Create dataframe for storage specification.
storage_specification = pd.DataFrame(index = net.storage.index, 
                                     columns = ['difference',
                                                'storage_size',
                                                'max_charge',
                                                'max_discharge',
                                                'max_storage_constraint',
                                                'min_storage_constraint',
                                                'starting_storage_level'])
#Initialize the storage specification.
storage_specification['max_charge'] = float('inf')
storage_specification['max_discharge'] = float('-inf')
storage_specification['max_storage_constraint'] = float('inf')
storage_specification['min_storage_constraint'] = float('-inf')
storage_specification['starting_storage_level'] = 0


#Create Gurobi model.
model = gp.Model("DC_OPF")
model.Params.LogToConsole = 0
###Setup variable###
#Add external grid power variable.
ext_grid_power = model.addVars(net.ext_grid.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add generator power variable.
gen_power = model.addVars(net.gen.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add static generator power variable.
sgen_power = model.addVars(net.sgen.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add storage power variable.
storage_power = model.addVars(net.storage.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add bus voltage angle variable. 
bus_voltage_angle = model.addVars(net.bus.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)

#Add bus power injection holding variable (not actually a variable).
bus_injection = model.addVars(net.bus.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add line power injection holding variable (not actually a variable).
line_injection = model.addVars(net.line.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add transformer power injection holding variable (not actually a variable).
transformer_injection = model.addVars(net.trafo.index,time_period_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)

#Add storage profile variable.
storage_profile = model.addVars(net.storage.index, time_step_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add the binary decision variable for using charge or discharge efficiency.
b = model.addVars(net.storage.index, time_step_index, vtype=GRB.BINARY)

###Setup objective###
#Objective.
model.setObjective(gp.quicksum(
    gp.quicksum(gen_cost.loc[gen_index,'cp0_eur'] + gen_power[gen_index,time_period]*gen_cost.loc[gen_index,'cp1_eur_per_mw'] + gen_power[gen_index,time_period]*gen_power[gen_index,time_period]*gen_cost.loc[gen_index,'cp2_eur_per_mw2'] for gen_index in net.gen.index)+
    gp.quicksum(sgen_cost.loc[sgen_index,'cp0_eur'] + sgen_power[sgen_index,time_period]*sgen_cost.loc[sgen_index,'cp1_eur_per_mw'] + sgen_power[sgen_index,time_period]*sgen_power[sgen_index,time_period]*sgen_cost.loc[sgen_index,'cp2_eur_per_mw2'] for sgen_index in net.sgen[net.sgen['type']==''].index) +
    gp.quicksum(ext_grid_cost.loc[ext_grid_index,'cp0_eur'] + ext_grid_power[ext_grid_index,time_period]*ext_grid_cost.loc[ext_grid_index,'cp1_eur_per_mw'] + ext_grid_power[ext_grid_index,time_period]*ext_grid_power[ext_grid_index,time_period]*ext_grid_cost.loc[ext_grid_index,'cp2_eur_per_mw2'] for ext_grid_index in net.ext_grid.index)
    for time_period in time_period_index), GRB.MINIMIZE)

for time_period in time_period_index:
    #Wind generation data for the time period.
    net.sgen['max_p_mw'].iloc[-6:] = net.sgen['min_p_mw'].iloc[-6:] = WIND.iloc[time_period,:]
    #Demand data for the time period.
    net.load['p_mw'] = LOAD.iloc[time_period,:]
    ###Setup constraint###
    #Voltage angle of external grid reference bus.
    for ext_grid_index in net.ext_grid.index:
        model.addConstr(bus_voltage_angle[net.ext_grid.loc[ext_grid_index,'bus'], time_period] >= net.ext_grid.loc[ext_grid_index,'va_degree'] * np.pi / 180)
        model.addConstr(bus_voltage_angle[net.ext_grid.loc[ext_grid_index,'bus'], time_period] <= net.ext_grid.loc[ext_grid_index,'va_degree'] * np.pi / 180)
    #External grid power constraint.     
    for ext_grid_index in net.ext_grid.index:
        model.addConstr(ext_grid_power[ext_grid_index, time_period] >= net.ext_grid.loc[ext_grid_index,'min_p_mw'])
        model.addConstr(ext_grid_power[ext_grid_index, time_period] <= net.ext_grid.loc[ext_grid_index,'max_p_mw'])
    #Generator power constraint.
    for gen_index in net.gen.index:
        model.addConstr(gen_power[gen_index, time_period] >= net.gen.loc[gen_index,'min_p_mw'])
        model.addConstr(gen_power[gen_index, time_period] <= net.gen.loc[gen_index,'max_p_mw'])
    #Static generator power constraint.
    for sgen_index in net.sgen.index:
        model.addConstr(sgen_power[sgen_index, time_period] >= net.sgen.loc[sgen_index,'min_p_mw'])
        model.addConstr(sgen_power[sgen_index, time_period] <= net.sgen.loc[sgen_index,'max_p_mw'])
    
    #Bus injection.
    for from_bus in net['bus'].index:
        model.addConstr(bus_injection[from_bus, time_period] == gp.quicksum(B_bus.iloc[from_bus,to_bus] * bus_voltage_angle[to_bus, time_period] for to_bus in net['bus'].index))
    #Nodal power balance.
    for from_bus in net['bus'].index:
        model.addConstr((bus_injection[from_bus, time_period] + Pbus_inj.iloc[from_bus,0]) * base_MVA + gp.quicksum(net.load.loc[load_index,'p_mw'] for load_index in net.load.index[net.load['bus']==from_bus])
                                                ==  gp.quicksum(ext_grid_power[ext_grid_index, time_period] for ext_grid_index in net.ext_grid.index[net.ext_grid['bus']==from_bus]) + 
                                                    gp.quicksum(gen_power[gen_index, time_period] for gen_index in net.gen.index[net.gen['bus']==from_bus]) + 
                                                    gp.quicksum(sgen_power[sgen_index, time_period] for sgen_index in net.sgen.index[net.sgen['bus']==from_bus]) +
                                                    gp.quicksum(storage_power[storage_index, time_period] for storage_index in net.storage.index[net.storage['bus']==from_bus]))
    
    #Line power injection.
    for line in net['line'].index:
        model.addConstr(line_injection[line, time_period] == gp.quicksum(Bf.iloc[line,bus] * bus_voltage_angle[bus, time_period] for bus in net['bus'].index))
    #Line power limit.
    for line in net['line'].index:
        #note that line may not have Pf_inj due to no angle shift in line
        model.addConstr(line_injection[line, time_period] * base_MVA <= (net['line'].loc[line,'max_i_ka'] * net['bus'].loc[net['line'].loc[line,'from_bus'],'vn_kv'] * np.sqrt(3.) * net['line'].loc[line,'df']))
        model.addConstr(line_injection[line, time_period] * base_MVA >= -(net['line'].loc[line,'max_i_ka'] * net['bus'].loc[net['line'].loc[line,'from_bus'],'vn_kv'] * np.sqrt(3.) * net['line'].loc[line,'df']))
    
    #Transformer power injection.
    trafo_start_index = len(net['line'])
    for trafo in net['trafo'].index:
        model.addConstr(transformer_injection[trafo, time_period] == gp.quicksum(Bf.iloc[trafo_start_index+trafo,bus] * bus_voltage_angle[bus, time_period] for bus in net['bus'].index))
    #Transformer line power limit.
    for trafo in net['trafo'].index:
        #df is derating factor.
        model.addConstr((transformer_injection[trafo, time_period] + Pf_inj.loc[trafo_start_index+trafo]) * base_MVA  <=  (net['trafo'].loc[trafo,'sn_mva'] * net['trafo'].loc[trafo,'df']))
        model.addConstr((transformer_injection[trafo, time_period] + Pf_inj.loc[trafo_start_index+trafo]) * base_MVA  >= -(net['trafo'].loc[trafo, 'sn_mva'] * net['trafo'].loc[trafo,'df']))

for storage_index in net.storage.index:
    
    upper_storage_constraint[storage_index] = {}
    lower_storage_constraint[storage_index] = {}
    max_charge_constraint[storage_index] = {}
    max_discharge_constraint[storage_index] = {}
    
    for time_step in time_step_index:
        #Minimum storage level <= Storage level <= maximum storage level.
        upper_storage_constraint[storage_index][time_step] = model.addConstr(storage_profile[storage_index, time_step] <= storage_specification.loc[storage_index,'max_storage_constraint'])
        lower_storage_constraint[storage_index][time_step] = model.addConstr(storage_profile[storage_index, time_step] >= storage_specification.loc[storage_index,'min_storage_constraint'])
    
    for time_step in time_period_index:
        #-(maximum discharge limit) <= storage power <= maximum charge limit.
        max_charge_constraint[storage_index][time_step] = model.addConstr(storage_power[storage_index, time_step]  <= storage_specification.loc[storage_index,'max_charge'])
        max_discharge_constraint[storage_index][time_step] = model.addConstr(storage_power[storage_index, time_step]  >= storage_specification.loc[storage_index,'max_discharge'])

        #Conditional method for linear programming, where b is 1 for charging, and 0 for discharging.
        model.addConstr(storage_power[storage_index, time_step] >= -(GRB.INFINITY) * (1- b[storage_index, time_step]))
        model.addConstr(storage_power[storage_index, time_step] <= GRB.INFINITY * b[storage_index, time_step])
        
        #Add indicator constraints for Storage level at a future time = Storage level at current time * (1-leakage) + storage power * storage efficiency.
        model.addConstr((b[storage_index, time_step] == 1) >> (storage_profile[storage_index, time_step + 1] == storage_profile[storage_index, time_step] * (1 - LEAKAGE_RATE) + storage_power[storage_index, time_step] * CHARGE_EFFICIENCY))
        model.addConstr((b[storage_index, time_step] == 0) >> (storage_profile[storage_index, time_step + 1] == storage_profile[storage_index, time_step] * (1 - LEAKAGE_RATE) + storage_power[storage_index, time_step] / DISCHARGE_EFFICIENCY))

    #Starting storage level.
    starting_storage_level_constraint[storage_index] = model.addConstr(storage_profile[storage_index, time_step_index[0]] == storage_specification.loc[storage_index,'starting_storage_level'])


start_time = time.time()

storage_cost = 0.00001
model.params.MIPGap = 0.01
#Optimize the gurobi model.
model.optimize()
model.write('solution.sol')

storage_profile_result = np.zeros((len(net.storage.index),len(time_step_index)))
for i in net.storage.index:
    for j in time_step_index:
        storage_profile_result[i,j] = storage_profile[i, j].X

cost = model.ObjVal

iteration_count = 0

#Iterate until difference condition is met.
while True:
    storage_profile_result = np.c_[storage_profile_result, storage_profile_result[:, -1] + storage_profile_result[:, 1] - storage_profile_result[:, 0]]
    
    #Calculate storage parameters.
    for storage_index in net.storage.index:
        #Calculate the difference between end and start storage levels.
        storage_specification.loc[storage_index, :] = storage_parameter(storage_profile_result[storage_index,:])
    
    print(iteration_count, ' ', 
          cost + storage_cost * storage_specification['storage_size'].sum(), ' ',
          int(time.time()-start_time), 's')
    
    #When difference is smaller than the significant figure limit, stop the iteration.
    if max(np.abs(storage_specification.loc[:,'difference'])) < SIGNIFICANT_FIGURES: break
    
    #Run optimal power flow with demand and generation conditions.
    cost, storage_profile_result = run_optimal_power_flow(
        starting_storage_level = storage_specification['starting_storage_level'],
        max_charge = storage_specification['max_charge'],
        max_discharge = storage_specification['max_discharge'],
        max_storage_constraint = storage_specification['max_storage_constraint'], 
        min_storage_constraint = storage_specification['min_storage_constraint'])
    
    model.write('solution.sol')

    iteration_count += 1
