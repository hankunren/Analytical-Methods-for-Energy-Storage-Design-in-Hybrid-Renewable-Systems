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


def create_network():

    net = pp.networks.case24_ieee_rts()

    for new_bus in net.bus.index:
        new_storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 0, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)

    #Wind farms at bus 3, 5, 7, 16, 21 and 23
    pp.create_sgen(net, 2, type='wind', p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 4, type='wind', p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 6, type='wind', p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 15, type='wind',p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 20, type='wind',p_mw=0., q_mvar=0, controllable=True)
    pp.create_sgen(net, 22, type='wind',p_mw=0., q_mvar=0, controllable=True)

    net.ext_grid['min_p_mw']=-197
    net.poly_cost.loc[3,'cp0_eur']=0
    net.poly_cost.loc[3,'cp1_eur_per_mw']=0
    net.poly_cost.loc[3,'cp2_eur_per_mw2'] = 1

    return net


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

#Create Gurobi model.
model = gp.Model("DC_OPF")

time_period_index = pd.RangeIndex(0,len(LOAD))
time_step_index = pd.RangeIndex(0,len(LOAD)+1)
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

#Add storage size variable.
storage_size = model.addVars(net.storage.index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add total storage capacity variable.
storage_capacity = model.addVars(net.storage.index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add maximum storage energy limit variable.
max_storage_limit = model.addVars(net.storage.index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add minimum storage energy limit variable.
min_storage_limit =  model.addVars(net.storage.index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add maxmimum charge limit variable.
max_charge = model.addVars(net.storage.index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add maximum discharge limit variable.
max_discharge = model.addVars(net.storage.index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add storage profile variable.
storage_profile = model.addVars(net.storage.index, time_step_index, lb=-GRB.INFINITY, ub=GRB.INFINITY)
#Add the binary decision variable for using charge or discharge efficiency.
b = model.addVars(net.storage.index, time_step_index, vtype=GRB.BINARY)

###Setup objective###
#Objective.
model.setObjective(
    gp.quicksum(
    gp.quicksum(gen_cost.loc[gen_index,'cp0_eur'] + gen_power[gen_index,time_period]*gen_cost.loc[gen_index,'cp1_eur_per_mw'] + gen_power[gen_index,time_period]*gen_power[gen_index,time_period]*gen_cost.loc[gen_index,'cp2_eur_per_mw2'] for gen_index in net.gen.index)+
    gp.quicksum(sgen_cost.loc[sgen_index,'cp0_eur'] + sgen_power[sgen_index,time_period]*sgen_cost.loc[sgen_index,'cp1_eur_per_mw'] + sgen_power[sgen_index,time_period]*sgen_power[sgen_index,time_period]*sgen_cost.loc[sgen_index,'cp2_eur_per_mw2'] for sgen_index in net.sgen[net.sgen['type']==''].index) +
    gp.quicksum(ext_grid_cost.loc[ext_grid_index,'cp0_eur'] + ext_grid_power[ext_grid_index,time_period]*ext_grid_cost.loc[ext_grid_index,'cp1_eur_per_mw'] + ext_grid_power[ext_grid_index,time_period]*ext_grid_power[ext_grid_index,time_period]*ext_grid_cost.loc[ext_grid_index,'cp2_eur_per_mw2'] for ext_grid_index in net.ext_grid.index)
    for time_period in time_period_index) + 
    gp.quicksum(1*storage_size[storage_index] for storage_index in net.storage.index), 
    GRB.MINIMIZE)

for time_period in time_period_index:
    net.sgen['max_p_mw'].iloc[-6:] = net.sgen['min_p_mw'].iloc[-6:] = WIND.iloc[time_period,:]#WIND.iloc[1,:]#
    net.load['p_mw'] = LOAD.iloc[time_period,:] #LOAD.iloc[18,:] #
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
    #Storage power constraint.
    for storage_index in net.storage.index:
        model.addConstr(storage_power[storage_index, time_period] >= net.storage.loc[storage_index,'min_p_mw'])
        model.addConstr(storage_power[storage_index, time_period] <= net.storage.loc[storage_index,'max_p_mw'])
    
    #Bus injection.
    for from_bus in net['bus'].index:
        model.addConstr(bus_injection[from_bus, time_period] == gp.quicksum(B_bus.iloc[from_bus,to_bus] * bus_voltage_angle[to_bus, time_period] for to_bus in net['bus'].index))
    #Nodal power balance.
    for from_bus in net['bus'].index:
        model.addConstr((bus_injection[from_bus, time_period] + Pbus_inj.iloc[from_bus,0]) * base_MVA + gp.quicksum(net.load.loc[load_index,'p_mw'] for load_index in net.load.index[net.load['bus']==from_bus])
                                                ==  gp.quicksum(ext_grid_power[ext_grid_index, time_period] for ext_grid_index in net.ext_grid.index[net.ext_grid['bus']==from_bus]) + 
                                                    gp.quicksum(gen_power[gen_index, time_period] for gen_index in net.gen.index[net.gen['bus']==from_bus]) + 
                                                    gp.quicksum(sgen_power[sgen_index, time_period] for sgen_index in net.sgen.index[net.sgen['bus']==from_bus]) +
                                                    gp.quicksum(storage_power[storage_index, time_period] for storage_index in net.storage.index[net.storage['bus']==from_bus])
                                                    )
    
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
    #Storage capacity is storage size divided by the depth of discharge.
    model.addConstr(storage_capacity[storage_index] == storage_size[storage_index] / (MAX_DOD - MIN_DOD))
    #Maximum charge rate is related to storage energy capacity via the charge C-rating.
    model.addConstr(max_charge[storage_index] == storage_capacity[storage_index] * CHARGE_C_RATING)
    #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
    model.addConstr(max_discharge[storage_index] == storage_capacity[storage_index] * DISCHARGE_C_RATING)
    #Maximum storage limit of the usable storage.
    model.addConstr(max_storage_limit[storage_index] == storage_capacity[storage_index] * (1 - MIN_DOD))
    #Minimum storage limit of the usable storage.
    model.addConstr(min_storage_limit[storage_index] == storage_capacity[storage_index] * (1 - MAX_DOD))

    for time_step in time_step_index:
        #Minimum storage level <= Storage level <= maximum storage level. ###THIS and storage proifle constraints one is 25 time steps, the other is 24 time steps
        model.addConstr(storage_profile[storage_index, time_step] <= max_storage_limit[storage_index])
        model.addConstr(storage_profile[storage_index, time_step] >= min_storage_limit[storage_index])
    
    for time_step in time_period_index:
        #-(maximum discharge limit) <= storage power <= maximum charge limit.
        model.addConstr(storage_power[storage_index, time_step]  <= max_charge[storage_index])
        model.addConstr(storage_power[storage_index, time_step]  >= -(max_discharge[storage_index]))

        # Model if x > y, then b = 1, otherwise b = 0
        model.addConstr(storage_power[storage_index, time_step] >= -(GRB.INFINITY)* (1- b[storage_index, time_step]))
        model.addConstr(storage_power[storage_index, time_step] <= GRB.INFINITY* b[storage_index, time_step])
        
        #Add indicator constraints for Storage level at a future time = Storage level at current time * (1-leakage) + storage power * storage efficiency.
        model.addConstr((b[storage_index, time_step] == 1) >> (storage_profile[storage_index, time_step + 1] == storage_profile[storage_index, time_step] * (1 - LEAKAGE_RATE) + storage_power[storage_index, time_step] * CHARGE_EFFICIENCY))
        model.addConstr((b[storage_index, time_step] == 0) >> (storage_profile[storage_index, time_step + 1] == storage_profile[storage_index, time_step] * (1 - LEAKAGE_RATE) + storage_power[storage_index, time_step] / DISCHARGE_EFFICIENCY))

    #Starting and ending storage level must equal.
    model.addConstr(storage_profile[storage_index, time_step_index[0]] == storage_profile[storage_index, time_step_index[-1]])


#Optimize the gurobi model.
model.params.MIPGap = 0.015
#model.params.LogFile= 'example.log'
#model.params.TimeLimit = 20*60
model.optimize()


result_ext_grid_power = [ext_grid_power[0,i].X for i in range(24)]
result_gen_power = [[gen_power[i, j].X for j in range(24)] for i in range(10)]
result_sgen_power =[[sgen_power[i, j].X for j in range(24)] for i in range(28)]
result_storage_profile = [[storage_profile[i, j].X for j in range(25)] for i in range(24)]
result_storage_power = [[storage_power[i, j].X for j in range(24)] for i in range(24)]
result_storage_size = [storage_size[i].X for i in range(24)]

result_storage_profile = pd.DataFrame(result_storage_profile)
result_storage_profile.to_excel('Result Storage Profile.xlsx')
result_storage_size = pd.DataFrame(result_storage_size)
result_storage_size.to_excel('Result.xlsx')
