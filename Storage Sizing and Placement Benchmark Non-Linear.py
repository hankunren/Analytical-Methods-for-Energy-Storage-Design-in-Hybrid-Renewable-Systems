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
SIGNIFICANT_FIGURES = DATA["Significant Figures"][0]/10
MULTIPLIER = 0.8

def create_network():

    net = pp.networks.case24_ieee_rts()

    for new_bus in net.bus.index:
        pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 0, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)

    #Wind farms at bus 3, 5, 7, 16, 21 and 23
    pp.create_sgen(net, 2, type='', p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 4, type='', p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 6, type='', p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 15, type='',p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 20, type='',p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 22, type='',p_mw=0., q_mvar=0, controllable=False)

    net.ext_grid['min_p_mw']=-197
    net.poly_cost.loc[3,'cp0_eur']=0
    net.poly_cost.loc[3,'cp1_eur_per_mw']=0
    net.poly_cost.loc[3,'cq2_eur_per_mw2'] = 1

    return net


def run_optimal_power_flow (generation, demand, max_active_charge, max_active_discharge):

    #Load should always be positive.
    net.load['p_mw'] = demand.values
    net.sgen.loc[22:, 'p_mw'] = generation.values

    #Storage charging is positive.
    net.storage['max_p_mw'] = max_active_charge
    #Storage discharging is negative
    net.storage['min_p_mw'] = max_active_discharge
    #Rune the DC OPF.
    pp.runopp(net)
    #Return the resulting cost from the OPF.
    return net.res_cost
            

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


def new_storage_level(storage_level, storage_power, leakage_rate, min_storage_constraint,  max_storage_constraint):
    
    #Calculate energy leakage.
    energy_leakage = storage_level * leakage_rate
    #If energy leakage is negative, set it to zero.
    if energy_leakage < 0: energy_leakage = 0

    #Calculate state of charge
    SOC = (storage_level-min_storage_constraint) / (max_storage_constraint-min_storage_constraint)
    
    #Storage is charging.
    if storage_power > 0:
        #Calculate charge efficiency
        charge_efficiency = -0.0197 * SOC + 1
        #Energy charged into storage.
        energy = storage_power * charge_efficiency
    #Storage is discharging.
    else:
        #Calculate charge efficiency
        discharge_efficiency = 0.0688 * SOC + 0.9822
        if discharge_efficiency > 1 : discharge_efficiency = 1
        #Energy discharged by storage.
        energy = storage_power / discharge_efficiency
    '''
    #Storage is charging.
    if storage_power > 0:
        #Energy charged into storage.
        energy = storage_power * CHARGE_EFFICIENCY
    #Storage is discharging.
    else:
        #Energy discharged by storage.
        energy = storage_power / DISCHARGE_EFFICIENCY
    '''
    #Calculate new storage level.
    new_storage_level = storage_level - energy_leakage + energy
        
    return new_storage_level


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

#Number of time steps.
number_of_time_step = len(LOAD)
#Range of time steps for iteration.
time_step_range = range(0, number_of_time_step)
#Number of storage sites on the grid.
number_of_storage = len(net.storage)
#Range of storage number for iteration.
storage_number_range = net.storage.index


#Create numpy array for power to and from grid.
overall_cost = np.zeros((number_of_time_step,1))
line_load = np.zeros((number_of_time_step,len(net.line)))
transformer_load = np.zeros((number_of_time_step,len(net.trafo)))
#Create numpy array for storage level.
storage_profile = np.zeros((number_of_storage, number_of_time_step+1))
#Create dataframe for storage specification.
storage_specification = pd.DataFrame(index = pd.RangeIndex(0,number_of_storage), 
                                     columns = ['difference',
                                                'storage_size',
                                                'max_charge',
                                                'max_discharge',
                                                'max_storage_constraint',
                                                'min_storage_constraint',
                                                'starting_storage_level'])
max_charge = np.zeros((number_of_storage,1))
max_discharge = np.zeros((number_of_storage,1))
storage_power = np.zeros((number_of_storage,1))

#First iteration through time steps.
for time_step in time_step_range:

    #Run optimal power flow with demand and generation conditions.
    overall_cost[time_step,:] = run_optimal_power_flow(generation = WIND.iloc[time_step],
                                                       demand = LOAD.iloc[time_step],
                                                       max_active_charge = float('inf'), 
                                                       max_active_discharge = float('-inf'))
    
    #Calculate storage level at the next time step, with no limit on storage.
    storage_profile[:, time_step + 1] = storage_profile[:, time_step] + net.res_storage.loc[:, "p_mw"]

#Add change in storage level in first time period to the storage level in the last time step, to check if last time step is a critical point.
storage_profile[:, -1]=storage_profile[:, -2]+storage_profile[:, 1]-storage_profile[:, 0]

#Iterate until difference condition is met.
while True:

    #Calculate storage parameters.
    for storage_number in storage_number_range:
        #Calculate the difference between end and start storage levels.
        storage_specification.loc[storage_number, :] = storage_parameter(storage_profile[storage_number,:])
        #Start the new storage profile at the sustainable starting storage level.
        storage_profile[storage_number,0] = storage_specification.loc[storage_number, 'starting_storage_level']

    #Iterate to construct the new storage profile.
    for time_step in time_step_range:
        print(time_step)
        #Calculate max charge and discharge available for this timestep.
        leakage = storage_profile[:, time_step]*LEAKAGE_RATE
        leakage[np.where(leakage[:] < 0)] = 0
        
        #Calculate state of charge
        SOC = (storage_profile[:, time_step]-storage_specification.loc[:, 'min_storage_constraint']) / (storage_specification.loc[:, 'max_storage_constraint']-storage_specification.loc[:, 'min_storage_constraint'])
        #Calculate charge efficiency
        charge_efficiency = -0.0197 * SOC + 1
        #Calculate charge efficiency
        discharge_efficiency = 0.0688 * SOC + 0.9822
        discharge_efficiency[discharge_efficiency>1]=1
        #Calculate the maximum charge and discharge for OPF.
        max_charge = np.minimum((storage_specification.loc[:, 'max_storage_constraint'] - (storage_profile[:, time_step] - leakage)) / charge_efficiency, storage_specification.loc[:, 'max_charge'])
        max_discharge = np.maximum((storage_specification.loc[:, 'min_storage_constraint'] - (storage_profile[:, time_step]  - leakage)) * discharge_efficiency, storage_specification.loc[:, 'max_discharge'])
        
        #Run optimal power flow
        overall_cost[time_step,:] = run_optimal_power_flow(generation = WIND.iloc[time_step],
                                                           demand = LOAD.iloc[time_step],
                                                           max_active_charge = max_charge, 
                                                           max_active_discharge = max_discharge)
        
        line_load[time_step,:] = net.res_line['loading_percent']
        transformer_load[time_step,:] = net.res_trafo['loading_percent']
        storage_power = net.res_storage["p_mw"].copy()
        storage_power[storage_power>=0] *= charge_efficiency[storage_power>=0]
        storage_power[storage_power< 0] /= discharge_efficiency[storage_power<0]

        storage_profile[:, time_step + 1] = storage_profile[:, time_step] - leakage + storage_power

    #Add change in storage level in first time period to the storage level in the last time step, to check if last time step is a critical point.   
    storage_profile[:, -1]=storage_profile[:, -2]+storage_profile[:, 1]-storage_profile[:, 0]
    
    print(overall_cost.sum()+0.001*(storage_specification['max_storage_constraint']-storage_specification['min_storage_constraint']).sum())
    
    #Check if maximum difference is below the significant figures, then stop iteration.
    if max(np.abs(storage_specification.loc[:,'difference'])) < SIGNIFICANT_FIGURES: break

    
#storage_specification.to_excel("Storage Size AC Non-Linear 2.xlsx")
