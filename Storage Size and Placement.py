import pandapower as pp
import numpy as np
import pandas as pd

#Import data from excel and put them into constant variables.
DATA = pd.read_excel("Case Study 1.xlsx")
ENERGY_INPUT = pd.read_excel(r"Generation 4 kW Solar 2 Wind.xlsx", index_col=0)/1000
ENERGY_OUTPUT = pd.read_excel(r"Demand.xlsx", index_col=0)/1000
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
    '''
                 wind   wind
                 bus_3  bus_4
                    |   |     transformer_0 bus_1
    external_grid   bus_0 
                    |   |     transformer_1 bus_2
                bus_5  bus_6  
                wind   wind
    '''
    
    net = pp.create_empty_network()
    
    bus_0 = pp.create_bus(net, 110, min_vm_pu=0.95, max_vm_pu=1.05)
    
    external_grid = pp.create_ext_grid(net, bus_0, min_p_mw=-1000, max_p_mw=1000, min_q_mvar=-1000, max_q_mvar=1000)
    external_grid_p_cost = pp.create_pwl_cost(net, external_grid, 'ext_grid', points=[[net.ext_grid.min_p_mw.at[0], 0, -1],[0, net.ext_grid.max_p_mw.at[0], 1]], power_type="p")

    storage_0 = pp.create_storage(net=net, bus=bus_0, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)

    for wind in range(0,2):
        new_bus = pp.create_bus(net, 110, min_vm_pu=0.95, max_vm_pu=1.05)
        new_line = pp.create_line(net, new_bus, bus_0, 1, "149-AL1/24-ST1A 110.0", max_loading_percent=100)
        wind = pp.create_sgen(net, new_bus, p_mw=0, q_mvar=0, controllable=False)
        storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)

    bus_1 = pp.create_bus(net, 10, min_vm_pu=0.95, max_vm_pu=1.05)
    #bus_2 = pp.create_bus(net, 10, min_vm_pu=0.95, max_vm_pu=1.05)
    
    transformer_1 = pp.create_transformer_from_parameters(net, hv_bus=bus_0, lv_bus=bus_1, sn_mva=63, vn_hv_kv=110, vn_lv_kv=10, vkr_percent=0.32, vk_percent=18, pfe_kw=22, i0_percent=0.04, max_loading_percent=100)
    #transformer_2 = pp.create_transformer_from_parameters(net, hv_bus=bus_0, lv_bus=bus_2, sn_mva=63, vn_hv_kv=110, vn_lv_kv=10, vkr_percent=0.32, vk_percent=18, pfe_kw=22, i0_percent=0.04, max_loading_percent=100)

    storage_1 = pp.create_storage(net=net, bus=bus_1, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)
    #storage_2 = pp.create_storage(net=net, bus=bus_2, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)

    for line in range(0,2):
        old_bus = bus_1
        for line_bus in range(0,32):
            new_bus = pp.create_bus(net, 10, min_vm_pu=0.95, max_vm_pu=1.05)
            new_line = pp.create_line(net, old_bus, new_bus, 0.05, "149-AL1/24-ST1A 10.0", max_loading_percent=100)
            solar = pp.create_sgen(net, new_bus, p_mw=0., q_mvar=0, controllable=False)
            load = pp.create_load(net, new_bus, p_mw=0, q_mvar=0, controllable=False)
            storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)
            old_bus = new_bus
    
    for line in range(0,6):
        old_bus = bus_1
        for line_bus in range(0,31):
            new_bus = pp.create_bus(net, 10, min_vm_pu=0.95, max_vm_pu=1.05)
            new_line = pp.create_line(net, old_bus, new_bus, 0.05, "149-AL1/24-ST1A 10.0", max_loading_percent=100)
            solar = pp.create_sgen(net, new_bus, p_mw=0., q_mvar=0, controllable=False)
            load = pp.create_load(net, new_bus, p_mw=0, q_mvar=0, controllable=False)
            storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)
            old_bus = new_bus
    
    for line in range(0,6):
        old_bus = bus_1
        #old_bus = bus_2
        for line_bus in range(0,31):
            new_bus = pp.create_bus(net, 10, min_vm_pu=0.95, max_vm_pu=1.05)
            new_line = pp.create_line(net, old_bus, new_bus, 0.05, "149-AL1/24-ST1A 10.0", max_loading_percent=100)
            solar = pp.create_sgen(net, new_bus, p_mw=0., q_mvar=0, controllable=False)
            load = pp.create_load(net, new_bus, p_mw=0, q_mvar=0, controllable=False)
            storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)
            old_bus = new_bus
            
    for line in range(0,2):
        old_bus = bus_1
        #old_bus = bus_2
        for line_bus in range(0,32):
            new_bus = pp.create_bus(net, 10, min_vm_pu=0.95, max_vm_pu=1.05)
            new_line = pp.create_line(net, old_bus, new_bus, 0.05, "149-AL1/24-ST1A 10.0", max_loading_percent=100)
            solar = pp.create_sgen(net, new_bus, p_mw=0., q_mvar=0, controllable=False)
            load = pp.create_load(net, new_bus, p_mw=0, q_mvar=0, controllable=False)
            storage = pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 1000, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)
            old_bus = new_bus
    
    return net


def run_optimal_power_flow (generation, demand, max_active_charge, max_active_discharge):
    
    #Generation should always be positive.
    net.sgen['p_mw'] = generation.values
    #Load should always be positive.
    net.load['p_mw'] = demand.values
    
    #Storage charging is positive.
    net.storage['max_p_mw'] = max_active_charge
    #Storage discharging is negative
    net.storage['min_p_mw'] = max_active_discharge
    
    net.storage['max_q_mvar'] = 0
    net.storage['min_q_mvar'] = 0

    count = 0
    #Run optimal power flow.
    while True:
        try:
            pp.runopp(net)
            net.trafo['sn_mva']=63
            break
        except:
            net.trafo['sn_mva']+=1
            count+=1
            
    return count
            

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
    
    #Storage is charging.
    if storage_power > 0:
        #Energy charged into storage.
        energy = storage_power * CHARGE_EFFICIENCY
    #Storage is discharging.
    else:
        #Energy discharged by storage.
        energy = storage_power / DISCHARGE_EFFICIENCY
    
    #When storage level is going below the minimum storage level constraint.
    if storage_level - energy_leakage + energy < min_storage_constraint:
        #Stop discharging when storage level reaches the minimum.
        new_storage_level = min_storage_constraint
    #When storage level is going above the maximum storage level constraint.
    elif storage_level - energy_leakage + energy > max_storage_constraint:
        #Stop charging when storage level reaches the maximum.
        new_storage_level = max_storage_constraint
    else:
        #Continue to charge or discharge.
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

#Copy the first value to the last, for critical point comparison of the ending storage level, double [[]] keep the returned value as series or dataframe for concat.
energy_input = pd.concat([ ENERGY_INPUT, ENERGY_INPUT.iloc[[0]] ], axis = 0, ignore_index=True)
energy_output = pd.concat([ ENERGY_OUTPUT, ENERGY_OUTPUT.iloc[[0]] ], axis = 0, ignore_index=True)

#Number of time steps.
number_of_time_step = len(energy_input)
#Range of time steps for iteration.
time_step_range = range(0, number_of_time_step)
#Number of storage sites on the grid.
number_of_storage = len(net.storage)
#Range of storage number for iteration.
storage_number_range = range(0, number_of_storage)

#Create numpy array for power to and from grid.
grid_power = np.zeros((number_of_time_step,2))
#Create numpy array for transformer loading.
transformer_loading = np.zeros((number_of_time_step,len(net.trafo)))
#Create numpy array for power to and from storage.
storage_power = np.zeros((number_of_storage, number_of_time_step))
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

#First iteration through time steps.
for time_step in time_step_range:
    print(time_step)
    #Run optimal power flow with demand and generation conditions.
    run_optimal_power_flow(generation = energy_input.iloc[time_step],
                           demand = energy_output.iloc[time_step],
                           max_active_charge = float('inf'), 
                           max_active_discharge = float('-inf'))
    #Obtain grid power from the optimal power flow.
    grid_power[time_step,:] = net.res_ext_grid
    #Obtain transformer loading from the optimal power flow.
    transformer_loading[time_step,:] = net.res_trafo.loc[:, "loading_percent"]
    #Obtain storage power from the optimal power flow.
    storage_power[:, time_step] = net.res_storage.loc[:, "p_mw"]

    #Iteration through all storage.
    for storage_number in storage_number_range:
        #Calculate storage level at the next time step, with no limit on storage.
        storage_profile[storage_number, time_step + 1] = new_storage_level(storage_level = storage_profile[storage_number, time_step], 
                                                                           storage_power = net.res_storage.loc[storage_number, "p_mw"],
                                                                           leakage_rate = 0,
                                                                           min_storage_constraint = float('-inf'),
                                                                           max_storage_constraint = float('inf'))

'''
storage_profile = pd.read_excel(r"storage_profile_iteration_31.xlsx", index_col=0)
storage_profile = storage_profile.to_numpy()
'''
trafo_count = 31
iteration_count = 0

#Iterate until difference condition is met.
while True:
    energy_profile = pd.DataFrame(storage_profile)
    energy_profile.to_excel("storage_profile_iteration_"+str(iteration_count)+".xlsx")
    power_profile = pd.DataFrame(storage_power)
    power_profile.to_excel("storage_power_profile_iteration_"+str(iteration_count)+".xlsx")
    grid_profile = pd.DataFrame(grid_power)
    grid_profile.to_excel("grid_profile_iteration_"+str(iteration_count)+".xlsx")
    transformer_profile = pd.DataFrame(transformer_loading)
    transformer_profile.to_excel("transformer_profile_iteration_"+str(iteration_count)+".xlsx")

    #Calculate storage parameters.
    for storage_number in storage_number_range:
        #Calculate the difference between end and start storage levels.
        storage_specification.loc[storage_number, :] = storage_parameter(storage_profile[storage_number,:])
        #Start the new storage profile at the sustainable starting storage level.
        storage_profile[storage_number,0] = storage_specification.loc[storage_number, 'starting_storage_level']

    storage_specification.to_excel("storage_specification_iteration_"+str(iteration_count)+".xlsx")
    #When difference is smaller than the significant figure limit, stop the iteration.
    if max(np.abs(storage_specification.loc[:,'difference'])) < SIGNIFICANT_FIGURES: break

    #Iterate to construct the new storage profile.
    for time_step in time_step_range:
        print(time_step)
        #Calculate max charge and discharge available for this timestep.
        max_charge = np.minimum((storage_specification.loc[:, 'max_storage_constraint'] - (storage_profile[:, time_step] * (1 - LEAKAGE_RATE))) / CHARGE_EFFICIENCY, storage_specification.loc[:, 'max_charge'])
        max_discharge = np.maximum( np.minimum((storage_specification.loc[:, 'min_storage_constraint'] - (storage_profile[:, time_step] * (1 - LEAKAGE_RATE))) * DISCHARGE_EFFICIENCY, 0), storage_specification.loc[:, 'max_discharge'])
        
        #Run optimal power flow
        trafo_count += run_optimal_power_flow(generation = energy_input.iloc[time_step], 
                               demand = energy_output.iloc[time_step],
                               max_active_charge = max_charge, 
                               max_active_discharge = max_discharge)
        #Obtain grid power from the optimal power flow.
        grid_power[time_step,:] = net.res_ext_grid
        #Obtain transformer loading from the optimal power flow.
        transformer_loading[time_step,:] = net.res_trafo.loc[:, "loading_percent"]
        #Obtain storage power from the optimal power flow.
        storage_power[:, time_step] = net.res_storage.loc[:, "p_mw"]
        
        #Construct the storage profile.
        for storage_number in storage_number_range:
            #Calculate storage level at the next time step.
            storage_profile[storage_number, time_step + 1] = new_storage_level(storage_level = storage_profile[storage_number, time_step], 
                                                                               storage_power = net.res_storage.loc[storage_number, "p_mw"],
                                                                               leakage_rate = LEAKAGE_RATE, 
                                                                               min_storage_constraint = storage_specification.loc[storage_number, 'min_storage_constraint'],
                                                                               max_storage_constraint = storage_specification.loc[storage_number, 'max_storage_constraint'])
    
    iteration_count +=1

storage_specification.to_excel("Result.xlsx")
