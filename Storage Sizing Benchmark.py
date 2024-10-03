import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB

#from pyswarms.utils.plotters import plot_cost_history
#from pyswarms.single.global_best import GlobalBestPSO

import time

def storage_sizing_analytical (ENERGY_INPUT, ENERGY_OUTPUT, MULTIPLIER):
    #Convert energy input (generation) and energy output (demand) data into numpy array format.
    energy_input = ENERGY_INPUT.to_numpy(dtype=float)
    energy_output = ENERGY_OUTPUT.to_numpy(dtype=float)
    #Net energy is the difference between energy input (Generation) and energy output (Demand).
    NET_ENERGY = energy_input - energy_output
    #Create a copy of Net_Energy, with the first value copied to the last, for critical point comparison of the ending storage level.
    energy_change = np.append(NET_ENERGY, NET_ENERGY[0])
    #When net energy is positive, generation is bigger than demand, and extra generation is charged into storage.
    energy_change[energy_change > 0] *= CHARGE_EFFICIENCY
    #When net energy is negative, generation is less than demand, and the extra demand is met by discharging energy from storage.
    energy_change[energy_change < 0] /= DISCHARGE_EFFICIENCY
    #Add 0 to first entry of imagined storage, calculate cumulative sum to create imagined storage profile.
    storage_profile = np.append(np.zeros((1)), np.cumsum(energy_change))
    
    while True:
        #Calculate the difference between end and start storage levels.
        difference = storage_profile[-2] - storage_profile[0]
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
  
        #Storage capacity is storage size divided by the depth of discharge.
        storage_capacity = storage_size / (MAX_DOD - MIN_DOD)
        #Maximum storage limit of the usable storage.
        max_storage_limit = storage_capacity * (1 - MIN_DOD)
        #Minimum storage limit of the usable storage.
        min_storage_limit = storage_capacity * (1 - MAX_DOD)

        #Decreasing scenario.        
        if difference > 0:
            #Sustainable starting storage = ending storage level - maximum storage level + maximum storage limit.
            starting_storage_level = storage_profile[-2] - np.amax(storage_profile[0:-1]) + max_storage_limit
        #Increasing scenario.        
        else:
            #Sustainable starting storage level = ending storage level - minimum storage level + minimum storage limit.
            starting_storage_level = storage_profile[-2] - np.amin(storage_profile[0:-1]) + min_storage_limit
            
        #Start the new storage profile at the sustainable starting storage level.
        storage_profile[0] = starting_storage_level
        
        #Maximum charge rate is related to storage energy capacity via the charge C-rating.
        max_charge = storage_capacity * CHARGE_C_RATING + MULTIPLIER * np.abs(difference)
        #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
        max_discharge = storage_capacity * DISCHARGE_C_RATING * (-1) - MULTIPLIER * np.abs(difference)
        #Create a copy of net_energy (generation-demand).
        energy_change = NET_ENERGY.copy()
        #When net energy is greater than max charge rate, limit to max charge rate.
        energy_change[energy_change>max_charge] = max_charge
        #When net energy is greater than max discharge rate, limit to max discharge rate.
        energy_change[energy_change<max_discharge] = max_discharge
        #When net energy is positive, surplus generation is charged into storage.
        energy_change[energy_change>0] *= CHARGE_EFFICIENCY
        #When net energy is negative, excess demand is met via energy discharged from storage.
        energy_change[energy_change<0] /= DISCHARGE_EFFICIENCY

        #Minimum storage level constraint.
        min_storage_constraint = min_storage_limit - MULTIPLIER * np.abs(difference)
        #Maximum storage level constraint.
        max_storage_constraint = max_storage_limit + MULTIPLIER * np.abs(difference)
        #Iterate to construct the new storage profile.
        for index, energy in enumerate(energy_change):
            
            #Calculate energy leakage.
            energy_leakage = storage_profile[index] * LEAKAGE_RATE
            #If energy leakage is negative, set it to zero.
            if energy_leakage < 0: energy_leakage = 0
            
            #When storage level is going below the minimum storage level constraint.
            if storage_profile[index] - energy_leakage + energy < min_storage_constraint:
                #Stop discharging when storage level reaches the minimum.
                storage_profile[index + 1] = min_storage_constraint
            #When storage level is going above the maximum storage level constraint.
            elif storage_profile[index] - energy_leakage + energy > max_storage_constraint:
                #Stop charging when storage level reaches the maximum.
                storage_profile[index + 1] = max_storage_constraint
            else:
                #Continue to charge or discharge.
                storage_profile[index + 1] = storage_profile[index] - energy_leakage + energy
        
        MULTIPLIER*=0.99
        #print([storage_size, difference])
        #When difference is smaller than the significant figure limit, stop the iteration.
        if np.abs(difference) < SIGNIFICANT_FIGURES: break

    return [storage_size]#, storage_profile]

def storage_sizing_mathematical (ENERGY_INPUT, ENERGY_OUTPUT):
    
    #Convert energy input (generation) and energy output (demand) data into numpy array format.
    energy_input = ENERGY_INPUT.to_numpy(dtype=float)
    energy_output = ENERGY_OUTPUT.to_numpy(dtype=float)
    #Net energy is the difference between energy input (generation) and energy output (demand).
    NET_ENERGY = energy_input - energy_output
    
    model = gp.Model("Storage Sizing")
    
    number_of_time_period = ENERGY_INPUT.shape[0]
    
    number_of_time_instance = number_of_time_period + 1
    
    storage_size = model.addVar(name="storage size")
    
    storage_capacity = model.addVar(name="storage capacity")
    
    max_storage_limit = model.addVar(name="max storage limit")
    
    min_storage_limit =  model.addVar(name="min storage limit")
    
    storage_profile = model.addVars(number_of_time_instance, name="storage profile")
    
    curtail_generation = model.addVars(number_of_time_period, name="curtail generation")
    
    curtail_demand = model.addVars(number_of_time_period, name="curtail demand")
    
    max_charge = model.addVar(name="max charge")
    
    max_discharge = model.addVar(name="max discharge")
    
    #Maximum charge rate is related to storage energy capacity via the charge C-rating.
    max_charge_value = model.addConstr(max_charge == storage_size * CHARGE_C_RATING)
    #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
    max_discharge_value = model.addConstr(max_discharge == storage_size * DISCHARGE_C_RATING)
    
    # Energy charge < maximum charge
    max_charge_constraint = model.addConstrs((NET_ENERGY[i] - curtail_generation[i]) * CHARGE_EFFICIENCY <= max_charge for i in range(number_of_time_period))
    # Energy discharge (negative) > maximum (negative)
    max_discharge_constraint = model.addConstrs((NET_ENERGY[i] + curtail_demand[i]) / DISCHARGE_EFFICIENCY >= (-1) * max_discharge for i in range(number_of_time_period))
    
    # Storage level at a future time = Storage level at current time + (Generation-Demand)*Storage efficiency
    storage_profile_charge = model.addConstrs(storage_profile[i + 1] == storage_profile[i] * (1 - LEAKAGE_RATE) + (NET_ENERGY[i] - curtail_generation[i]) * CHARGE_EFFICIENCY for i in range(number_of_time_period) if NET_ENERGY[i] >= 0 )
    
    storage_profile_discharge = model.addConstrs(storage_profile[i + 1] == storage_profile[i] * (1 - LEAKAGE_RATE) + (NET_ENERGY[i] + curtail_demand[i]) / DISCHARGE_EFFICIENCY for i in range(number_of_time_period) if NET_ENERGY[i] < 0)

    #Storage capacity is storage size divided by the depth of discharge.
    storage_capacity_value = model.addConstr(storage_capacity == storage_size / (MAX_DOD - MIN_DOD))
    #Maximum storage limit of the usable storage.
    max_storage_limit_value = model.addConstr(max_storage_limit == storage_capacity * (1 - MIN_DOD))
    #Maximum storage limit of the usable storage.
    min_storage_limit_value = model.addConstr(min_storage_limit == storage_capacity * (1 - MAX_DOD))
    
    #Minimum storage level <= Storage level.
    min_storage_limit_rule = model.addConstrs(storage_profile[i] >= min_storage_limit for i in range(number_of_time_instance))
    #Storage level <= maximum storage level.
    max_storage_limit_rule = model.addConstrs(storage_profile[i] <= max_storage_limit for i in range(number_of_time_instance))
    
    #Starting and ending storage level must equal.
    equal_start_and_end_storage_level = model.addConstr(storage_profile[0] == storage_profile[number_of_time_period])
    
    #objective include storage cost and demand curtailment cost.
    objective = 1 * storage_size + 1000 * gp.quicksum(curtail_demand[i] for i in range(number_of_time_period)) 
    
    #Optimization objective and type of optimization
    model.setObjective(objective, GRB.MINIMIZE)
    model.write("Storage Sizing.lp")
    model.optimize()
    
    print(storage_size)

    return storage_size.x

def storage_sizing_heuristic (ENERGY_INPUT, ENERGY_OUTPUT):
    #Convert energy input (generation) and energy output (demand) data into numpy array format.
    energy_input = ENERGY_INPUT.to_numpy(dtype=float)
    energy_output = ENERGY_OUTPUT.to_numpy(dtype=float)
    #Net energy is the difference between energy input (generation) and energy output (demand).
    NET_ENERGY = energy_input - energy_output
    
    def energy_deficit (number_of_particle):
        
        result=np.zeros(shape = number_of_particle.shape)
        
        for particle_number, storage_size in enumerate(number_of_particle):
            
            #Create real storage profile
            storage_profile = np.zeros(shape = NET_ENERGY.shape[0] + 1)
            #Storage capacity is storage size divided by the depth of discharge.
            storage_capacity = storage_size / (MAX_DOD - MIN_DOD)
            #Maximum storage limit of the usable storage.
            max_storage_limit = storage_capacity * (1 - MIN_DOD)
            #Minimum storage limit of the usable storage.
            min_storage_limit = storage_capacity * (1 - MAX_DOD)
    
            #Maximum charge rate is related to storage energy capacity via the charge C-rating.
            max_charge = storage_capacity * CHARGE_C_RATING
            #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
            max_discharge = storage_capacity * DISCHARGE_C_RATING * (-1)
            
            #Create a copy of net_energy (generation-demand).
            curtail_discharge = NET_ENERGY.copy()
            #Curtail discharge to below the maximum discharge
            curtail_discharge[curtail_discharge < max_discharge] = max_discharge
            #Calculate the amount of curtailment
            curtail_demand = curtail_discharge - NET_ENERGY
            
            #Create a copy of net_energy (generation-demand).
            energy_change = NET_ENERGY.copy()
            #When net energy is greater than max charge rate, limit to max charge rate.
            energy_change[energy_change>max_charge] = max_charge
            #When net energy is greater than max discharge rate, limit to max discharge rate.
            energy_change[energy_change<max_discharge] = max_discharge
            #When net energy is positive, surplus generation is charged into storage.
            energy_change[energy_change>0] *= CHARGE_EFFICIENCY
            #When net energy is negative, excess demand is met via energy discharged from storage.
            energy_change[energy_change<0] /= DISCHARGE_EFFICIENCY
    
            while True:
                #Set starting storage level as ending storage level.
                storage_profile[0] = storage_profile[-1]
                #Iterate to construct the real storage profile.
                for index, energy in enumerate(energy_change):
    
                    #Calculate energy leakage.
                    energy_leakage = storage_profile[index] * LEAKAGE_RATE
                    #If energy leakage is negative, set it to zero.
                    if energy_leakage < 0: energy_leakage = 0
                    
                    #When storage level is going above the maximum storage level constraint.
                    if storage_profile[index] - energy_leakage + energy > max_storage_limit:
                        #Stop charging when storage level reaches the maximum.
                        storage_profile[index + 1] = max_storage_limit
                    #When storage level is going below the minimum storage level constraint.
                    elif storage_profile[index] - energy_leakage + energy < min_storage_limit:
                        #Stop discharging when storage level reaches the minimum.
                        storage_profile[index + 1] = min_storage_limit
                    else:
                        #Continue to charge or discharge.
                        storage_profile[index + 1] = storage_profile[index] - energy_leakage + energy
                #When difference between start and end storage levels is smaller than the significant figure limit, stop the iteration.
                if np.abs(storage_profile[-1] - storage_profile[0]) < SIGNIFICANT_FIGURES: break
            
            for index, energy in enumerate(energy_change):
                
                #Calculate energy leakage.
                energy_leakage = storage_profile[index] * LEAKAGE_RATE
                #If energy leakage is negative, set it to zero.
                if energy_leakage < 0: energy_leakage = 0
                
                #When storage level is going above the maximum storage level constraint.
                if storage_profile[index] - energy_leakage + energy > max_storage_limit:
                    #Stop charging when storage level reaches the maximum.
                    storage_profile[index + 1] = max_storage_limit
                #When storage level is going below the minimum storage level constraint.
                elif storage_profile[index] - energy_leakage + energy < min_storage_limit:
                    #Stop discharging when storage level reaches the minimum.
                    storage_profile[index + 1] = min_storage_limit
                    #When discharging.
                    if energy < 0:
                        #Curtailed demand = energy * (energy change (discharge + leakage) beyond the minimum storage limit / total energy change (discharge + leakage)) * Discharge efficiency.
                        curtail_demand[index] += (-1) * energy * (min_storage_limit - (storage_profile[index] - energy_leakage + energy)) / (energy_leakage + (-1) * energy) * DISCHARGE_EFFICIENCY
                else:
                    #Continue to charge or discharge.
                    storage_profile[index + 1] = storage_profile[index] - energy_leakage + energy
                        
            result[particle_number]=curtail_demand.sum()
            
        return result.ravel() #need to be 1D array, and .ravel() covert 2D to 1D
    
    
    # instatiate the optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds = ([0], [1000]) #bounds need to be tuple of array
    optimizer = GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds)#

    cost, result = optimizer.optimize(energy_deficit, iters=20)
    
    #plot_cost_history(cost_history=optimizer.cost_history)

    return result

def storage_sizing_enumerative (ENERGY_INPUT, ENERGY_OUTPUT):
    
    #Convert energy input (generation) and energy output (demand) data into numpy array format.
    energy_input = ENERGY_INPUT.to_numpy(dtype=float)
    energy_output = ENERGY_OUTPUT.to_numpy(dtype=float)
    #Net energy is the difference between energy input (generation) and energy output (demand).
    NET_ENERGY = energy_input - energy_output

    def energy_deficit (storage_size):
    
        #Create real storage profile
        storage_profile = np.zeros(shape = NET_ENERGY.shape[0] + 1)
        #Storage capacity is storage size divided by the depth of discharge.
        storage_capacity = storage_size / (MAX_DOD - MIN_DOD)
        #Maximum storage limit of the usable storage.
        max_storage_limit = storage_capacity * (1 - MIN_DOD)
        #Minimum storage limit of the usable storage.
        min_storage_limit = storage_capacity * (1 - MAX_DOD)

        #Maximum charge rate is related to storage energy capacity via the charge C-rating.
        max_charge = storage_capacity * CHARGE_C_RATING
        #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
        max_discharge = storage_capacity * DISCHARGE_C_RATING * (-1)
        
        #Create a copy of net_energy (generation-demand).
        curtail_discharge = NET_ENERGY.copy()
        #Curtail discharge to below the maximum discharge
        curtail_discharge[curtail_discharge < max_discharge] = max_discharge
        #Calculate the amount of curtailment
        curtail_demand = curtail_discharge - NET_ENERGY
        
        #Create a copy of net_energy (generation-demand).
        energy_change = NET_ENERGY.copy()
        #When net energy is greater than max charge rate, limit to max charge rate.
        energy_change[energy_change>max_charge] = max_charge
        #When net energy is greater than max discharge rate, limit to max discharge rate.
        energy_change[energy_change<max_discharge] = max_discharge
        #When net energy is positive, surplus generation is charged into storage.
        energy_change[energy_change>0] *= CHARGE_EFFICIENCY
        #When net energy is negative, excess demand is met via energy discharged from storage.
        energy_change[energy_change<0] /= DISCHARGE_EFFICIENCY

        while True:
            #Set starting storage level as ending storage level.
            storage_profile[0] = storage_profile[-1]
            #Iterate to construct the real storage profile.
            for index, energy in enumerate(energy_change):

                #Calculate energy leakage.
                energy_leakage = storage_profile[index] * LEAKAGE_RATE
                #If energy leakage is negative, set it to zero.
                if energy_leakage < 0: energy_leakage = 0
                
                #When storage level is going above the maximum storage level constraint.
                if storage_profile[index] - energy_leakage + energy > max_storage_limit:
                    #Stop charging when storage level reaches the maximum.
                    storage_profile[index + 1] = max_storage_limit
                #When storage level is going below the minimum storage level constraint.
                elif storage_profile[index] - energy_leakage + energy < min_storage_limit:
                    #Stop discharging when storage level reaches the minimum.
                    storage_profile[index + 1] = min_storage_limit
                else:
                    #Continue to charge or discharge.
                    storage_profile[index + 1] = storage_profile[index] - energy_leakage + energy
            #When difference between start and end storage levels is smaller than the significant figure limit, stop the iteration.
            if np.abs(storage_profile[-1] - storage_profile[0]) < SIGNIFICANT_FIGURES: break
        
        for index, energy in enumerate(energy_change):
            
            #Calculate energy leakage.
            energy_leakage = storage_profile[index] * LEAKAGE_RATE
            #If energy leakage is negative, set it to zero.
            if energy_leakage < 0: energy_leakage = 0
            
            #When storage level is going above the maximum storage level constraint.
            if storage_profile[index] - energy_leakage + energy > max_storage_limit:
                #Stop charging when storage level reaches the maximum.
                storage_profile[index + 1] = max_storage_limit
            #When storage level is going below the minimum storage level constraint.
            elif storage_profile[index] - energy_leakage + energy < min_storage_limit:
                #Stop discharging when storage level reaches the minimum.
                storage_profile[index + 1] = min_storage_limit
                #When discharging.
                if energy < 0:
                    #Curtailed demand = energy * (energy change (discharge + leakage) beyond the minimum storage limit / total energy change (discharge + leakage)) * Discharge efficiency.
                    curtail_demand[index] += (-1) * energy * (min_storage_limit - (storage_profile[index] - energy_leakage + energy)) / (energy_leakage + (-1) * energy) * DISCHARGE_EFFICIENCY
            else:
                #Continue to charge or discharge.
                storage_profile[index + 1] = storage_profile[index] - energy_leakage + energy
        
        return curtail_demand.sum() #{"Curtail Demand": curtail_demand.sum(), "Storage Profile": storage_profile, "Storage Limits": [max_storage_limit, min_storage_limit] }
    
    #Initialize result dataframe.
    result = pd.DataFrame(columns={'storage size','energy provided'})
    #Initialize storage size
    storage_size = 0
    #Obtain the reference energy deficit of the system without storage.
    reference_energy_deficit = energy_deficit(storage_size)
    #Initilize the variable that holds energy provided from previous storage size.
    previous_energy_provided = 0
    #Enumeratively iterate through storage size.
    
    while True:
        #Put energy provided into result table.
        result = result.append({'storage size':storage_size,'energy provided':previous_energy_provided},ignore_index=True)
        #Incrementally increase storage size.
        storage_size += 1
        #The decrease in energy deficit is energy provided by storage to the system
        energy_provided = reference_energy_deficit - energy_deficit(storage_size)
        #print([storage_size, energy_provided])

        #Compare with previous iteration, when the energy provided no longer increase as size increase, stop iteration.
        #if energy_provided >= previous_energy_provided: break
        if storage_size > 1000:break
        #Current result becomes the previous result for next iteration.
        previous_energy_provided = energy_provided
    
    result['energy provided'] = pd.to_numeric(result['energy provided'])
    max_index = result['energy provided'].idxmax()
    return result.iloc[max_index]


time_result = np.zeros(shape = (31,10))

for j in range(1,2,1):
    
    for i in range(0,10,1):
        
        number_of_data = j*10
        
        DATA=pd.DataFrame()
        DATA["Energy Input"]=(np.random.rand(number_of_data))*2000/number_of_data
        DATA["Energy Output"]=(np.random.rand(number_of_data))*1000/number_of_data
        
        DATA["Charge Efficiency"]=np.random.rand()
        DATA["Discharge Efficiency"]=np.random.rand()
        
        DATA["Leakage Rate"] = np.random.rand()/number_of_data #0.05/366/24#
        
        DoD = np.sort(np.random.rand(2))
        DATA["Maximum DoD"] = DoD[1]
        DATA["Minimum DoD"] = DoD[0]
        
        DATA["Charge C Rating"] =np.random.rand()
        DATA["Discharge C Rating"] =np.random.rand()
        
        DATA["Significant Figures"]=0.001
        
        #Import data from excel and put them into constant variables.
        ENERGY_INPUT = DATA["Energy Input"]
        ENERGY_OUTPUT = DATA["Energy Output"]
        CHARGE_EFFICIENCY = DATA["Charge Efficiency"][0]
        DISCHARGE_EFFICIENCY = DATA["Discharge Efficiency"][0]
        LEAKAGE_RATE = DATA["Leakage Rate"][0]
        MAX_DOD = DATA["Maximum DoD"][0]
        MIN_DOD = DATA["Minimum DoD"][0]
        CHARGE_C_RATING = DATA["Charge C Rating"][0]
        DISCHARGE_C_RATING = DATA["Discharge C Rating"][0]
        SIGNIFICANT_FIGURES = DATA["Significant Figures"][0]
        
        start_time_analytical = time.time()
        result_analytical = storage_sizing_analytical(ENERGY_INPUT, ENERGY_OUTPUT, 0.1)
        end_time_analytical = time.time()
        time_analytical = end_time_analytical - start_time_analytical
        
        time_result[j,i] = time_analytical
        
    print(j)


start_time_mathematical = time.time()
result_mathematical = storage_sizing_mathematical(ENERGY_INPUT, ENERGY_OUTPUT)
end_time_mathematical = time.time()
time_mathematical = end_time_mathematical - start_time_mathematical
'''
start_time_analytical = time.time()
result_analytical = storage_sizing_analytical(ENERGY_INPUT, ENERGY_OUTPUT, 0.5)
end_time_analytical = time.time()
time_analytical = end_time_analytical - start_time_analytical

start_time_heuristic = time.time()
result_heuristic = storage_sizing_heuristic(ENERGY_INPUT, ENERGY_OUTPUT)
end_time_heuristic = time.time()
time_heuristic = end_time_heuristic - start_time_heuristic

start_time_enumerative = time.time()
result_enumerative = storage_sizing_enumerative(ENERGY_INPUT, ENERGY_OUTPUT)
end_time_enumerative = time.time()
time_enumerative = end_time_enumerative - start_time_enumerative
'''
