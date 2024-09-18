# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:18:14 2020

@author: hanku
"""

import numpy as np
import pandas as pd

DATA= pd.read_excel(r"Data.xlsx")
LOAD = DATA["Demand"]
SOLAR = DATA["Solar Capacity Factor"]
WIND = DATA["Wind Capacity Factor"]
ELECTRICITY_COST = DATA["Electricity Cost"]

STORAGE = pd.read_excel("Storage.xlsx")
CHARGE_EFFICIENCY = STORAGE["Charge Efficiency"][0]
DISCHARGE_EFFICIENCY = STORAGE["Discharge Efficiency"][0]
LEAKAGE_RATE = STORAGE["Leakage Rate"][0]
MAX_DOD = STORAGE["Maximum DoD"][0]
MIN_DOD = STORAGE["Minimum DoD"][0]
CHARGE_C_RATING = STORAGE["Charge C Rating"][0]
DISCHARGE_C_RATING = STORAGE["Discharge C Rating"][0]
SIGNIFICANT_FIGURES = 0.01
MULTIPLIER = 0.5


def storage_sizing_analytical (ENERGY_INPUT, ENERGY_OUTPUT):
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
        energy_change = np.append(NET_ENERGY, NET_ENERGY[0])
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
        for index, Energy in enumerate(energy_change):
            #Calculate energy leakage.
            energy_leakage = storage_profile[index] * LEAKAGE_RATE
            #If energy leakage is negative, set it to zero.
            if energy_leakage < 0: energy_leakage = 0
            #When storage level is going below the minimum storage level constraint.
            if storage_profile[index] - energy_leakage + Energy < min_storage_constraint:
                #Stop discharging when storage level reaches the minimum.
                storage_profile[index + 1] = min_storage_constraint
            #When storage level is going above the maximum storage level constraint.
            elif storage_profile[index] - energy_leakage + Energy > max_storage_constraint:
                #Stop charging when storage level reaches the maximum.
                storage_profile[index + 1] = max_storage_constraint
            else:
                #Continue to charge or discharge.
                storage_profile[index + 1] = storage_profile[index] - energy_leakage + Energy
        
        #When difference is smaller than the significant figure limit, stop the iteration.
        if np.abs(difference) < SIGNIFICANT_FIGURES: break

    return storage_size


def storage_simulation (ENERGY_INPUT, ENERGY_OUTPUT, storage_size):
    
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

    time_interval = len(ENERGY_INPUT)
    #Create storage profile.
    storage_profile = np.zeros((time_interval+1))

    
    while True:
        #Create storage deficit zero array.
        energy_deficit = np.zeros((time_interval))
        #Iterate to construct the actual storage profile with starting stoage level at 0, to find ending storage level.
        for i in range(time_interval):
            
            #Calculate energy leakage.
            energy_leakage = storage_profile[i] * LEAKAGE_RATE * (-1)
            #If energy leakage is negative, set it to zero.
            if energy_leakage > 0: energy_leakage = 0
            
            #Calculate power difference between generation and demand.
            power = ENERGY_INPUT[i]-ENERGY_OUTPUT[i]
            
            #If power difference calls for discharge from storage.
            if power < 0 :
                #Check if power difference is beyond max storage discharge.
                if power < max_discharge:
                    #Calculate energy deficit cause by the storage discharge limit.
                    energy_deficit[i] += power - max_discharge
                    #Calculate max storage internal discharge.
                    storage_power = max_discharge / DISCHARGE_EFFICIENCY
                else:
                    #Calculate storage internal discharge.
                    storage_power = power / DISCHARGE_EFFICIENCY
                
                #Check if storage level will fall below minimum storage limit.
                if storage_profile[i] + storage_power + energy_leakage < min_storage_limit:
                    #Calculate energy deficit due to emptying storage.
                    energy_deficit[i] += (storage_profile[i] + storage_power + energy_leakage - min_storage_limit) * DISCHARGE_EFFICIENCY
                    #New storage level is the minium storage limit.
                    storage_profile[i + 1] = min_storage_limit
                else:
                    #Calculate new storage level.
                    storage_profile[i + 1] = storage_profile[i] + storage_power + energy_leakage
            #If power difference calls for charge from storage. 
            else:
                #Check if power difference is greater than max storage charge.
                if power > max_charge:
                    #Calculate max storage internal charge.
                    storage_power = max_charge * CHARGE_EFFICIENCY
                else:
                    #Calculate max storage charge.
                    storage_power = power * CHARGE_EFFICIENCY
                #Check if storage level will go beyond maximum storage level.
                if storage_profile[i] + storage_power + energy_leakage > max_storage_limit:
                    #New storage level is the maximum storage limit.
                    storage_profile[i + 1] = max_storage_limit
                else:
                    #Calculate new storage level.
                    storage_profile[i + 1] = storage_profile[i] + storage_power + energy_leakage

        #Check if starting and ending storage levels are equal.
        if storage_profile[0] == storage_profile[-1]: 
            #Stop iteration
            break
        else:
            #Set starting storage level as ending storage level.
            storage_profile[0] = storage_profile[-1]
            
    return storage_profile, energy_deficit





result = pd.DataFrame(columns=["wind_size", "solar_size", "storage_size","energy_deficit","energy_deficit_cost"])
result_energy_deficit = pd.DataFrame()
result_storage_profile = pd.DataFrame()



#Iterate through wind size range.
for wind_size in range(0,1):
    #Scale wind generation according to size
    wind_generation = WIND*wind_size
    
    #Iterate through solar size range:
    for solar_size in range(0,5010,50):
        #Scale solar generation according to size.
        solar_generation = SOLAR*solar_size
        #Combine wind and solar generation
        combined_generation = wind_generation + solar_generation
        
        #Calculate max storage size.
        max_storage_size = storage_sizing_analytical(ENERGY_INPUT = combined_generation, ENERGY_OUTPUT = LOAD)
        
        #Step size cannot be 0 when using np.arange
        if max_storage_size != 0:
            storage_size_step = max_storage_size/100
        else:
            storage_size_step = 1
            max_storage_size = 0
        
        #Iterate through storage size.
        for storage_size in np.arange(0 , max_storage_size + storage_size_step, storage_size_step): #np.arange(max_storage_size, max_storage_size + 1, 1): #
            
            storage_profile, energy_deficit = storage_simulation (combined_generation, LOAD, storage_size)
            
            energy_deficit_cost = energy_deficit*ELECTRICITY_COST
            
            result=result.append({"wind_size": wind_size,
                                    "solar_size": solar_size,
                                    "storage_size": storage_size,
                                    "energy_deficit": energy_deficit.sum()*(-1),
                                    "energy_deficit_cost": energy_deficit_cost.sum()*(-1)},
                                    ignore_index=True)

            #result_energy_deficit[str(wind_size) + " " + str(solar_size) + " " + str(storage_size)]=energy_deficit
            #result_storage_profile[str(wind_size) + " " + str(solar_size) + " " + str(storage_size)]=storage_profile
            print(str(wind_size) + " " + str(solar_size) + " " + str(storage_size))
        
        
result.to_excel("result.xlsx")        
#result_energy_deficit.to_excel("result energy deficit test.xlsx")
#result_storage_profile.to_excel("result storage profile test.xlsx")





