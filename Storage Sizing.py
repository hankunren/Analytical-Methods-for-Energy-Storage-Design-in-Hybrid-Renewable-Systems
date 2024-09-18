import numpy as np
import pandas as pd

#Import data from excel and put them into constant variables.
DATA = pd.read_excel(r"Case Study 1.xlsx")
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
        
        #MULTIPLIER*=0.99999
        print([storage_size, difference])
        #When difference is smaller than the significant figure limit, stop the iteration.
        if np.abs(difference) < SIGNIFICANT_FIGURES: break

    return [storage_size, storage_profile]

result = storage_sizing_analytical(ENERGY_INPUT, ENERGY_OUTPUT, 0.9)

print(result)

