# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:26:19 2023

@author: orie4058
"""

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import pandapower as pp
import pandas as pd

#Import data from excel and put them into constant variables.
DATA = pd.read_excel("Storage.xlsx")
ENERGY_INPUT = pd.read_excel(r"Wind.xlsx", index_col=0)
ENERGY_OUTPUT = pd.read_excel(r"Load.xlsx", index_col=0)
CHARGE_EFFICIENCY = DATA["Charge Efficiency"][0]
DISCHARGE_EFFICIENCY = DATA["Discharge Efficiency"][0]
LEAKAGE_RATE = DATA["Leakage Rate"][0]
MAX_DOD = DATA["Maximum DoD"][0]
MIN_DOD = DATA["Minimum DoD"][0]
CHARGE_C_RATING = DATA["Charge C Rating"][0]
DISCHARGE_C_RATING = DATA["Discharge C Rating"][0]
SIGNIFICANT_FIGURES = DATA["Significant Figures"][0]


def create_network():
    
    #Import 24 bus reliability test system.
    net = pp.networks.case24_ieee_rts()
    
    #Create potential storage at each bus.
    for new_bus in net.bus.index:
        pp.create_storage(net=net, bus=new_bus, p_mw = 0, q_mvar = 0, max_e_mwh = 0, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=0, min_q_mvar=0, controllable=True)
    
    #Wind farms at bus 3, 5, 7, 16, 21 and 23.
    pp.create_sgen(net, 2, type='', p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 4, type='', p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 6, type='', p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 15, type='',p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 20, type='',p_mw=0., q_mvar=0, controllable=False)
    pp.create_sgen(net, 22, type='',p_mw=0., q_mvar=0, controllable=False)
    
    #Set external grid for energy export from microgrid.
    net.ext_grid['min_p_mw']=-197
    #Set external grid objective function.
    net.poly_cost.loc[3,'cp0_eur']=0
    net.poly_cost.loc[3,'cp1_eur_per_mw']=0
    net.poly_cost.loc[3,'cq2_eur_per_mvar2'] = 1

    return net


def run_optimal_power_flow (generation, demand, max_active_charge, max_active_discharge):

    #Load demand data at the time step.
    net.load['p_mw'] = demand.values
    #Wind generation data at the time step.
    net.sgen.loc[22:, 'p_mw'] = generation.values

    #Storage charging is positive.
    net.storage['max_p_mw'] = max_active_charge
    #Storage discharging is negative
    net.storage['min_p_mw'] = max_active_discharge
    
    pp.rundcopp(net)
   
    return 
            

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


def storage_parameter (storage_size):
    
    #Storage capacity is storage size divided by the depth of discharge.
    storage_capacity = storage_size / (MAX_DOD - MIN_DOD)
    
    #Maximum storage level constraint.
    max_storage_constraint = storage_capacity * (1 - MIN_DOD)
    #Minimum storage level constraint.
    min_storage_constraint = storage_capacity * (1 - MAX_DOD)

    #Maximum charge rate is related to storage energy capacity via the charge C-rating.
    max_charge = storage_capacity * CHARGE_C_RATING
    #Maximum discharge rate is related to storage energy capacity via the discharge C-rating.
    max_discharge = -1 * storage_capacity * DISCHARGE_C_RATING

    return [storage_size, max_charge, max_discharge, max_storage_constraint, min_storage_constraint]


def cost_function (storage_size):
    
    #Create numpy array for power to and from grid.
    overall_cost = np.zeros((number_of_time_step,1))
    #Create numpy array for storage level.
    storage_profile = np.zeros((number_of_storage, number_of_time_step+1))
    #Create numpy array for amount of untilized storage
    unutilized_storage = np.zeros((number_of_storage,1))
    #Create numpy array for max charge at each time step.
    max_charge = np.zeros((number_of_storage,1))
    #Create numpy array for max discharge at each time step.
    max_discharge = np.zeros((number_of_storage,1))
    #Create dataframe for storage specification.
    storage_specification = pd.DataFrame(index = pd.RangeIndex(0,number_of_storage), 
                                         columns = ['storage_size',
                                                    'max_charge',
                                                    'max_discharge',
                                                    'max_storage_constraint',
                                                    'min_storage_constraint'])
    
    
    #Calculate storage parameters.
    for storage_number in storage_number_range:
        #Calculate the difference between end and start storage levels.
        storage_specification.loc[storage_number, :] = storage_parameter(storage_size[storage_number])
    
    
    while True:
        #Iterate to construct the new storage profile.
        for time_step in time_step_range:
            print(time_step)
            #Calculate max charge and discharge available for this timestep.
            power_leakage = storage_profile[:, time_step]*LEAKAGE_RATE
            power_leakage[np.where(power_leakage[:]<0)]=0
            max_charge = np.minimum((storage_specification.loc[:, 'max_storage_constraint'] - (storage_profile[:, time_step] - power_leakage[:])) / CHARGE_EFFICIENCY, storage_specification.loc[:, 'max_charge'])
            max_discharge = np.maximum((storage_specification.loc[:, 'min_storage_constraint'] - (storage_profile[:, time_step]  - power_leakage[:])) * DISCHARGE_EFFICIENCY, storage_specification.loc[:, 'max_discharge'])
            
            #Run optimal power flow
            run_optimal_power_flow(generation = energy_input.iloc[time_step],
                                    demand = energy_output.iloc[time_step],
                                    max_active_charge = max_charge, 
                                    max_active_discharge = max_discharge)
            
            #Obtain overall cost from the optimal power flow.
            overall_cost[time_step,:] = net.res_cost
            
            #Construct the storage profile.
            for storage_number in storage_number_range:
                #Calculate storage level at the next time step.
                storage_profile[storage_number, time_step + 1] = new_storage_level(storage_level = storage_profile[storage_number, time_step], 
                                                                                   storage_power = net.res_storage.loc[storage_number, "p_mw"],
                                                                                   leakage_rate = LEAKAGE_RATE, 
                                                                                   min_storage_constraint = storage_specification.loc[storage_number, 'min_storage_constraint'],
                                                                                   max_storage_constraint = storage_specification.loc[storage_number, 'max_storage_constraint'])
        
        #When storage profile become sustainable, stop the iteration.
        if (storage_profile[:,0]==storage_profile[:,-1]).all(): 
            break
        else:
            storage_profile[:,0]=storage_profile[:,-1]
            
        unutilized_storage = (storage_specification.loc[:, 'max_storage_constraint'] - np.amax(storage_profile, axis=1)) +  (np.amin(storage_profile, axis=1) - storage_specification.loc[:, 'min_storage_constraint'])
        
    return overall_cost.sum()+storage_size.sum()+unutilized_storage.sum()



class geneticalgorithm():
    
    '''  Genetic Algorithm (Elitist version) for Python
    
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    
    
    
    Implementation and output:
        
        methods:
                run(): implements the genetic algorithm
                
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }
            
                report: a list including the record of the progress of the
                algorithm over iterations
    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters={'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},\
                     convergence_curve=True,\
                         progress_bar=True):


        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function. 
        (For maximization multiply function by a negative sign: the absolute 
        value of the output would be the actual objective function)
        
        @param dimension <integer> - the number of decision variables
        
        @param variable_type <string> - 'bool' if all variables are Boolean; 
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)
        
        @param variable_boundaries <numpy array/None> - Default None; leave it 
        None if variable_type is 'bool'; otherwise provide an array of tuples 
        of length two as boundaries for each variable; 
        the length of the array must be equal dimension. For example, 
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first 
        and upper boundary 200 for second variable where dimension is 2.
        
        @param variable_type_mixed <numpy array/None> - Default None; leave it 
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first 
        variable is integer but the second one is real the input is: 
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1] 
        in variable_boundaries. Also if variable_type_mixed is applied, 
        variable_boundaries has to be defined.
        
        @param function_timeout <float> - if the given function does not provide 
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function. 
        
        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int> 
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or 
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of 
            successive iterations without improvement. If None it is ineffective
        
        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.
        
        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm
  
        '''
        self.__name__=geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f=function
        #############################################################
        #dimension
        
        self.dim=int(dimension)
        
        #############################################################
        # input variable type
        
        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:
            
            if variable_type=='real': 
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)            

 
        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."       

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                

            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries 

            
        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)
 
        ############################################################# 
        #Timeout
        self.funtimeout=float(function_timeout)
        ############################################################# 
        #convergence_curve
        if convergence_curve==True:
            self.convergence_curve=True
        else:
            self.convergence_curve=False
        ############################################################# 
        #progress_bar
        if progress_bar==True:
            self.progress_bar=True
        else:
            self.progress_bar=False
        ############################################################# 
        ############################################################# 
        # input algorithm's parameters
        
        self.param=algorithm_parameters
        
        self.pop_s=int(self.param['population_size'])
        
        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 
        
        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut=self.param['mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"
        
        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        
        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)
            
        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
        
        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])
        
        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string" 
        
        
        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(self.param['max_iteration_without_improv'])

        
        ############################################################# 
    def run(self):
        
        initial_time = time.time() #Obtain the initial time.
        ############################################################# 
        # Initial Population
        
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        
        
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)
        var=np.zeros(self.dim)       
        time_record = np.zeros(self.iterate+1) #add list that records time.
        
        for p in range(0,self.pop_s):
         
            for i in self.integers[0]:
                var[i]=np.random.randint(self.var_bound[i][0],\
                        self.var_bound[i][1]+1)  
                solo[i]=var[i].copy()
            for i in self.reals[0]:
                var[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
                solo[i]=var[i].copy()


            obj=self.sim(var)            
            solo[self.dim]=obj
            pop[p]=solo.copy()

        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.test_obj=obj
        self.best_variable=var.copy()
        self.best_function=obj
        ##############################################################   
     
        t=1
        counter=0
        while t<=self.iterate:
            
            if self.progress_bar==True:
                self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

                
            
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report

            self.report.append(pop[0,self.dim])
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=pop[:,self.dim].copy()
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,self.par_s):
                pop[k]=par[k].copy()
                
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)               
                solo[: self.dim]=ch1.copy()                
                obj=self.sim(ch1)
                solo[self.dim]=obj
                pop[k]=solo.copy()                
                solo[: self.dim]=ch2.copy()                
                obj=self.sim(ch2)               
                solo[self.dim]=obj
                pop[k+1]=solo.copy()
        #############################################################       
            time_record[t]=time.time()-initial_time #Calculate how much time has passed for this iteration.
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    if self.progress_bar==True:
                        self.progress(t,self.iterate,status="GA is running...")
                    time.sleep(2)
                    t+=1
                    self.stop_mniwi=True

        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0,self.dim])
        
        
 
        
        self.output_dict={'variable': self.best_variable, 'function':\
                          self.best_function}
        if self.progress_bar==True:
            show=' '*100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush() 
        re=np.array(self.report)
        if self.convergence_curve==True:
            #plt.plot(re)
            plt.plot(time_record,re) #Plot time instead of iteration on the x-axis.
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()
        
            time_objective = pd.DataFrame({'time':time_record,'objective':re})
            time_objective.to_excel('Time Objective Result.xlsx')
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')
##############################################################################         
##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()
        

        if c_type=='one_point':
            ran=np.random.randint(0,self.dim)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
  
        if c_type=='two_point':
                
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if c_type=='uniform':
                
            for i in range(0, self.dim):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
###############################################################################  
    
    def mut(self,x):
        
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        

        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   

               x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
            
        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
###############################################################################    
    def sim(self,X):
        self.temp=X.copy()
        obj=None
        try:
            obj=func_timeout(self.funtimeout,self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        return obj

###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()     
###############################################################################            
###############################################################################


#Create electrical network.
net = create_network()

#Copy the first value to the last, for critical point comparison of the ending storage level, double [[]] keep the returned value as series or dataframe for concat.
energy_input = pd.DataFrame(ENERGY_INPUT)
energy_output = pd.DataFrame(ENERGY_OUTPUT)

#Number of time steps.
number_of_time_step = len(energy_input)
#Range of time steps for iteration.
time_step_range = range(0, number_of_time_step)
#Number of storage sites on the grid.
number_of_storage = len(net.storage)
#Range of storage number for iteration.
storage_number_range = range(0, number_of_storage)


#Variable bound for each storage.
varbound=np.array([[0,300]]*number_of_storage)

#Parameter of the genetic algorithm.
algorithm_param = {'max_num_iteration': 3,
                   'population_size':3,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

#Genetic algorithm model.
model=geneticalgorithm(function=cost_function,
         dimension=number_of_storage,
         variable_type='real',
         variable_boundaries=varbound, 
         algorithm_parameters=algorithm_param, 
         function_timeout = 10000)

#Run genetic algorithm.
model.run()
