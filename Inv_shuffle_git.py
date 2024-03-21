# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:06:52 2022

@author: user
"""
import datetime
start_time = datetime.datetime.now()
import pandas as pd
import pulp
from pulp import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#from datetime import datetime, timedelta

distance_df = pd.read_csv('Path to file')
distance_df['0'] = 0
distance_df.shape
distance_df.loc[len(distance_df)] = 0

xls = pd.ExcelFile('Path to file'))
Price_map_file = pd.read_excel(xls, 'FedEx Retail DCNode',skiprows=1)

data1 = pd.read_csv('Path to file'))
data_null = data1[data1.isna().any(axis=1)]
data1 = data1[~data1.isna().any(axis=1)]
data1['TOTAL_OH'] = np.where(data1['TOTAL_OH'] < 0,0,data1['TOTAL_OH'])
data1['TOTAL_OH'] = data1['TOTAL_OH'].replace({np.nan:0})
data1['STORE_NUM']=data1['STORE_NUM'].astype('str')

data1_no_avg_week_sales = data1[data1['avg_week_sales_forecast']==0]
data1_no_avg_week_sales['cat'] = np.where((data1_no_avg_week_sales['TOTAL_OH']>data1_no_avg_week_sales['ALLOC_TARGET']),'Surplus','Deficit')
data1_no_avg_week_sales['cat'] = np.where((data1_no_avg_week_sales['TOTAL_OH']==data1_no_avg_week_sales['ALLOC_TARGET']),'No transfer',data1_no_avg_week_sales['cat'])
data1_no_avg_week_sales['difference'] = np.where((data1_no_avg_week_sales['TOTAL_OH']>data1_no_avg_week_sales['ALLOC_TARGET']),data1_no_avg_week_sales['TOTAL_OH']-data1_no_avg_week_sales['ALLOC_TARGET'],data1_no_avg_week_sales['ALLOC_TARGET']-data1_no_avg_week_sales['TOTAL_OH'])

data1_no_avg_week_sales['Deficit_Store_Weightage'] = 0.01
data1_no_avg_week_sales['Surplus_Store_Weightage'] = 0.01
data1_no_avg_week_sales.loc[data1_no_avg_week_sales['RANK_SURPLUS_PRIORITY'] ==2, 'Surplus_Store_Weightage'] = 100
data1_no_avg_week_sales.loc[data1_no_avg_week_sales['RANK_SURPLUS_PRIORITY'] ==3, 'Surplus_Store_Weightage'] = 1000
data1_no_avg_week_sales.loc[data1_no_avg_week_sales['RANK_DEFICIT_PRIORITY'] ==2, 'Deficit_Store_Weightage'] = 100
data1_no_avg_week_sales.loc[data1_no_avg_week_sales['RANK_DEFICIT_PRIORITY'] ==3, 'Deficit_Store_Weightage'] = 1000
data1_no_avg_week_sales['WOS_Target']=''
data1_no_avg_week_sales['WOS_OH']=''
surplus_stores_no_avg_week_sales = data1_no_avg_week_sales[data1_no_avg_week_sales['cat']== 'Surplus'].rename(columns = {'state':'Origin_State','Region':'Origin_Region'})
deficit_stores_no_avg_week_sales = data1_no_avg_week_sales[data1_no_avg_week_sales['cat']== 'Deficit'].rename(columns = {'state':'Destination_State','Region':'Destination_Region'})

data1 = data1[data1['avg_week_sales_forecast']!=0]
#data1['avg_week_sales_forecast'] = np.ceil(data1['avg_week_sales_forecast'])
data1['WOS_OH'] = data1['TOTAL_OH']/data1['avg_week_sales_forecast']
data1['Deficit_Store_Weightage'] = 0.01
data1['Surplus_Store_Weightage'] = 0.01
data1.loc[data1['RANK_SURPLUS_PRIORITY'] ==2, 'Surplus_Store_Weightage'] = 100
data1.loc[data1['RANK_SURPLUS_PRIORITY'] ==3, 'Surplus_Store_Weightage'] = 1000
data1.loc[data1['RANK_DEFICIT_PRIORITY'] ==2, 'Deficit_Store_Weightage'] = 100
data1.loc[data1['RANK_DEFICIT_PRIORITY'] ==3, 'Deficit_Store_Weightage'] = 1000
data1['WOS_Target'] = data1['ALLOC_TARGET']/data1['avg_week_sales_forecast']
data1['cat'] = ''
data1['cat'] = np.where(((data1['WOS_OH'] < data1['WOS_Target']) | (data1['WOS_OH']<2)),'Deficit',data1['cat'])
data1['cat'] = np.where(((data1['cat'] == '')&(data1['WOS_OH']>8)),'Surplus',data1['cat']) #As per Daniela's Logic

data1['cat'] = np.where(((data1['cat'] == '')),'No transfer',data1['cat'])
surplus_stores = data1[data1['cat']== 'Surplus'].rename(columns = {'state':'Origin_State','Region':'Origin_Region'})
deficit_stores = data1[data1['cat']== 'Deficit'].rename(columns = {'state':'Destination_State','Region':'Destination_Region'})

#The below deficit difference logic might give -ve values. to overcome,additional below line is coded
deficit_stores['difference'] = (2-deficit_stores['WOS_OH'])*deficit_stores['avg_week_sales_forecast']
#wherever wos_oh is less than the allocation target get the difference of allocation target and wos_oh
deficit_stores['difference'] = np.where((deficit_stores['WOS_Target']>deficit_stores['WOS_OH']),((deficit_stores['WOS_Target']-deficit_stores['WOS_OH'])*deficit_stores['avg_week_sales_forecast']),deficit_stores['difference'])
surplus_stores['difference'] = (surplus_stores['WOS_OH']-8)*surplus_stores['avg_week_sales_forecast']
surplus_stores['difference'] = np.where((surplus_stores['WOS_Target']>8),((surplus_stores['WOS_OH']-surplus_stores['WOS_Target'])*surplus_stores['avg_week_sales_forecast']),surplus_stores['difference'])

deficit_stores = pd.concat([deficit_stores, deficit_stores_no_avg_week_sales], ignore_index=True, axis=0)
surplus_stores = pd.concat([surplus_stores, deficit_stores_no_avg_week_sales], ignore_index=True, axis=0)
deficit_stores_subset = deficit_stores[['SKU','STORE_NUM','difference','Deficit_Store_Weightage']]
surplus_stores_subset = surplus_stores[['SKU','STORE_NUM','difference','Surplus_Store_Weightage']]
surplus_stores_subset = surplus_stores_subset[surplus_stores_subset['difference']>1]
#deficit_stores_subset = deficit_stores[['SKU','STORE_NUM','difference','Deficit_Store_Weightage']][deficit_stores['SKU'].isin([22739171,37855053,37855079])]

def get_distance(origin, destination):
    return distance_df[distance_df['STORE_NUM'] == int(origin)][str(destination)].values[0]

def get_dist(origin,destination):
    x = distance_df.loc[distance_df['STORE_NUM'].isin(map(int,origin)),destination]
    x.insert(loc=0, column='Origin', value=origin)
    x = x.set_index("Origin")
    x=x.to_dict('index')
    return x

def get_cost_Ship(Weight,Zone):
    if ((Weight > 50)):
        return Price_map_file[Price_map_file['Weight (lb)'] == 50][Zone].values[0]
    else:
        return Price_map_file[Price_map_file['Weight (lb)'] == Weight][Zone].values[0]

sku = 14088355
def optimize(SKU):
    op = pd.DataFrame(columns =['Item_ID','Route','Distance','Qty','Demand'])
    for sku in SKU:
        print(sku)
        samp_deficit = deficit_stores_subset[deficit_stores_subset['SKU']==sku]
        samp_surplus = surplus_stores_subset[surplus_stores_subset['SKU']==sku] 
        
        #Dictionary of max units that can be shipped to each Deficit Store
        need = dict(samp_deficit[['STORE_NUM','difference']].values)
        
        #Dictionary of units each Surplus store will supply
        give = dict(samp_surplus[['STORE_NUM','difference']].values)
        
        #Dictionary that stores the priority ranks of each surplus store
        priority_surplus = dict(samp_surplus[['STORE_NUM','Surplus_Store_Weightage']].values)
        #Dictionary that stores the priority ranks of each deficit store
        priority_deficit = dict(samp_deficit[['STORE_NUM','Deficit_Store_Weightage']].values)

        
        if (sum(need.values())<sum(give.values())):
            need['0'] = sum(give.values()) - sum(need.values())
            samp_deficit.loc[len(samp_deficit.index)] = [sku, '0', need['0'],'0']
            priority_deficit['0'] = 0
            samp_dest = np.unique(samp_deficit['STORE_NUM']).tolist()
            samp_origin = np.unique(samp_surplus['STORE_NUM']).tolist()
            far = get_dist(samp_origin,samp_dest)
            samp_routes = [(i,j) for i in samp_origin for j in samp_dest]
            
        else:
            give['0'] = sum(need.values()) - sum(give.values())
            samp_surplus.loc[len(samp_surplus.index)] = [sku, '0', give['0'],0]
            priority_surplus['0'] = 0
            samp_dest = np.unique(samp_deficit['STORE_NUM']).tolist()
            samp_origin = np.unique(samp_surplus['STORE_NUM']).tolist()
            far = get_dist(samp_origin,samp_dest)
            samp_routes = [(i,j) for i in samp_origin for j in samp_dest]   
            
        #Set Problem variable
        Samp_Prob = LpProblem("Transportation",LpMinimize)
        
        #Decision variable
        samp_amount_vars = LpVariable.dicts(name = "ShipUnits",indices = (samp_origin,samp_dest),lowBound=0)
        
        #Objective Function
        Samp_Prob+= lpSum(samp_amount_vars[i][j]*far[i][j]*priority_surplus[i]*priority_deficit[j] for (i,j) in samp_routes)
        
        #Constraints
        for j in samp_dest:
            Samp_Prob+=lpSum(samp_amount_vars[i][j] for i in samp_origin)<=need[j]
        
        for i in samp_origin:
            Samp_Prob+=lpSum(samp_amount_vars[i][j] for j in samp_dest)==give[i]
                           
        Samp_Prob.solve(PULP_CBC_CMD(msg=1))
        print(LpStatus[Samp_Prob.status])
        
        # Check the solver being used
        solver_name = Samp_Prob.solver
        print("Solver used:", solver_name)

        
        for v in Samp_Prob.variables():
            if len(samp_surplus)>1:
                if v.varValue is not None:
                    if v.varValue>0:
                        item = sku 
                        Route = v.name
                        distance = get_distance(Route.split('_')[1], Route.split('_')[2])
                        Qty = v.varValue
                        Demand = need[v.name.split('_')[2]]
                        data = [item,Route,distance,Qty,Demand]
                        op.loc[len(op)] = data
    return op

skus = np.unique(deficit_stores['SKU'])
skus = skus[0:1]
ans_0 = optimize(skus)
ans = ans_0[ans_0['Distance']!=0]
End_time = datetime.datetime.now()
initialjob_Run_time = End_time-start_time
ans[['Units','From', 'To']] = ans['Route'].str.split('_', 3,expand= True)
ans = ans.drop(columns = ['Units','Route']).rename(columns = {'From':'Origin_Store','To':'Destination_Store'})
ans[['Origin_Store','Destination_Store']] = ans[['Origin_Store','Destination_Store']].astype(str)
ans.to_csv("C:/Users/user/Downloads/Regionlization/Store transfer project/pulp_prod_semi_output.csv")

###########################
import pandas as pd
import numpy as np
ans = pd.read_csv("C:/Users/user/Downloads/Regionlization/Store transfer project/pulp_prod_semi_output.csv")
ans[['From','To']] = ans[['From','To']].astype(str)
ans1 = ans
ans1['Qty'] = np.ceil(ans1['Qty']).astype('int64')
#ans1['Demand'] = np.ceil(ans1['Demand']).astype('int64')
ans1 = pd.merge(ans1,deficit_stores[['SKU','SALES_PROFIT_UT','STORE_NUM','DEPT_STORE_GRADE','SKU_Weight','Destination_State','Destination_Region','LIFE_CYCLE_PRIORITY']],how = 'left',left_on=['Item_ID','Destination_Store'],right_on=['SKU','STORE_NUM'])
ans1 = ans1.drop(columns=['SKU','STORE_NUM']).rename(columns = {'DEPT_STORE_GRADE':'Destination_DEPT_STORE_GRADE'}).drop_duplicates()
ans1 = pd.merge(ans1,surplus_stores[['SKU','STORE_NUM','DEPT_STORE_GRADE','Origin_State','Origin_Region']],how = 'left',left_on=['Item_ID','Origin_Store'],right_on=['SKU','STORE_NUM'])
ans1 = ans1.drop(columns=['SKU','STORE_NUM']).rename(columns = {'DEPT_STORE_GRADE':'Origin_DEPT_STORE_GRADE'}).drop_duplicates()
ans1['Destination_Region'] = ans1['Destination_Region'].str.split('-').str[0]
ans1['Origin_Region'] = ans1['Origin_Region'].str.split('-').str[0]
ans1['Tot_Wt'] = np.ceil(ans1['SKU_Weight']*ans1['Qty'])
ans1['Tot_Wt'] = np.where(ans1['Tot_Wt']==0.0,1.0,ans1['Tot_Wt']) #Assuming minimum weight to be 1 pound
ans1 = ans1[['Item_ID','SALES_PROFIT_UT','SKU_Weight', 'Distance', 'Qty', 'Demand', 'Origin_Store','Origin_DEPT_STORE_GRADE','Origin_State','Origin_Region','Destination_Store','Destination_DEPT_STORE_GRADE','Destination_State','Destination_Region','LIFE_CYCLE_PRIORITY', 'Tot_Wt']]
ans1[['Origin_Store','Destination_Store']] = ans1[['Origin_Store','Destination_Store']].astype('str')
ans1[['LIFE_CYCLE_PRIORITY']] = ans1[['LIFE_CYCLE_PRIORITY']].astype('int64')
ans1 = ans1.rename(columns = {"Qty":"Quantity_to_ship"})
ans1['Item_ID'] = ans1['Item_ID'].astype('int64')
ans2 = ans1.groupby(['Origin_Store','Destination_Store'])[['Tot_Wt','Quantity_to_ship','SALES_PROFIT_UT']].sum().reset_index()
ans2= pd.merge(ans1[['Distance','Origin_Store','Destination_Store']],ans2,how = 'right',left_on=['Origin_Store','Destination_Store'],right_on=['Origin_Store','Destination_Store']).drop_duplicates().reset_index()
ans2 = ans2.drop(columns= ['index'])
ans2[['Distance','Tot_Wt']] = ans2[['Distance','Tot_Wt']].astype('float64')
bins = [0, 150, 300, 600, 1000, 1400, 1800,999999999]
labels = [2,3,4,5,6,7,8]
ans2['Ship_Zone'] = pd.cut(ans2['Distance'], bins=bins, labels=labels).astype('int64')
ans2['Shipping_Cost'] = ans2.apply(lambda x: get_cost_Ship(x['Tot_Wt'], x['Ship_Zone']),axis =1)
ans2['shipping_cost_per_qty'] = ans2['Shipping_Cost']/ans2['Quantity_to_ship']
ans1 = pd.merge(ans1,ans2[['Origin_Store', 'Destination_Store', 'shipping_cost_per_qty']],how = 'left',left_on=['Origin_Store', 'Destination_Store'],right_on=['Origin_Store', 'Destination_Store'])
ans1['Shipping_Cost'] = ans1['shipping_cost_per_qty']*ans1['Quantity_to_ship']
ans1['Tot_Profit'] = ans1['SALES_PROFIT_UT']*ans1['Quantity_to_ship']
ans1['Actual_Profit'] = ans1['Tot_Profit'] - ans1['Shipping_Cost']
ans1 = ans1[['Item_ID', 'SALES_PROFIT_UT', 'SKU_Weight', 'Distance',
       'Quantity_to_ship', 'Demand', 'Origin_Store', 'Origin_DEPT_STORE_GRADE',
       'Origin_State', 'Origin_Region', 'Destination_Store',
       'Destination_DEPT_STORE_GRADE', 'Destination_State',
       'Destination_Region', 'LIFE_CYCLE_PRIORITY', 'Tot_Wt',
       'shipping_cost_per_qty', 'Shipping_Cost', 'Tot_Profit',
       'Actual_Profit']]
ans1.to_csv("C:/Users/user/Downloads/Regionlization/Store transfer project/pulp_prod_full_output.csv")

################################
sku = 22739247
x = deficit_stores[deficit_stores['SKU']==sku]
y = surplus_stores[surplus_stores['SKU']==sku]
print(len(x))
print(len(y))
print(len(x)*len(y))

def optimize_sku(sku):
    print(sku)
    samp_deficit = deficit_stores_subset[deficit_stores_subset['SKU']==sku]

    samp_surplus = surplus_stores_subset[surplus_stores_subset['SKU']==sku] 
    
    #Dictionary of max units that can be shipped to each Deficit Store
    need = dict(samp_deficit[['STORE_NUM','difference']].values)
    print(need)
    #Dictionary of units each Surplus store will supply
    give = dict(samp_surplus[['STORE_NUM','difference']].values)
    print(give)
    #Dictionary that stores the priority ranks of each surplus store
    priority_surplus = dict(samp_surplus[['STORE_NUM','Surplus_Store_Weightage']].values)
    #Dictionary that stores the priority ranks of each deficit store
    priority_deficit = dict(samp_deficit[['STORE_NUM','Deficit_Store_Weightage']].values)

    
    if (sum(need.values())<sum(give.values())):
        need['0'] = sum(give.values()) - sum(need.values())
        samp_deficit.loc[len(samp_deficit.index)] = [sku, '0', need['0'],'0']
        priority_deficit['0'] = 0
        samp_dest = np.unique(samp_deficit['STORE_NUM']).tolist()
        samp_origin = np.unique(samp_surplus['STORE_NUM']).tolist()
        samp_surplus_Weightage = np.unique(samp_surplus['Surplus_Store_Weightage']).tolist()
        samp_deficit_Weightage = np.unique(samp_deficit['Deficit_Store_Weightage']).tolist()
        far = get_dist(samp_origin,samp_dest)
        samp_routes = [(i,j) for i in samp_origin for j in samp_dest]
        
    else:
        give['0'] = sum(need.values()) - sum(give.values())
        samp_surplus.loc[len(samp_surplus.index)] = [sku, '0', give['0'],0]
        priority_surplus['0'] = 0
        samp_dest = np.unique(samp_deficit['STORE_NUM']).tolist()
        samp_origin = np.unique(samp_surplus['STORE_NUM']).tolist()
        samp_surplus_Weightage = np.unique(samp_surplus['Surplus_Store_Weightage']).tolist()
        samp_deficit_Weightage = np.unique(samp_deficit['Deficit_Store_Weightage']).tolist()        
        far = get_dist(samp_origin,samp_dest)
        samp_routes = [(i,j) for i in samp_origin for j in samp_dest]

    #Set Problem variable
    Samp_Prob = LpProblem("Transportation",LpMinimize)
    
    #Decision variable
    #samp_amount_vars = LpVariable.dicts(name = "ShipUnits",indices = (samp_origin,samp_dest),lowBound=0)
    samp_amount_vars = LpVariable.dicts("ShipUnits",(samp_origin,samp_dest),0,None,LpInteger)

    #Objective Function
    Samp_Prob+= lpSum(samp_amount_vars[i][j]*far[i][j]*priority_surplus[i]*priority_deficit[j] for (i,j) in samp_routes)   
    #Constraints
#    if len(samp_dest)>1:
    for i in samp_origin:
        Samp_Prob+=lpSum(samp_amount_vars[i][j] for j in samp_dest)==give[i]

    for j in samp_dest:
        Samp_Prob+=lpSum(samp_amount_vars[i][j] for i in samp_origin)<=need[j]
        
#    print(Samp_Prob)
                
    Samp_Prob.solve()
    
    for v in Samp_Prob.variables():
       if v.varValue>0:
            item = sku 
            Route = v.name
            Qty = v.varValue
            Demand = need[v.name.split('_')[2]]
            data = [item,Route,Qty,Demand]
            distance = get_distance(Route.split('_')[1], Route.split('_')[2])
            print(v.name,'=',v.varValue)
            
    return far,need,give

far,need,give = optimize_sku(sku)


'''
