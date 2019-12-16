import pandas as pd
import numpy as np
#import xlrd
fname = "Traing_battery_DST.xlsx"
data= pd.read_excel("Traing_battery_DST.xlsx")

#data= data.sheet_by_index(0)
a=data['Current(A)']
#a = list(a)
#curr = np.array([a])
#print(curr)
#irate= curr.reshape(26674,1)
#print('Live current',curr.shape)
#print(irate)

v=data['Voltage(V)']
#volt = np.array([v])
#vrate= volt.reshape(26674,1)
#print('Live volt',vrate.shape)
#print(vrate)

t=data['Temp(C)']
#te=np.array(t)
#trate= te.reshape(26674,1)
#print('Live temprature',trate.shape)
#print(trate)

ah=data['cap(Ah)']
cha=np.array(ah)
charge= cha.reshape(26674,1)
#print('desierd charge',charge.shape)
#print(charge)

data_in =np.array([a,v,t])
data_in = data_in.T
print(data_in[15000])
#print(data_in)
#cl = list(data_in.columns)
#print(cl.head())
print('Input data matrix',data_in.shape)

data_ref = np.array(charge)
print('refrance data matrix',data_ref.shape)
#for a in range(0,data_in.shape[0]-2):
    #data_in.loc[data_in.loc[a+2], 'SMA_3'] = np.round(((data_in.loc[a,1]+data_in.loc[a + 2,1])/3),1)
    #data_in.head()
    