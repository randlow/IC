"""STAGE 1: 
IMPORT LIBRARY"""
import pandas as pd
import systemicRiskMeasures as srm   

"""STAGE 2: 
IMPORT DATA"""
src= pd.read_pickle('sourcedata.pickle')
src_returns= srm.logreturns(Returns=src)

"""STAGE 3: 
IMPORT SYSTEMIC RISK MEASURES AND RUN SIGNALS"""
Input= src_returns.resample('M') #input monthly returns

"""Mahalanobis Distance"""
        #Input
MD_input=Input           #Change this value for data required
        #Run
SRM_mahalanobis= srm.MahalanobisDist(Returns=MD_input)   
SRM_mahalanobis_turbulent_nonturbulent_days= srm.MahalanobisDist_Turbulent_Returns(MD_returns= SRM_mahalanobis, Returns=MD_input)
                    #drop inputs

Input=Input.drop('MD',1)

MD_input= MD_input.drop('MD',1)
       #Graph
SRM_HistoricalTurbulenceIndexGraph= srm.HistoricalTurbulenceIndexGraph_bokeh( Mah_Days=SRM_mahalanobis,  width=30, figsize=(10,2.5), datesize='M')

#-------------------------

Corr_Input= Input
SRM_Correlation_Surprise=srm.Correlation_Surprise(Returns=Corr_Input)
srm.Corr_plot_bokeh( Corr_sur=SRM_Correlation_Surprise[0], Mag_sur=SRM_Correlation_Surprise[1],  width=25, figsize=(10,4.5), datesize='M')

AR_input= Input
SRM_absorptionratio= srm.Absorption_Ratio(Returns= AR_input, halflife=int(500/12))
SRM_AR_plot= srm.plot_AR_bokeh(AR=SRM_absorptionratio, figsize=(10,2.5),yaxis=[0.84,0.9])

#------------------------- PROBIT
"""Stage5: 
Probit Model"""  
#----------------------------------
""" 
This model uses monthly returns to generate a forecasting Probit model
"""
# ----------------------------------
   #  1: INPUTS
        #Import Same Data for Comparision 
Input_returns=Input    #Start 1990-01-31

Start='19940131'
End='20140131' #20140630 latest St Louis Recession data date
window_range= 60  #months

Start_Recession_Values='19940101'
Balanced_port= srm.logreturns(Returns=pd.read_pickle('probit_portfolio.pickle')).resample('M',how='sum').loc[Start:End]
Recession_Values= pd.read_pickle('recession_values.pickle')
Recession_Values=srm.CreateDataFrameWithTimeStampIndex(DataFrame=Recession_Values)
Recession_Values= Recession_Values[Start_Recession_Values:]
SP500_TB= pd.read_pickle('probit_portfolio.pickle').resample('M').loc[Start:End]

Balanced_port= Balanced_port

Window_Range= 17+window_range       #10 year window   #Must be greater than 41 as Absorption Ratio requires 500day rolling window. Therefore the Window size is Window-41                                                           
Forecast_Range=len(Input_returns)-Window_Range +1               #Months
#---------------------------------------------------------
#--------------------------
    #  2: RUN PROBIT
#http://arch.readthedocs.org/en/latest/bootstrap/bootstrap_examples.html
Probit_Forecast=pd.DataFrame()


#writer.save()
for i in range(Forecast_Range):
    
    window= int(Window_Range) 
    Input= Input_returns[0:window+i]
    
    
    Recession_data= Recession_Values[0:window+i] #Set Input    
    Recession_data=pd.DataFrame(Recession_data.values, index=Input.index)    
    Probit_function=srm.Probit(Input_Returns=Input, recession_data=Recession_data)  #Generate Probit Parameters 
    
    Intercept= Probit_function[0][0]                                                       
    First_coeff= Probit_function[0][1]              
    Second_coeff= Probit_function[0][2]
    Third_coeff= Probit_function[0][3]
    Input_first_variable=Probit_function[1]['MD'].tail()[4]
    Input_second_varibale=Probit_function[1]['Mag_Corr'].tail()[4]
    Input_third_varibale=Probit_function[1]['AR'].tail()[4]
    Function= Intercept+ First_coeff*Input_first_variable + Second_coeff*Input_second_varibale + Third_coeff*Input_third_varibale
    #Create Probit Function and generate Forecast Value
    
    df=pd.DataFrame(index=(Input_returns[0:window+i+1].tail()[4:].index)) #Appending month ahead at the moment    
    df['Probit']=Function
    Probit_Forecast=Probit_Forecast.append(df) 
    #print ['Probit Iteration', i, 'Out of', Forecast_Range-1]
#When Probit is for that month grab that month...it is always forecasted ahead and will be appended with the Forecast month's results
#Therefore when chosing ...Probit is a certain value chose the month that it is appended to(which is the forecast) as this will take the results from that previous point
df=pd.DataFrame(index=((Input_returns[0:window+i+1].tail()[4:].index)+1)) #Appending month ahead at the moment    
df['Probit']=Function
Probit_Forecast=Probit_Forecast[0:len(Probit_Forecast)-1]
Probit_Forecast=Probit_Forecast.append(df) 
#---------------------------------------------------------
#----------------------------------
    #  3  Generating Switching Portfolfio Stradgy in combination with Threshold calculations
#1: Set Parameters
Probit_Forecast=Probit_Forecast[Probit_Forecast>-10].fillna(0) #outlier
 #Set Probit Forecasts for the previous input of US monthly returns
Rebalanced_portfolio= Balanced_port[Window_Range:]      #Set the Portfolio of Equities and Bonds at same starting date as Probit
Switch_Portfolio=pd.DataFrame()                     #Define Switch_Portfolio as empty dataframe to append values later in loops below
Theshold_Values=[]              #Set empty Theshold Value to append values form loop below
Returns_that_you_will_get=[]    #"" "" ""
Initial_Theshold=0                  #Let intial theshold equal 0          
Theshold=Initial_Theshold
for i in range(0,len(Probit_Forecast)-1):   #need to make sure it finishes at end and starts at orgin    
    """ What you will get"""
    Predicted_Probit_decider=Probit_Forecast['Probit'][i:i+1][0]                    #Grabs First Row
    Theshold=Theshold
    if (Predicted_Probit_decider>Theshold):   
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[i:1+i].ix[:,1:2]) #Fixed Income
    else:
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[i:1+i].ix[:,0:1])  #Equity 
    Switch_Portfolio=Switch_Portfolio.fillna(0)
    Returns_that_you_will_get.append(Switch_Portfolio.sum().sum())
       #-----------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
    #  4  RESULTS
#Probit Graph 

SRM_AR_plot= srm.plot_probit_bokeh(PF=Probit_Forecast, figsize=(10,3),yaxis=[0.84,0.9])