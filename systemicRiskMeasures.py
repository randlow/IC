def CreateDataFrameWithTimeStampIndex(DataFrame):
    import datetime 
    import pandas as pd    
    
    New_index=[]
    for i in range(len(DataFrame)):
        timestamp= datetime.datetime.strptime(DataFrame.index[i],'%Y-%m-%d')
        New_index.append(timestamp)
    New_DataFrame=pd.DataFrame(DataFrame.values, index=New_index, columns=DataFrame.columns)    
  
  #--------------------------------------------------------------------------- 
    return New_DataFrame
   #---------------------------------------------------------------------------


def logreturns(Returns):    #GENERATED LOGARITHMIC RETURNS
    
    import numpy as np    
        
    returns = np.log(Returns/Returns.shift(1)).dropna()  #Generate log returns
    resampled_data=returns.resample('d').dropna()                              #Choose if Daily, Monthly, Yearly(ect) dataframe is required
  
  #--------------------------------------------------------------------------- 
    return   resampled_data                                                    #Return Log returns
   #---------------------------------------------------------------------------

                ##Systemic Risk Measures##

#Journal Article: Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
def MahalanobisDist(Returns):                                                  #define MahalanobisDistance function with Returns being a singal dataFrame with n number of columns
  
        #stage1: IMPORT LIBRARIES
    import pandas as pd                                                        #import pandas    
    import numpy as np                                                         #import numpy
    
        #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= Returns.cov()                                           #Generate covariance matrix for historical returns
    return_inverse= np.linalg.inv(return_covariance)                           #Generate inverse covariance matrix for historical returns

        #stage3: CALCULATE THE DIFFERENCE BETWEEN SAMPLE MEAN AND HISTORICAL DATA
    means= Returns.mean()                                                      #Calculate means for each asset's historical returns 
    diff_means= Returns.subtract(means)                                         #Calculate difference between historical return means and the historical returns

        #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values                                                   #Split historical returns from Dataframe index
    dates= diff_means.index                                                    #Split Dataframe index from historical returns

        #stage5: BUILD FORMULA
    md = []                                                                    #Define Mahalanobis Distance as md and create empty array for iteration
    for i in range(len(values)):
        md.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))#Construct Mahalanobis Distance formula and iterate over empty md array
        
        #stage6: CONVERT LIST TYPE TO DATAFRAME TYPE
    md_array= np.array(md)                                                     #Translate md List type to md Numpy Array type in order to join values into a Dataframe
    MD_daily=pd.DataFrame(md_array,index=dates,columns=list('R'))              #Join Dataframe index and Numpy array back together
    #MD_monthly= MD_daily.resample('M')                                         #resample data by average either as daily, monthly, yearly(ect.) 
   
   #---------------------------------------------------------------------------
    return    MD_daily                                                         #Return Malanobis Distance resampled returns, Malanobis Distance daily returns,  Turbulent returns and non-Turbulent returns
   #---------------------------------------------------------------------------


def MahalanobisDist_Turbulent_Returns(MD_returns, Returns):
    
    #Turbulent Returns
    turbulent= MD_returns[MD_returns>MD_returns.quantile(.75)[0]].dropna()
        #Day_with_Turbulent_returns
    returns=Returns
    returns['MD']=MD_returns
    Turbulent_Days=returns[returns['MD']>MD_returns.quantile(.75)[0]]
    Turbulent_Days= Turbulent_Days.drop('MD', 1)
    
    #Non_turbulent Returns
    non_turbulent=MD_returns[MD_returns<MD_returns.quantile(.75)[0]].dropna()
        #Day_with_non_Turbulent_returns
    non_Turbulent_Days=returns[returns['MD']<MD_returns.quantile(.75)[0]]
    non_Turbulent_Days= non_Turbulent_Days.drop('MD', 1)
    
    Returns=Returns.drop('MD',1)
    
   #---------------------------------------------------------------------------
    return turbulent, non_turbulent, Turbulent_Days,non_Turbulent_Days
   #---------------------------------------------------------------------------
    
   

def HistoricalTurbulenceIndexGraph( Mah_Days,  width, figsize, datesize):
    import matplotlib.pyplot as plt
    
    
    Monthly_Mah_Returns= Mah_Days.resample(datesize)   
    Monthly_Mah_Turbulent_Returns= Monthly_Mah_Returns[Monthly_Mah_Returns>Monthly_Mah_Returns.quantile(.75)[0]].dropna()    
    
    fig= plt.figure(1, figsize=figsize)
    plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
    plt.xlabel('Year')                                                          #label x axis Year
    plt.ylabel('Index')                                                         #label y axis Index
    plt.suptitle(['Historical Turbulence Index Calcualted from', datesize, 'Returns'],fontsize=12)   
    plt.bar(Monthly_Mah_Returns.index,Monthly_Mah_Returns.values, width,color='w', label='Quiet')#graph bar chart of Mahalanobis Distance
    plt.bar(Monthly_Mah_Turbulent_Returns.index,Monthly_Mah_Turbulent_Returns.values, width,color='k',alpha=0.8, label='Turbulent')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)      
    plt.show()
   
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------

# Modified HistoricalTurbulenceIndexGraph-version for visualization with Bokeh
def HistoricalTurbulenceIndexGraph_bokeh( Mah_Days,  width, figsize, datesize):
    from bokeh.plotting import figure, output_file, show #, VBox
    
    Monthly_Mah_Returns= Mah_Days.resample(datesize)
    Monthly_Mah_Turbulent_Returns= Monthly_Mah_Returns[Monthly_Mah_Returns>Monthly_Mah_Returns.quantile(.75)[0]].dropna()
    
    # Get 'R'-List out    
    Monthly_Mah_Returns = Monthly_Mah_Returns['R']
    Monthly_Mah_Turbulent_Returns = Monthly_Mah_Turbulent_Returns['R']
    
    # Export data for sphinx/bokeh
    Monthly_Mah_Returns.to_pickle('hti-quiet.pickle')
    Monthly_Mah_Turbulent_Returns.to_pickle('hti-turbulent.pickle')
    
    """ From this point not necessary any more if combined with sphinx """
    
    # Define graph
    gra = figure(title="Historical Turbulence Index",
                 x_axis_label="Year",
                 y_axis_label="Index",
                 x_axis_type="datetime",
                 y_range=[0, 140],
                 plot_width=900,
                 plot_height=300,
                 logo=None)
    
    # Set the footer labels (including zoomed-state)
    gra.below[0].formatter.formats = dict(years=['%Y'],
                                          months=['%b %Y'],
                                          days=['%d %b %Y'])
    
    # Paint quiet graph
    gra.segment(x0=Monthly_Mah_Returns.index,
                y0=0,
                x1=Monthly_Mah_Returns.index,
                y1=Monthly_Mah_Returns.values,
                color='#000000',legend='Quiet')
    
    # Paint turbulent graph
    gra.segment(x0=Monthly_Mah_Turbulent_Returns.index,
                y0=0,
                x1=Monthly_Mah_Turbulent_Returns.index,
                y1=Monthly_Mah_Turbulent_Returns.values,
                color='#BB0000',legend='Turbulent')

    #output_file("hti.html", title="Historical Turbulence Index")
    #show(VBox(gra))
   
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------



def MahalanobisDist_Table1(Market_Returns):
    
    #stage 1: IMPORT LIBRARIES 
    import pandas as pd
        
    for a in range(len(Market_Returns)):
        #stage 2: IMPORT MAHALANOBIS DISTANCE RETURNS
        Mah_Dist_Retunrs= MahalanobisDist(Market_Returns[a])
    
       #stage 3: NORMALISE MAHALANOBIS DISTANCE RETURNS
        Max=  Mah_Dist_Retunrs.max()                           #Let x equal the iterated row of previously itereated column
        Min=   Mah_Dist_Retunrs.min()                                           #Calculate normalised return
        Normalised_Data=(Mah_Dist_Retunrs-Min)/(Max-Min)                                     #Append normalised returns empty array to DataFrame
    
        #stage 4: GENERATE TABLE 1 DATA
        Next_5=[]                                                                  #Create empty array of Next 5 days
        Next_10=[]                                                                 #Create empty array of Next 10 days
        Next_20=[]                                                                 #Create empty array of NExt 20 days
        index=[] 
        Top_75_Percentile= Normalised_Data[ Normalised_Data> Normalised_Data.quantile(.75)[0]].dropna()   
        Top_10_Percentile_of_75=Top_75_Percentile[ Top_75_Percentile> Top_75_Percentile.quantile(.1)[0]].dropna() 
        for i in range(len(Top_10_Percentile_of_75)):
            x= Top_10_Percentile_of_75.index[i]
            for j in range(len(Normalised_Data)):
                if (x==Normalised_Data.index[j]):
                    x=Normalised_Data['R'][j+1:j+6].mean()                             #Calcualte mean of 5 days after 
                    y=Normalised_Data['R'][j+1:j+11].mean()                            #Calcualte mean of 10 days after 
                    z=Normalised_Data['R'][j+1:j+21].mean()  
                    zz=Normalised_Data.index[i] 
                    Next_5.append(x)                                                   #Append mean of 5 days after to  Next_5 empty array
                    Next_10.append(y)                                                  #Append mean of 10 days after to  Next_10 empty array
                    Next_20.append(z)                                                  #Append mean of 20 days after to  Next_20 empty array
                    index.append(zz)  
                Table_1=pd.DataFrame(index=index)                                          #Create Table 1 DataFrame over most turbulent days index
                Table_1['Next 5 Days']= Next_5                                             #Append  Next_5 array to Dataframe
                Table_1['Next 10 Days']= Next_10                                           #Append  Next_10 array to Dataframe
                Table_1['Next 20 Days']= Next_20 
        
          
        #Create DataFrame
#Rows= ['Global Assets', 'US Assets', 'US Sectors', 'Currencies']
#pd.DataFrame(columns=(,index=rows,'Next 5','Percentile Rank','Next 10','Percentile Rank','Next 20','Percentile Rank','10th Threshold'))
                
        if a==0:
            
            Global_Assets_returns=Table_1.mean()
        elif a==1:
            US_Assets_returns=Table_1.mean()
        elif a==2:
            Currency_returns=Table_1.mean()
            
    Persistence_of_Turbulence_for_next_5_10_20= Global_Assets_returns,US_Assets_returns,Currency_returns
            
#from scipy import stats
#stats.percentileofscore(Table_1.mean(),Currency_returns[0][0])    #example of comparing the percentile   
            
            #need to locate percentiles of each average turbuelence returns and find out if my 10% threshold is correct
            
                   
    return Persistence_of_Turbulence_for_next_5_10_20                                #Return Table 1,  return Top_75 Percentile of Normalised Data
   #---------------------------------------------------------------------------

    
def MahalanobisDist_Table2(Asset_Class,Weights):
    
    import pandas as pd
    import numpy as np
       
    
   #Stage1: Create Portfolios
    Conservative_Portfolio=Weights[0]*Asset_Class
    Moderate_Portfolio=Weights[1]*Asset_Class
    Aggressive_Portfolio=Weights[2]*Asset_Class
    Portfolios=Conservative_Portfolio,Moderate_Portfolio,Aggressive_Portfolio
    
    #Stage2: Expeceted Returns 
    Expected_Return=[]
    for i in range(len(Portfolios)):
        expected_return= Portfolios[i].sum().sum()
        Expected_Return.append(expected_return)
       
    #Stage3: Full-Sample Risk
    Full_sample_risk=[]   
    for i in range(len(Portfolios)):
        Risk=np.sqrt(np.diagonal((Portfolios[i]).cov()).sum())
        Full_sample_risk.append(Risk)
        
    #Stage4: Turbulent Risk
    #CREATE DATAFRAME OF RETURN VALUES FOR EVERY DATE THAT GENERATES A TOP 75% TURBULENCE SCORE 
    Turbulent_Risk=[]
    for i in range(len(Portfolios)):
        MahalanobisDist_returns= MahalanobisDist(Returns=Portfolios[i])
        Turbulent_days= MahalanobisDist_Turbulent_Returns(MD_returns= MahalanobisDist_returns, Returns=Portfolios[i])[2]
        turbulent_risk=  np.sqrt(np.diagonal((Turbulent_days).cov()).sum())   
        Turbulent_Risk.append(turbulent_risk)
    
                #stage 5: Create Table
    Portfolio_Rows= list(Asset_Class.columns.values)
    All_table_rows= Portfolio_Rows.extend(('Expected Return', 'Full-Sample Risk','Turbulent Risk'))

    Table_2= pd.DataFrame(index=Portfolio_Rows) 
    Weights[0].extend([Expected_Return[0], Full_sample_risk[0], Turbulent_Risk[0]]) 
    Weights[1].extend([Expected_Return[1], Full_sample_risk[1], Turbulent_Risk[1]])   
    Weights[2].extend([Expected_Return[2], Full_sample_risk[2], Turbulent_Risk[2]])                                   #Create open Table_2
    Table_2['Conservative Portfolio(%)']= Weights[0]                            #Append each Portfolio's Data to Table 2
    Table_2['Moderate portfolio(%)']= Weights[1]
    Table_2['Aggressive Portfolio(%)'] = Weights[2]
    
    
    return Table_2*100, Portfolios
   #---------------------------------------------------------------------------

    
     
def MahalanobisDist_Table3(Portfolios, beta): 
    
    import pandas as pd
      
    #VaR for Full Sample, End of Horizon
    VaR_for_Full_Sample = []  
    for i in range(len(Portfolios)):
        VaR= abs(Portfolios[i].quantile(beta)).mean()    
        VaR_for_Full_Sample.append(VaR)
        
    #VaR for Turbulent Sample, with Horizon
    VaR_for_Turbulent_Sample = []  
    for i in range(len(Portfolios)):
        MahalanobisDist_returns= MahalanobisDist(Portfolios[i])
        Turbulent_days= MahalanobisDist_Turbulent_Returns(MD_returns= MahalanobisDist_returns, Returns=Portfolios[i])[2] 
        VaR= abs(Turbulent_days.quantile(beta)).mean()    
        VaR_for_Turbulent_Sample.append(VaR)

    Table_3= pd.DataFrame(index=('Conservative','Moderate','Aggressive'))
    Table_3['VaR_for_Full_Sample(%)']= VaR_for_Full_Sample
    Table_3['VaR_for_Turbulent_Sample(%)']= VaR_for_Turbulent_Sample
    
    #IS maximum drawdown that of the returns?
    #maximum drawdown

                
    return  Table_3*100   
   #---------------------------------------------------------------------------


def regression(Primarily_return, Secondary_returns):
    from scipy import stats
    import numpy as np
    x= Primarily_return
    y=Secondary_returns
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
  
  #---------------------------------------------------------------------------
    return intercept,slope, std_err, r_value, p_value, std_err
   #---------------------------------------------------------------------------

    
def shrinkage_ret_est(sample):
    import numpy as np
        
    u= np.array(sample.mean())
    returns=sample
        
    #Calculate u_m (shrinkage target)        
    e_size= np.zeros(shape=(1,len(sample.columns)))
    e_filled= e_size.fill(1)
    e= np.transpose(e_size)

    Covariance_Matrix_inverse= np.linalg.inv(returns.cov())
 
    u_m_numerator= np.dot((np.transpose(e)*Covariance_Matrix_inverse),u)
    u_m_denominator= np.dot((np.transpose(e)*Covariance_Matrix_inverse),e)
    u_m=np.divide(u_m_numerator, u_m_denominator)[0][0]                             #grab value out from array so it is a number


    n=len(sample.columns)
    T=len(returns)
    w_numerator= n+2
    w_denominator=n+2+T*(np.dot(np.transpose(u-u_m*e),(np.dot(Covariance_Matrix_inverse,u-u_m*e))))
    w=np.divide(w_numerator, w_denominator)[0][0]

    u_s=w*u_m +(1-w)*u
    
    #---------------------------------------------------------------------------
    return u_s   
   #---------------------------------------------------------------------------
    

def shrinking_cov(Market_Portfolio):
   
    import numpy as np

    x=Market_Portfolio.values

    t= x.shape[0]
    n= x.shape[1]
    meanx= x.mean(0)
    e_size= np.zeros(shape=(1,x.shape[0]))
    e_filled= e_size.fill(1)
    e= np.transpose(e_size)
    x=x-meanx*e
    xmkt= np.transpose(np.transpose(x).mean(0))
    xmkt=np.transpose((xmkt*e[1:2]))
    
    sample= np.divide((np.cov(np.transpose((np.concatenate((x, xmkt), 1)))))*(t-1),t) 
    covmkt= (sample[:,n][0:n])
    covmkt=np.reshape(covmkt, (n, 1))
    varmkt= sample[:,n][n:n+1]
    sample= sample[:,0:n]
    sample= sample[0:n,:]
    prior=(covmkt*np.transpose(covmkt))/varmkt
    np.fill_diagonal(prior, np.diagonal(sample))

    #shrinkage
    m_size= np.zeros(shape=(1,n))
    m_filled= m_size.fill(1)
    matrix_ones= m_size

    c= np.square(np.linalg.norm(sample-prior, ord= 'fro'))
    y= np.square(x)
    p= (1./t)*np.dot(np.transpose(y),y).sum(0).sum() - np.square(sample).sum(0).sum()
    rdiag=(1./t)*np.square(y).sum(0).sum() - np.square(np.diagonal(sample)).sum()
    z= x*xmkt
    v1= (1./t)*np.dot(np.transpose(y),z) - np.dot(covmkt,matrix_ones)*sample                #matrix ones is 1,49 matrix of ones
    roff1= ((v1*np.transpose(np.dot(covmkt,matrix_ones))).sum(0).sum())/varmkt - ((np.reshape(v1.diagonal(),(n,1))*covmkt).sum())/varmkt
    v3=(1./t)*np.dot(np.transpose(z),z) - varmkt*sample
    roff3=((v3*(covmkt*np.transpose(covmkt))).sum(0).sum())/np.square(varmkt) -((np.reshape(v3.diagonal(),(n,1))*np.square(covmkt)).sum())/np.square(varmkt)
    roff= 2*roff1 - roff3
    r= rdiag+roff
    k=(p-r)/c
    shrinkage= np.reshape(np.max(np.reshape(np.min(k/t),(1,1))),(1,1))

    sigma= shrinkage*prior + (1-shrinkage)*sample
    
    #---------------------------------------------------------------------------
    return sigma
   #---------------------------------------------------------------------------

def Mean_Variance(portfolio_means, portfolio_covariance, num_columns):   
   #step6: Implement Mean-Variance
   import numpy as np
   import pandas as pd
    
   """Step1:
   Import Data as DataFrame"""
   #-------------
   rets=[portfolio_means,portfolio_covariance]
   noa=num_columns                                #Generate Len of Columns
       #log returns
           #rets= np.log(data/data.shift(1))
       #-------------
       
   """Step2:
   Portfolio Optim"""
   import scipy.optimize as sco
   import sklearn.covariance
   #Step1:
   #----------------
   #Function returns major portfolio statistics for an input weights vector/array
   def statistics(weights):               
        """ Returns portfolio statistics.
        Parameters
        ==========
        weights : array-like
        weights for different securities in portfolio
        Returns
        =======
        pret : float
        expected portfolio return
        pvol : float
        expected portfolio volatility
        pret / pvol : float
        Sharpe ratio for rf=0
        """
        weights = np.array(weights)
        mean=rets[0]
        covariance=rets[1]
        pret = np.sum(mean * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(covariance * 252, weights)))
        return np.array([pret, pvol, pret / pvol])
           
     #Minimise Variance
   def min_func_variance(weights):
       return statistics(weights)[1] ** 2
     #----------------
       
     #Step2: Add Constraints
     #-------------------
   cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #list constraint that all weights add to 1    
   bnds = tuple((0, 1) for x in range(noa)) #bound weights so that values are only within 0 and 1
   Equal_weights= noa * [1. / noa,]  #Input an equal distribution of weights
    #-------------------
       
   #Step3: Generate Optimised Returns
   #-------------------
   optv = sco.minimize(min_func_variance, Equal_weights, method='SLSQP', bounds=bnds,  constraints=cons)
   Mean_var= optv     
   Optimal_weights= optv['x'].round(3)
   ER_VOL_SHAR =statistics(optv['x']).round(3)
    #print 'Optimised Vol Weights'
     #print optv['x'].round(3)
     #print #
     #print 'Expected Returns:' ,statistics(optv['x']).round(3)[0]
     #print 'Volatility:' ,statistics(optv['x']).round(3)[1]
     #print 'Sharpe:' ,statistics(optv['x']).round(3)[2]
   
   
   #first half
       
   return   "Optimal_weights" ,Optimal_weights, "Expected Return, Volatility and Sharpe Ratio",ER_VOL_SHAR 
    

def Mod_Mean_Var(portfolio, full_trailing): 
    
   import numpy as np 
   """Equilibrium Returns is constructed as a portfolio 
   of 60% US equities, 30% T bonds and 10% US Corporate Bonds"""
   full_training_sample=full_trailing
      
   #Step1:  Estimate unconditional expected returns
   market_portfolio= portfolio    #equilibrium returns on the basis of full training sample
  # market_portfolio_based_full= 
   
   market_portfolio_means= np.array(market_portfolio.mean())
   market_portfolio_mean= (market_portfolio_means.mean())*np.reshape(np.ones(len(full_training_sample.columns)), (1,len(full_training_sample.columns)))

   #---------------------------------------------------------------------------
   
   #Step2: Estimate Conditional Expected returns
   full_training_sample=full_trailing
       #Compute Average returns of the Turbulent subsample
              #Generate Turbulent Subsample
   MahalanobisDist_returns= MahalanobisDist(Returns= full_training_sample).resample('M')
   Turbulent_days= MahalanobisDist_Turbulent_Returns(MD_returns= MahalanobisDist_returns, Returns=full_training_sample)[2]
   full_training_sample= full_training_sample.drop('MD',1)
               #Estimate return after shrinkage
   u_t= shrinkage_ret_est(Turbulent_days)
   
          #Compute Average returns of the Full training sample
               #Estimate return after shrinkage
   u_f= shrinkage_ret_est(full_training_sample)
   
       #Blend difference between compressed sub and full sample with equilibrium returns
   Turbulent_ratio, Non_Turbulent_ratio  = (float(len(Turbulent_days))/(len(full_training_sample))),(1-(float(len(Turbulent_days))/(len(full_training_sample))))
   
   u_c_t= (Turbulent_ratio*u_t + Non_Turbulent_ratio*market_portfolio_mean)/2
   u_c_f= (0.5*u_f + 0.5*market_portfolio_mean)/2
   #---------------------------------------------------------------------------
   
   #step3: Estimate unconditional covariances by shrinking sample covariance
   import pandas as pd
   est_uncon_cov= shrinking_cov(Market_Portfolio=full_training_sample)
   est_uncon_cov= pd.DataFrame(est_uncon_cov,index=full_training_sample.cov().index)
   est_uncon_cov.columns= full_training_sample.cov().index
   #---------------------------------------------------------------------------

   #step4: Estimate the conditioned covariances
   full_sample_covariance= full_training_sample.cov() 
   est_con_cov= Turbulent_ratio*shrinking_cov(Market_Portfolio=Turbulent_days)+ Non_Turbulent_ratio*full_sample_covariance
   #---------------------------------------------------------------------------

    #Step5: Group Data
   Unconditional_portfolio= [market_portfolio_means, est_uncon_cov]
   Conditioned_portfolio_t= [u_c_t, est_con_cov]
   Conditioned_portfolio_f= [u_c_f, est_con_cov]
   
   portfolio_list=[Unconditional_portfolio,Conditioned_portfolio_t,Conditioned_portfolio_f]
   sample_list= [market_portfolio, full_training_sample,Turbulent_days] 
   
   Unconditional_portfolio_Mod_Mean_Var= Mean_Variance(portfolio_means=Unconditional_portfolio[0], portfolio_covariance=Unconditional_portfolio[1], num_columns=len(sample_list[0].columns))
   Conditioned_portfolio_t_Mod_Mean_Var= Mean_Variance(portfolio_means=Conditioned_portfolio_t[0], portfolio_covariance=Conditioned_portfolio_t[1], num_columns=len(sample_list[1].columns))
   Conditioned_portfolio_f_Mod_Mean_Var= Mean_Variance(portfolio_means=Conditioned_portfolio_f[0], portfolio_covariance=Conditioned_portfolio_f[1], num_columns=len(sample_list[2].columns))

   return 
   #---------------------------------------------------------------------------
    #Proble is that unconditional portfolio has weights of anything but he est_covariance will always be differnt due to using other sample    
    
    
    
    
    #Need to create expanding window:
    #This means that optimal weights must be calculated for every window

def Mod_Mean_Var_Exp_Wind(sample, market_portfolio):
    
    #Define portfolio covered over window    
    unweighted_intial_portfolio= sample    
    
    #set intial port    
    j=10    #window size
    noa= len(sample.columns)  #number of assets
    initial_weights= noa * [1. / noa,]       #equal weights for example
    initial_rets= sample[0:j]    #Grab set window of intial returns
    initial_portfolio=initial_rets*initial_weights     #construct intial portfolio window
       
            #Calculate mean_variance for given sample and market portfolio
    Rebalanced_portfolio= intial_portfolio
    for i in range(1,len(sample)-j):        #iterate 
        grab_next_month= unweighted_intial_port[i+j-1:i+j]
        weighted_next_month= Mod_Mean_Var(portfolio=Rebalanced_port, full_trailing=Rebalanced_port)[1] * grab_next_month
        Rebalanced_port= Rebalanced_p=ort.append(weighted_next_month)  
    
    
    return  Rebalanced_port   
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
    
    
#Journal Article: Kinlaw and Turkington - 2012 - Correlation Surprise
def Correlation_Surprise(Returns):
    
        #Stage1: IMPORT LIBRARIEs
    import pandas as pd                                                        #import pandas 
    import numpy as np                                                         #import numpy
     
         #Step1: CALCULATE TURBULENCE SCORE 
     
    #Stage 1: GENERATE TURBULENCE SCORE
    TS_daily= MahalanobisDist(Returns)                                               #calculate Turbulence Score from Mahalanobis Distance Function
    
             #Step2: CALCULATE MAGNITUDE SURPRISE   
    
        #Stage1: CALCULATE COVARIANCE MATRIX
    return_covariance= Returns.cov()                                           #Generate Covariance Matrix for hisotircal returns
    return_inverse= np.linalg.inv(return_covariance)                           #Generate Inverse Matrix for historical returns
    
        #stage2: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= Returns.mean()                                                      #Calculate historical returns means
    diff_means=Returns.subtract(means)                                         #Calculate difference between historical return means and the historical returns
    
        #stage3: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values                                                   #Split historical returns data from Dataframe
    dates= diff_means.index                                                    #Split Dataframe from historical returns
    
        #Stage4: Create Covariance and BLINDED MATRIX 
    inverse_diagonals=return_inverse.diagonal()                                #fetch only the matrix variances
    inverse_zeros=np.zeros(return_inverse.shape)                               #generate zeroed matrix with dynamic sizing properties 
    zeroed_matrix=np.fill_diagonal(inverse_zeros,inverse_diagonals)            #combine zeroed matrix and variances to form blinded matrix
    blinded_matrix=inverse_zeros                                               #define blinded matrix once the step above is completed
    
        #stage5: BUILD FORMULA
    ms = []                                                                    #Define Magnitude Surprise as ms                
    for i in range(len(values)):
        ms.append((np.dot(np.dot(np.transpose(values[i]),blinded_matrix),values[i])))       

        #stage6: CONVERT LIST Type TO DATAFRAME Type    
    ms_array= np.array(ms)                                                     #Translate ms List type to ts Numpy type
    Mag_Surprise_Daily=pd.DataFrame(ms_array,index=dates,columns=list('R'))               #Join Dataframe and Numpy array back together to calculate daily Magnitude Surprise Returns
    #MS=Mag_Surprise_Daily.resample('M')                                                   #create monthly returns for magnitude surprise
    
        
            #step3:CALCULATE CORRELATION SURPRISE
        #stage1: CALCULATE CORRELATION SURPRISE
    Corre_Surprise_Daily= TS_daily/(Mag_Surprise_Daily)   

                             # Calculate daily Correlation Surprise returns
    
    #Correlation_monthly_trail= Corre_Sur*Mag_Sur                                
    #resample_Correlation_monthly= Correlation_monthly_trail.resample('M',how=sum) 
    #MS_sum=Mag_Sur.resample('M',how=sum)                                       #Calculate monthly Magnitude Surprise returns 
    #Correlation_Surprise_monthly=resample_Correlation_monthly.divide(MS_sum)   #Calculate monthly Correlation Surprise retuns
    
    return  Corre_Surprise_Daily, Mag_Surprise_Daily               # Return Monthly Correlation Surprise Returns,  Monthly Magnitude Surprise returns, daily Correlation Surprise returns and daily magnitude surprise returns

   #---------------------------------------------------------------------------


def Corr_plot( Corr_sur, Mag_sur,  width, figsize, datesize):
    import matplotlib.pyplot as plt
    
    Corr_sur= Corr_sur.resample(datesize)
    Mag_sur= Mag_sur.resample(datesize)
    
    fig= plt.figure (figsize=figsize)
    fig.add_subplot(211)
    plt.xlabel('Year')                                                          #label x axis Year
    plt.ylabel('Index')                                                         #label y axis Index
    plt.suptitle(['Correlation Surprise and Magnitude Surprise', datesize, 'Returns'],fontsize=12)   
    plt.bar(Corr_sur.index,Corr_sur.values,color='w', width=width ,label= 'Correlation Surprise')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    
    fig.add_subplot(212)
    plt.xlabel('Year')                                                          #label x axis Year
    plt.ylabel('Index')                                                         #label y axis Index
    plt.bar(Mag_sur.index,Mag_sur.values,color='b', width=width ,label= 'Magnitude Surprise')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    

    plt.show()
    
    
    
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------
  
  
# Modified Corr_plot-version for visualization with Bokeh  
def Corr_plot_bokeh( Corr_sur, Mag_sur,  width, figsize, datesize):
    from bokeh.plotting import figure, output_file, show #, VBox
    
    Corr_sur= Corr_sur.resample(datesize)
    Mag_sur= Mag_sur.resample(datesize)
    
    # Get 'R'-List out    
    Corr_sur = Corr_sur['R']
    Mag_sur = Mag_sur['R']
    
    # Export data for sphinx/bokeh
    Corr_sur.to_pickle('cms-correlation.pickle')
    Mag_sur.to_pickle('cms-magnitude.pickle')
    
    """ From this point not necessary any more if combined with sphinx """

    # Define graph
    gra = figure(title="Correlation Surprise",
                 x_axis_label="Year",
                 y_axis_label="Index",
                 x_axis_type="datetime",
                 y_range=[0, 0.3],
                 plot_width=900,
                 plot_height=300,
                 logo=None)
    
    # Set the footer labels (including zoomed-state)
    gra.below[0].formatter.formats = dict(years=['%Y'],
                                          months=['%b %Y'],
                                          days=['%d %b %Y'])
    
    # Paint graph
    gra.segment(x0=Corr_sur.index,
                y0=0,
                x1=Corr_sur.index,
                y1=Corr_sur.values,
                color='#000000')
                
    # Define graph
    grb = figure(title="Magnitude Surprise",
                 x_axis_label="Year",
                 y_axis_label="Index",
                 x_axis_type="datetime",
                 y_range=[0, 25000],
                 plot_width=900,
                 plot_height=300,
                 logo=None)
    
    # Set the footer labels (including zoomed-state)
    grb.below[0].formatter.formats = dict(years=['%Y'],
                                          months=['%b %Y'],
                                          days=['%d %b %Y'])
    
    # Paint graph
    grb.segment(x0=Mag_sur.index,
                y0=0,
                x1=Mag_sur.index,
                y1=Mag_sur.values,
                color='#000000')
    
    #output_file("cms.html", title="Correlation and Magnitude Surprise")
    #show(VBox(gra, grb))
   
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------


def Conditional_ave_magn_sur_on_day_of_the_reading(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency):
    
    import pandas as pd
            
    #stage1: Generate Correlation Surprise Returns for US Equities, European Equities and Currencies
    CS_Exhibit5_USEquities= Correlation_Surprise(Returns=Exhibit5_USEquities)  # CS returns for USEquities
    CS_Exhibit5_EuropeanEquities= Correlation_Surprise(Returns=Exhibit5_Euro_Equities)# CS returns for EuropeanEquities
    CS_Exhibit5_Currency= Correlation_Surprise(Returns=Exhibit5_Currency)      # CS returns for Currency
    Correlation_Measures=CS_Exhibit5_USEquities, CS_Exhibit5_EuropeanEquities, CS_Exhibit5_Currency #Group CS returns together labelled "Correlation_Measures"
    
    #Stage2: Calculate Exhibit 5 returns 
    Exhibit5_USEquities_returns=[]                                             #Create empty array  USEquities_return
    Exhibit5_EuropeanEquities_returns=[]                                       #Create empty array  EuroEquities_return
    Exhibit5_Currency=[]                                                       #Create empty array  Currency
    Exhibit5_returns= Exhibit5_USEquities_returns, Exhibit5_EuropeanEquities_returns, Exhibit5_Currency #Group Empty Arrays together

#Step1: Identify the 20% of days in the historical sample with the highest magnitude surprise scores    
    for i in range(len(Exhibit5_returns)):                                     #Begin by iterating over the index of Exhibit5_returns to calculate returns for US Equities, Euro Equities and Currency individually
        Top_20_Percentile_Magnitude_Surprise= Correlation_Measures[i][1][Correlation_Measures[i][1]>Correlation_Measures[i][1].quantile(.80)[0]].dropna() #Calculate top 20% Magnitude Surprise returns for given returns
            
#Step2: Partition the sample from step 1 into two smaller subsamples: days with high correlation surprise and days with low correlation surprise     
    
        Top_20_Percentile_Magnitude_Surprise['CS']= Correlation_Measures[i][0]
        MS_20_CS_less_1= Top_20_Percentile_Magnitude_Surprise[Top_20_Percentile_Magnitude_Surprise['CS']<=1]
        MS_20_CS_greater_1= Top_20_Percentile_Magnitude_Surprise[Top_20_Percentile_Magnitude_Surprise['CS']>1]
        Top_20_Percentile_Magnitude_Surprise.drop('CS',1)
        MS_20_CS_less_1.drop('CS',1)
        MS_20_CS_greater_1.drop('CS',1)
            
        Average_Top_20_Percentile_Magnitude_Surprise= Top_20_Percentile_Magnitude_Surprise.mean()[0]           
        Average_MS_20_CS_less_1= MS_20_CS_less_1.mean()[0]
        Average_MS_20_CS_greater_1= MS_20_CS_greater_1.mean()[0]
        Exhibit5_returns[i].extend((Average_Top_20_Percentile_Magnitude_Surprise,Average_MS_20_CS_less_1,Average_MS_20_CS_greater_1))
        
     #Create Table
    Table_2= pd.DataFrame(index=['Day follow top 20% MS with CS<=1','Day follow top 20% MS','Day follow top 20% MS with CS>1'])
    Table_2['US Equities']= Exhibit5_returns[0]
    Table_2['European Equities']= Exhibit5_returns[1]
    Table_2['Currencies']= Exhibit5_returns[2]
    
    return Table_2, MS_20_CS_less_1,MS_20_CS_greater_1                           #There returns are like this because no Correlation Surprise is greater than 1 within the top 20$ of mangitude surprise returns
            
   #---------------------------------------------------------------------------


def Conditional_ave_magn_sur_on_day_after_reading(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency):
    
    import pandas as pd
    
    #stage1: Generate Correlation Surprise Returns for US Equities, European Equities and Currencies
    CS_Exhibit5_USEquities= Correlation_Surprise(Returns=Exhibit5_USEquities)  # CS returns for USEquities
    CS_Exhibit5_EuropeanEquities= Correlation_Surprise(Returns=Exhibit5_Euro_Equities)# CS returns for EuropeanEquities
    CS_Exhibit5_Currency= Correlation_Surprise(Returns=Exhibit5_Currency)      # CS returns for Currency
    Correlation_Measures=CS_Exhibit5_USEquities, CS_Exhibit5_EuropeanEquities, CS_Exhibit5_Currency #Group CS returns together labelled "Correlation_Measures"
    
    #Stage2: Calculate Exhibit 5 returns 
    Exhibit5_USEquities_returns=[]                                             #Create empty array  USEquities_return
    Exhibit5_EuropeanEquities_returns=[]                                       #Create empty array  EuroEquities_return
    Exhibit5_Currency=[]                                                       #Create empty array  Currency
    Exhibit5_returns= Exhibit5_USEquities_returns, Exhibit5_EuropeanEquities_returns, Exhibit5_Currency #Group Empty Arrays together
    
    for i in range(len(Exhibit5_returns)):                                     #Begin by iterating over the index of Exhibit5_returns to calculate returns for US Equities, Euro Equities and Currency individually
        Top_20_Percentile_Magnitude_Surprise= Correlation_Measures[i][1][Correlation_Measures[i][1]>Correlation_Measures[i][1].quantile(.80)[0]].dropna() #Calculate top 20% Magnitude Surprise returns for given returns 
        Correlation_Surprise_= Correlation_Measures[i][0]
        Magnitude_Surprise= Correlation_Measures[i][1]
        
        Top_20_Percentile_Magnitude_Surprise['CS']= Correlation_Measures[i][0]
        MS_20_CS_less_1= Top_20_Percentile_Magnitude_Surprise[Top_20_Percentile_Magnitude_Surprise['CS']<=1]
        MS_20_CS_greater_1= Top_20_Percentile_Magnitude_Surprise[Top_20_Percentile_Magnitude_Surprise['CS']>1]
        Top_20_Percentile_Magnitude_Surprise=Top_20_Percentile_Magnitude_Surprise.drop('CS',1)
        MS_20_CS_less_1=MS_20_CS_less_1.drop('CS',1)
        MS_20_CS_greater_1=MS_20_CS_greater_1.drop('CS',1)
            
    
    
        Next_day_MagSur= pd.DataFrame()
        MagSur_20= Top_20_Percentile_Magnitude_Surprise
        for j in range(len(MagSur_20)): 
            x= MagSur_20.index[j]
            for l in range(len(Magnitude_Surprise)):
                if x== Magnitude_Surprise.index[l]:
                    y= Magnitude_Surprise[l+1:l+2]
                    Next_day_MagSur= Next_day_MagSur.append(y)
    
        Average_Next_day_MagSur= (Next_day_MagSur.mean())[0]
        Exhibit5_returns[i].append(Average_Next_day_MagSur)
    
    #Next day magnitude surprise with Correlation Surprise greater than 1
        Next_day_MagSur_Greater_1= pd.DataFrame()
        
        MagSur_20_Greater_1= MS_20_CS_greater_1
        for j in range(len(MagSur_20_Greater_1)):
            x= MagSur_20_Greater_1.index[j]
            for l in range(len(Magnitude_Surprise)):
                if x== Magnitude_Surprise.index[l]:
                    y= Magnitude_Surprise[l+1:l+2]
                    Next_day_MagSur_Greater_1= Next_day_MagSur_Greater_1.append(y)
    
        Average_Next_day_MagSur_Greater_1= Next_day_MagSur_Greater_1.mean()
        Exhibit5_returns[i].append(Average_Next_day_MagSur_Greater_1)
    #Next day magnitude surprise with Correlation Surprise less than 1
        Next_day_MagSur_Less_1= pd.DataFrame()
        MagSur_20_Less_1= MS_20_CS_less_1
        for j in range(len(MagSur_20_Less_1)): 
            x= MagSur_20_Less_1.index[j]
            for l in range(len(MS_20_CS_less_1)):
                if x== MS_20_CS_less_1.index[l]:
                    y= MS_20_CS_less_1[l+1:j+2]
                    Next_day_MagSur_Less_1= Next_day_MagSur_Less_1.append(y)
    
        Average_Next_day_MagSur_Less_1= Next_day_MagSur_Less_1.mean()
        Exhibit5_returns[i].append(Average_Next_day_MagSur_Less_1)
                
    
    #Tbale: when importing different sets of data  
    Rows= ['MS 20%','MS 20% with CS<=1' ,'MS with CS>=1']
    Table_6= pd.DataFrame(index= Rows)                                         #Create open Table_2
    
    Table_6['US Equities']= Exhibit5_returns[0]                                #Append each Portfolio's Data to Table 2
    Table_6['European Equities']= Exhibit5_returns[1]
    Table_6['Currencies'] = Exhibit5_returns[2]
    
    
    return  Table_6
    #---------------------------------------------------------------------------
       

def Correlation_Surprise_Table_Exhbit7(): 
    
            
    return    
  
 
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------



#Journal Article: Kritzman et al. - 2011 - Principal Components as a Measure of Systemic Risk
#http://www.mas.gov.sg/~/media/resource/legislation_guidelines/insurance/notices/GICS_Methodology.pdf
def Absorption_Ratio(Returns, halflife):
    
    #problem with Absorption ratio is that it needs non-log return data. Once this is obtained it should take the exponential 250 day returns. After the log returns should be taken and then the 500day trailing window    
    
        #stage0: IMPORT LIBRARIES    
    import pandas as pd                                                        #import pandas    
    import numpy as np                                                         #import numpys  
    import math as mth                                                         #import math
    from sklearn.decomposition import PCA

        #stage1: GATHER DAILY TRAIL LENGTH
    
    time_series_of_500days=len(Returns)-int(500/12)                              #collect data that is outside of initial 500day window
    
        #stage2: GENERATE ABSORPTION RATIO DATA
    plotting_data=[]                                                           #create list titled plot data
    for i in range(time_series_of_500days):
        
                #stage1: CALCULATE EXPONENTIAL WEIGHTING
        window= Returns[i:i+int(500/12)]                                  #create 500 day trailing window      
        #centred_data= returns_500day.subtract(returns_500day.mean())       #Center Data
        
        pca = PCA(n_components= int(round(Returns.shape[1]*0.2)), whiten=False).fit(window)
        Eigenvalues= pca.explained_variance_       
        
                    #stage6: CALCULATE ABSORPTION RATIO DATA
        variance_of_ith_eigenvector=Eigenvalues.sum()

        #variance_of_ith_eigenvector= np.var(Eigenvectors,axis=1).sum()
        #variance_of_ith_eigenvector= ev_vectors.diagonal()#fetch variance of ith eigenvector
        variance_of_jth_asset= window.var().sum()                        #fetch variance of jth asset
    
            #stage7: CONSTRUCT ABSORPTION RATIO FORMULA     
        numerator= variance_of_ith_eigenvector                                 #calculate the sum to n of variance of ith eigenvector
        denominator= variance_of_jth_asset                                     #calculate the sum to n of variance of jth asset
               
        Absorption_Ratio= numerator/denominator                                #calculate Absorption ratio
    
            #stage8: Append Data
        plotting_data.append(Absorption_Ratio)                                 #Append Absorption Ratio iterations into plotting_data list
        
    
         #stage9: Plot Data
    plot_array= np.array(plotting_data)                                        #convert plotting_data into array
    dates= Returns[int(500/12):time_series_of_500days+int(500/12)].index                  #gather dates index over 500 day window iterations
    Absorption_Ratio_daily=pd.DataFrame(plot_array,index=dates,columns=list('R'))#merge dates and Absorption ratio returns
    Absorption_Ratio_daily= pd.ewma(Absorption_Ratio_daily, halflife=halflife)
    #Absorption_Ratio=Absorption_Ratio_daily.resample('M', how=None)#group daily data into monthly data
    
    return  Absorption_Ratio_daily #, Eigenvectors                                                  #print Absorption Ratio

def Absorption_Ratio_VS_MSCI_Graph(MSCI, AR_returns):
    
    import matplotlib.pyplot as plt    
    
    fig=plt.figure(figsize=(10,5))
    
    ax1= fig.add_subplot(2,1,1, axisbg='white')
    plt.suptitle('Absorption Ratio vs US Stock Prices')   
    plt.xticks(rotation=50)
    plt.xlabel('Year')#label x axis Year
    ax1.set_ylabel('MSCI USA Price', color='b')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,MSCI.max()[0]*1.10))
    ax1.plot(MSCI.index[500:3152],MSCI.values[500:3152])
    
    
    ax2= ax1.twinx()
    plt.ylabel('Index')#label y axis Index
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,1.2))
    ax2.plot(AR_returns.index,AR_returns.values, 'g')
    ax2.set_ylabel('Absorption Ratio Index', color='g')

    plt.show()
    fig.savefig('Absorption Ratio_vs_US_Stock_Prices.png')
    
    return 


def Absorption_Ratio_Standardised_Shift(AR_Returns):    
    
    import pandas as pd
           
    AR_15DAY= pd.ewma(AR_Returns, span=15)
    AR_Yearly= pd.ewma(AR_Returns, span=253)
    AR_Variance= AR_Yearly.std()
    
    delta_AR= (AR_15DAY-AR_Yearly)/AR_Variance
    
   
    return delta_AR

def Absorption_Ratio_and_Drawdowns(delta_AR):    #how to measure all drawdowns
    prevmaxi = 0
    prevmini = 0
    maxi = 0

    for i in range(len(delta_AR))[1:]:
        if delta_AR['R'][i] >= delta_AR['R'][maxi]:
            maxi = i
        else:
      # You can only determine the largest drawdown on a downward price!
          if (delta_AR['R'][maxi] - delta_AR['R'][i]) > (delta_AR['R'][prevmaxi] - delta_AR['R'][prevmini]):
              prevmaxi = maxi
              prevmini = i
    return (delta_AR['R'][prevmaxi], delta_AR['R'][prevmini])


def plot_AR(AR, figsize, yaxis):
    
    import matplotlib.pyplot as plt
    
    
    plt.figure( figsize=(figsize))    
    plt.suptitle(['Absorption Ratio Index from',' Daily Returns'],fontsize=12) 
    plt.xticks(rotation=50)
    plt.xlabel('Year')
    plt.ylabel('Index')
    
    y1, y2 = yaxis
    plt.ylim([y1, y2])    
    
    #x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,0.5,1))
    AR= AR
    x=AR.index
    y=AR.values
    #plt.plot(x,y, linewidth=2.5, color='k')
    plt.bar(x,y, width=0.2,color='w', label='Quiet')
    plt.grid()
    #cannot seem to find out how to colour this?
    

    plt.show()

    return 

# Modified plot_AR-version for visualization with Bokeh
def plot_AR_bokeh(AR, figsize, yaxis):
    from bokeh.plotting import figure, output_file, show #, VBox
    
    # Get 'R'-List out    
    AR = AR['R']
    
    # Export data for sphinx/bokeh
    AR.to_pickle('ari.pickle')
    
    """ From this point not necessary any more if combined with sphinx """

    # Define graph
    gra = figure(title="Absorption Ratio Index",
                 x_axis_label="Year",
                 y_axis_label="Index",
                 x_axis_type="datetime",
                 y_range=[0.84, 0.9],
                 plot_width=900,
                 plot_height=300,
                 logo=None)
    
    # Set the footer labels (including zoomed-state)
    gra.below[0].formatter.formats = dict(years=['%Y'],
                                          months=['%b %Y'],
                                          days=['%d %b %Y'])
    
    # Paint graph
    gra.segment(x0=AR.index,
                y0=0,
                x1=AR.index,
                y1=AR.values,
                color='#000000')

    #output_file("ari.html", title="Absorption Ratio Index")
    #show(VBox(gra))
   
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------

# Modified plot_AR-version for visualization with Bokeh
def plot_probit_bokeh(PF, figsize, yaxis):
    from bokeh.plotting import figure, output_file, show #, VBox
    
    # Flatten values to one dimension - has to be done after reading
    #PF.values = PF.values.flatten()
    
    # Export data for sphinx/bokeh
    PF.to_pickle('pf.pickle')
    
    """ From this point not necessary any more if combined with sphinx """

    from bokeh.plotting import figure, output_file, show #, VBox
    gra = figure(title="Probit Forecast",
              x_axis_label="Year",
              y_axis_label="Index",
              x_axis_type="datetime",
              plot_width=900,
              plot_height=300,
              logo=None)

    # Set the footer labels (including zoomed-state)
    gra.below[0].formatter.formats = dict(years=['%Y'],
                                       months=['%b %Y'],
                                       days=['%d %b %Y'])

    # Paint graph
    gra.segment(x0=PF.index,
             y0=0,
             x1=PF.index,
             y1=PF.values.flatten(),
             color='#000000')

    #output_file("pf.html", title="Probit Forecast")
    #show(VBox(gra))
   
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------

   
def plot_AR_ALL(US, UK, JPN, halflife):
        
    import matplotlib.pyplot as plt
    
    US_input= Absorption_Ratio(Returns= US, halflife=halflife)
    UK_input =Absorption_Ratio(Returns= UK, halflife=halflife)
    JPN_input =Absorption_Ratio(Returns= JPN, halflife=halflife)  
    
    plt.figure(figsize=(10,3))
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.5,1))
    plt.xlabel('Year')
    plt.ylabel('Absorption Ratio')
    plt.plot(US_input.index,US_input.values, label="US", linewidth=2, color = '0.2')
    plt.plot(UK_input.index, UK_input.values, label="UK", linewidth=3, linestyle='--', color = '0.1')
    plt.plot(JPN_input.index, JPN_input.values, label="JPN", linewidth=4, linestyle='-.', color = '0.05')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
    plt.grid()
    
    
    
    plt.show()
    
    return

#---------------------------------------------------------------------------
def Probit(Input_Returns,recession_data):
    import numpy as np
    import pandas as pd
    
            #Build Probit Dataframe
    Intial_window_Input=Input_Returns
    df=pd.DataFrame(index=MahalanobisDist(Returns=Intial_window_Input).index)  #will need to consider pushing this forward 500days due to 500AR window
    
    df['MD']=MahalanobisDist(Returns=Intial_window_Input[(17):])  
    Mag_Corr= Correlation_Surprise(Returns=Intial_window_Input[(17):])
    df['Mag_Corr']= Mag_Corr[1]/Mag_Corr[0]
    df['AR']=Absorption_Ratio(Returns= Intial_window_Input, halflife=int(8.5))
    df=df[int(17):] # Due to Absorption Ratio requiring 500day window
    df['Binary']=recession_data
                #A value of one is a recession period and a value of zero is an expandsion period
    #-----------------------------
    
    #writer = pd.ExcelWriter('output.xlsx')
    #df.to_excel(writer,'Sheet1')
    #writer.save()
    
    # AR sometimes empty?
    df = df[df['AR'] > 0]
    
    
    #Run Probit
    endog = df[['Binary']]      # Dependent
    exog = df[['MD','Mag_Corr','AR']]  #Independent
    
    const = pd.Series(np.ones(exog.shape[0]), index=endog.index)
    const.name = 'Const'
    exog = pd.DataFrame([const, exog.MD, exog.Mag_Corr, exog.AR]).T
    
    # Estimate the model
    import statsmodels.api as sm
    mod = sm.Probit(endog, exog)
    fit = mod.fit(disp=0)
    params = fit.params
    
    return params, df
#-----------------------------






"""
def AR_systemic_importance(AR):
    
    import numpy as np 
    
    Absorption_Ratio= AR[0]
    Top_20_eigenvectors= np.transpose(AR[1])
    
   
    #Calculate Centrality
        #Absolute value of the exposure of the ith asset within the jth eigenvector
    absolute_value_of_the_exposure_of_the_ith_asset_within_jth_eigenvector= []
    for i in range(len(np.transpose(Top_20_eigenvectors))):
        Relative_weights=[]
        for j in range(len(Top_20_eigenvectors)):
            jth_eigen_vector= Top_20_eigenvectors[j]
            weight= jth_eigen_vector[i]/(jth_eigen_vector.sum())
            Relative_weights.append(weight)
        
        
        
        
        
        
        absolute_value_of_the_exposure_of_the_ith_asset_within_jth_eigenvector.append(np.abs(np.sum(Relative_weights)))
            
   
    

    return absolute_value_of_the_exposure_of_the_ith_asset_within_jth_eigenvector

def Exhbit_8(delta_AR):
    
    
    return   
"""


    
"""
def Exhbit_9(Treasury_bonds, MSCIUS_PRICES):
    
    T_returns= logreturns(Returns=Treasury_bonds)
    MSCI_returns= logreturns(Returns=MSCIUS_PRICES)
    T_returns['MSCI']= MSCI_returns

    
    Returns=[]
    for i in range(1, len(T_returns)):
        Portfolio= T_returns[0:i]*[0.5,0.5]
        AR_Ratio= Absorption_Ratio(Returns= Portfolio)
        delta_AR=Absorption_Ratio_Standardised_Shift(AR_Returns=AR_Ratio)
        
        #it is     @it is not  #it is the same
        if delta_AR['R'][i]<delta_AR[0:i].quantile(.68)[0] and delta_AR['R'][i]>delta_AR[0:i].quantile(.32)[0]:
            Returns.append(Portfolio[i:i+1])
        
        elif delta_AR['R'][i]>delta_AR[0:i].quantile(.68)[0]:
            Portfolio= T_returns[0:i]*[0,1]
            AR_Ratio= Absorption_Ratio(Returns= Portfolio)
            delta_AR= Absorption_Ratio_Standardised_Shift(AR_Returns=AR_Ratio)
            Returns.append(Portfolio[i])
        
        elif delta_AR['R'][i]<delta_AR[0:i].quantile(.32)[0]:
            Portfolio= T_returns[0:i]*[1,0]
            AR_Ratio= Absorption_Ratio(Returns= Portfolio)
            delta_AR= Absorption_Ratio_Standardised_Shift(AR_Returns=AR_Ratio)
            Returns.append(Portfolio[i])
        
        

            
            
            
            
            
            
            
            
    
    #days with 
    AR_greater_one_std = delta_AR[delta_AR>delta_AR.quantile(.68)].dropna()
    
    
    #days with AR<-1o
    AR_less_one_std= delta_AR[delta_AR<delta_AR.quantile(.32)].dropna()
    
    
       
    return AR_greater_one_std, AR_less_one_std
    
    


#Plotting Systemic Risk Measures
def print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES):
    
   import matplotlib.pyplot as plt
    
   #1 MahalanobisDistances
   #1 MahalanobisDistance
   plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
   plt.xlabel('Year')                                                          #label x axis Year
   plt.ylabel('Index')                                                         #label y axis Index
   plt.suptitle('Mahalanobis Distance Index')                                  #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
   plt.bar(systemicRiskMeasure[0][0].index,systemicRiskMeasure[0][0].values, width=20,color='w', label='Quiet')#graph bar chart of Mahalanobis Distance
   plt.bar(systemicRiskMeasure[0][2].index,systemicRiskMeasure[0][2].values, width=20,color='k',alpha=0.8, label='Turbulent')
   plt.legend()
   plt.show()
 
   
   
   #2Correlation Surprise
#   Correlation_Surprise=systemicRiskMeasure[1][0]                              #gather Correlation surprise array
#   Magnitude_Surprise= systemicRiskMeasure[1][1]                               #gather turbulence score array
   
        #Magnitude Suprise   
  # plt.xticks(rotation=50)                                                    #rotate x axis labels 50 degrees
  # plt.xlabel('Year')                                                         #label x axis Year
  # plt.ylabel('Index')                                                        #label y axis Index
  # plt.suptitle('Magnitude Surprise Index')                                   #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
  # plt.bar(Magnitude_Surprise.index,Magnitude_Surprise.values, width=20)      #graph bar chart of Mahalanobis Distance
  # plt.show()
   
       #Correlation_Surprise
   #need to find weighted averaged return
#   plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
#   plt.xlabel('Year')                                                          #label x axis Year
#   plt.ylabel('Index')                                                         #label y axis Index
#   plt.suptitle('Correlation Surprise Index')                                  #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
#   plt.bar(Correlation_Surprise.index,Correlation_Surprise.values, width=2)     #graph bar chart of Mahalanobis Distance
#   plt.show()
   
   
   
   #3Absorption Ratio
   
#   fig=plt.figure()
    
#   ax1= fig.add_subplot(2,1,1, axisbg='white')
#   plt.suptitle('Absorption Ratio vs US Stock Prices')   
#   plt.xticks(rotation=50)
#   plt.xlabel('Year')#label x axis Year
#   ax1.set_ylabel('MSCI USA Price', color='b')
#   x1,x2,y1,y2 = plt.axis()
#   plt.axis((x1,x2,0,1600))
#   ax1.plot(MSCIUS_PRICES.index[500:3152],MSCIUS_PRICES.values[500:3152])

    
#   ax2= ax1.twinx()
   #plt.ylabel('Index')#label y axis Index
#   x1,x2,y1,y2 = plt.axis()
#   plt.axis((x1,x2,0,2))
#   ax2.plot(systemicRiskMeasure[2].index,systemicRiskMeasure[2].values, 'g')
#   ax2.set_ylabel('Absorption Ratio Index', color='g')

#   plt.show()
   
   
   
   

   
   
   
   
   
   
   
   
   #plt.xticks(rotation=50)  #rotate x axis labels 50 degrees
   #plt.xlabel('Year')#label x axis Year
   #plt.ylabel('Index')#label y axis Index
   #plt.suptitle('Absorption Ratio Index Calculated from Monthly Retuns of Yahoo Finance World Indices')#label title of graph Absorption Ratio
   #plt.plot(systemicRiskMeasure[2].index,systemicRiskMeasure[2].values)#graph line chart of Absorption Ratio
   #plt.show()
"""
 
 
 
 
 
 
 
 
"""
#stage2: CALCULATE COVARIANCE MATRIX
return_covariance= centred_data.cov()                                  #Generate Covariance Matrix over 500 day window
 
#stage3: CALCULATE EIGENVECTORS AND EIGENVALUES
ev_values,ev_vector= np.linalg.eig(return_covariance)                  #generate eigenvalues and vectors over 500 day window 
  
#Stage4: SORT EIGENVECTORS RESPECTIVE TO THEIR EIGENVALUES
ev_values_sort_high_to_low = ev_values.argsort()[::-1]                         
ev_values_sort=ev_values[ev_values_sort_high_to_low]                   #sort eigenvalues from highest to lowest
ev_vectors_sorted= ev_vector[:,ev_values_sort_high_to_low]             #sort eigenvectors corresponding to sorted eigenvalues
        
#Stage5: COLLECT 1/5 OF EIGENVALUES
shape= ev_vectors_sorted.shape[0]                                      #collect shape of ev_vector matrix
round_down_shape= mth.floor(shape*0.2)
#round_down_shape= mth.floor(shape*0.2) #round shape to lowest integer
ev_vectors= ev_vectors_sorted[:,0:round_down_shape]                    #collect 1/5th the number of assets in sample
"""

 
