import pandas_datareader as pdr
import pandas as pd
import numpy as np
from os import getcwd
import scipy.optimize as sco
import yfinance as yf

yf.pdr_override()

now=pd.Timestamp.today()
nametime=str(now.hour) + str(now.minute) + str(now.second)

#Dates to consider: YYYY-MM-DD
date_ini = '2022-06-05'
date_end = '2022-07-04'

#A textbox will be genearated to input a ticker.
#Write one by one pressing Intro after each one.
#When finished, just press Intro in an empty textbox
ticker=[]

while True:
    ele = input()
    if ele:
        ticker.append(ele)
    else:
        break

#data: prices downloaded from Yahoo (daily Adj Close)
data=pdr.data.get_data_yahoo(ticker, start=date_ini,  end=date_end)['Adj Close']
print(data)
data.replace('', np.nan,inplace=True)

#returns: returns calculated from data
returns = data.pct_change(fill_method='ffill')[1:]
returns.replace(0.,np.nan,inplace=True)

#dscpt= descriptive statistics from returns
ddscpt=data.describe()
rdscpt=returns.describe()

#corr: correlation matrix
corr=returns.corr()

#varcovar: variance and covariance matrix
varcovar=returns.cov()

#Weights
def shr_opt(weights,varcovar,dscpt,size):
        """
        Sub-funcion que retorna los pesos del portfolio con maximo
        sharpe ratio y minima varianza.
        """

        rf= 0.0175/252 #risk free diaria
        weights=np.array([weights])

        Yrets=dscpt.loc['mean']
        Yvar= varcovar

        def p_vol(weights):
            return np.sqrt((weights @ Yvar) @ weights)

        def inv_sr(weights):
            return - ((weights @ Yrets) - rf) / p_vol(weights)

        bounds=sco.Bounds(np.zeros(size),np.ones([size]))
        const=sco.LinearConstraint(np.ones(size),np.ones(1),np.ones(1))
        configs = { 'method':'trust-constr',
                    'jac':'2-point',
                    'hess': sco.BFGS()}

        shr = sco.minimize(inv_sr,np.full(size,1./size), bounds=bounds,
                            constraints=const,**configs)

        sdv = sco.minimize(p_vol,np.full(size,1./size), bounds=bounds,
                            constraints=const,**configs)

        return shr.x, sdv.x #weights optimos

size=varcovar.shape[1]

MSR_weights= pd.DataFrame(shr_opt(np.full(size,1./size),varcovar,rdscpt,size)[0])
MSR_weights.index=ticker
MSR_weights.columns = ['Weight']

MV_weights= pd.DataFrame(shr_opt(np.full(size,1./size),varcovar,rdscpt,size)[1])
MV_weights.index=ticker
MV_weights.columns = ['Weight']

#graph
#returns.plot()

#exporta excel
path = getcwd() + '\DataYahoo' + nametime + '.xlsx'
with pd.ExcelWriter(path) as excel_writer:
    data.to_excel(excel_writer,sheet_name='Adj Close')
    returns.to_excel(excel_writer,sheet_name='Returns')
    ddscpt.to_excel(excel_writer,sheet_name='Price_Stats')
    rdscpt.to_excel(excel_writer,sheet_name='Return_Stats')
    corr.to_excel(excel_writer,sheet_name='Correlation')
    varcovar.to_excel(excel_writer,sheet_name='Var Covar')
    MSR_weights.to_excel(excel_writer,sheet_name='Max SR')
    MV_weights.to_excel(excel_writer,sheet_name='Min Var')

print(f'Archivo finalizado : {path}')


