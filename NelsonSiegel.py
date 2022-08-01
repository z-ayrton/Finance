"""
Estimacion de parametros Nelson & Siegel a partir de precios de mercado
y cashflows de titulos: Parametrizacion de Diebold & Li (2006)
"""
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from random import random

#inicia con estos valores
betas_opt=[0.2,0.2,0.1]

class Bond_FixedRate():
    """
    Bono de renta fija

    Parameters:
        price (opt): dataframe de precios spot (TICKER,PRICE)
        cf: list of tuples with cashflow (TIME TO PAYMENT, CASH FLOW)
        stock (opt): cantidad de bonos en mi portfolio
    """

    def __init__(self,cf,price=0.,stock=0.):
        self.price=price
        self.cf=cf
        self.yields=Bond_Yields(ns_lambda,self.cf)
        self.dcf=Bond_dcf(self.cf,self.yields)
        self.modelprice=sum(self.dcf)
        self.residual= self.price - self.modelprice
        self.stock=stock

    def get_cf(self):
        return self.cf

def Bond_Yields(ns_lambda,cf):

    yields=[]
    global betas_opt

    yields=[
        betas_opt[0] +
        betas_opt[1] * (1-np.exp(-t*ns_lambda)) / (t*ns_lambda) +
        betas_opt[2] * ( (1-np.exp(-t*ns_lambda)) / (t*ns_lambda) - (np.exp(-t*ns_lambda)) )
        for (t,p) in cf
        ]

    return yields

def Bond_dcf(cf,yields):

    cf_y=list(zip(cf,yields))

    dcf= [p*np.exp(-y*t) for ((t,p),y) in cf_y]

    return dcf

def NelsonSiegel(bonds):

    global betas_opt

    def ns_opt(betas):

        nonlocal bonds

        SSE=0.
        for bond in bonds:
                ac_bond=bond

                yields=[
                    betas[0] +
                    betas[1] * (1-np.exp(-t*ns_lambda)) / (t*ns_lambda) +
                    betas[2] * ( (1-np.exp(-t*ns_lambda)) / (t*ns_lambda) - (np.exp(-t*ns_lambda)) )
                    for (t,p) in ac_bond.get_cf()
                    ]

                ac_bond.yields=yields
                ac_bond.dcf=Bond_dcf(ac_bond.get_cf(),ac_bond.yields)

                ac_bond.modelprice=sum(ac_bond.dcf)

                ac_bond.residual= np.absolute(ac_bond.price - ac_bond.modelprice)

                SSE+=ac_bond.residual

        return SSE

    bt = sco.minimize(fun=ns_opt,x0=np.array(betas_opt),method='SLSQP',bounds=((0.,None),(0.,None),(0.,None)))

    betas_opt=bt.x

    return bt.x

def curva_tasas(betas,ns_lambda,maturities=[7/365,30/365,60/365,90/365,0.5,
                1,1.5,2,2.5,3,5,10,15,20,25,30]):

    data_yields={'Maturity':maturities, 'Yield':[
        betas[0] +
        betas[1] * (1-np.exp(-t*ns_lambda)) / (t*ns_lambda) +
        betas[2] * ( (1-np.exp(-t*ns_lambda)) / (t*ns_lambda) - (np.exp(-t*ns_lambda)) )
        for t in maturities
        ]}

    yields=pd.DataFrame(data_yields)
    yields.index=yields['Maturity']
    del yields['Maturity']

    return yields

def VaR(betas,ns_lambda,scenarios,portfolio,alpha=0.01):

    scenario_base= sum([sum(Bond_dcf(cf=Bond.cf, yields=Bond.yields)) * Bond.stock for Bond in portfolio])

    sim_temp={'Value':scenario_base,'PnL':0}
    simulation=pd.DataFrame(data=sim_temp,index=[0])

    for n in range (scenarios+1):
        sce_betas=[beta + norm.ppf(random())/100 for beta in betas]

        for Bond in portfolio:
            maturities= [x[0] for x in Bond.cf]
            sce_curva=curva_tasas(sce_betas,ns_lambda,maturities)

            scenario_n= sum(sum(Bond_dcf(cf=Bond.cf,yields=sce_curva.values) * Bond.stock))


            simulation.loc[len(simulation.index)]= [scenario_n, scenario_n-scenario_base]

    VaR_Valor= simulation.loc[:,'Value'].quantile(alpha)
    VaR_PnL= simulation.loc[:,'PnL'].quantile(alpha)

    print(f'VaR Valor: {VaR_Valor:.2f} __ VaR PnL: {VaR_PnL:.2f} ')


if __name__ == '__main__':

    ns_lambda = 0.0609 # Diebold & Li fijan lambda

    #Bonos en el mercado, para hacer curva de tasas:

    M_Bond1=Bond_FixedRate(price=99.5,cf=[(5/12,3.94216666666667),
                        (1,3.79380555555556),(1.5,3.92097222222222),
                        (2,3.85738888888889),(2.5,3.87858333333333),
                        (3,3.83619444444444),(3.5,4.),(4,104.)])

    M_Bond2=Bond_FixedRate(price=103.,cf=[(1.,10.),(2.,10.),(3.,10.),
                        (4.,60.),(5.,5.),(6.,5.),(7.,5.),(8.,55.)])

    M_Bond3=Bond_FixedRate(price=104.5,cf=[(0.5,5.),(1.,105.)])

    M_Bond4=Bond_FixedRate(price=99.,cf=[(0.5,100.)])

    M_Bond5=Bond_FixedRate(price=99.8,cf=[(1/12,100.)])

    bonds=[M_Bond1,M_Bond2,M_Bond3,M_Bond4,M_Bond5]
    betas=NelsonSiegel(bonds)

    print(f'Betas: {",".join(f"{beta:.2f}" for beta in betas)}')
    print()

    fig, (ax1)=plt.subplots(1,1)
    ax1.set_title('Curva de tasas')
    ax1.plot(curva_tasas(betas,ns_lambda),'o-k')
    ax1.set_ylabel('Yield')
    ax1.set_xlabel('Maturity (years)')

    #Bonos en mi portfolio, para calcular VaR:

    P_Bond1=Bond_FixedRate(stock=5,
                cf=[(0.5,5.),(1.5,5.),(1.5,5.),(2.,5.),(2.5,5.),(3.,105.)])

    portfolio=[P_Bond1]

    scenarios=1000

    VaR(betas,ns_lambda,scenarios,portfolio)




