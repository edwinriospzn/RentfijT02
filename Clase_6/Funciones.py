import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import mysql.connector
import itertools
import scipy.optimize as opt
    
class DB_class:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def open_db(self):
        class2_db = mysql.connector.connect(host=self.host, user=self.user, 
                        passwd=self.password, database=self.database, port = 3306)
        return class2_db

def coupon_m(parametros):
    """
    coupon_m: Calcula la matriz de cupones para titulos Act/365, FS, cupon anual
    :parametros: entrar como arrays
    """
    val_date = np.array(parametros['val_date'])
    ini_date = np.array(parametros['ini_date'])
    fin_date = np.array(parametros['fin_date'])
    coupon = np.array(parametros['coupon'])

    cpn_complete = [[0]]*coupon.size
    days_complete = [[0]]*coupon.size
    for i in range(0,coupon.size):
        ## Calcular las fechas de cupones
        maturity = dt.datetime.strptime(str(fin_date[i]),'%Y-%m-%d').date()
        valuation = dt.datetime.strptime(str(val_date),'%Y-%m-%d').date()
        issue = dt.datetime.strptime(str(ini_date[i]),'%Y-%m-%d').date()
        days_diff = maturity-issue
        flow = days_diff.days/365+2 #Se suma 2 para garantizar
                                    # que se generan todos los flujos requeridos.
        coupon_dates = np.array([issue + relativedelta(years=+x) for x in range(int(flow))])
        #print(coupon_dates.shape)
        coupon_dates[coupon_dates>maturity] = maturity
        coupon_dates = np.unique(coupon_dates)
        future_coupon_dates = coupon_dates[coupon_dates>valuation]
        future_coupon_dates = np.sort(future_coupon_dates)
        try:
            prev_coupon = coupon_dates[coupon_dates<=valuation][-1]
        except:
            prev_coupon = issue
        #Calcular el delta de los cupones
        coupon_dates_adj = coupon_dates[coupon_dates>=prev_coupon]
        coupon_days_diff = [np.diff(coupon_dates_adj)[i].days for i in\
                            range(0,coupon_dates_adj.size-1)]
        delta = np.array([coupon_days_diff[-future_coupon_dates.size:][i]/365 \
                          for i in range(0,future_coupon_dates.size)])
    
        coupon_pays = np.array([(1+coupon[i]/100)**delta[x]-1 for x in range(0,delta.size)])
        ## nominal o nocional o principal
        coupon_pays[-1] = coupon_pays[-1]+ 1.0
        #calcular los dias al descuento
        days_count = (future_coupon_dates-valuation)
        days_count = np.array([days_count[x].days for x in range(0,days_count.size)])
        ### Agregar informacion 
        cpn_complete[i] = coupon_pays
        days_complete[i] = days_count
    
    days_matrix = np.sort(np.unique(list(itertools.chain.from_iterable(days_complete))))
    instruments = coupon.size
    coupon_matrix = np.zeros((days_matrix.size,instruments))
    coupon_matrix =pd.DataFrame(coupon_matrix)
    #Construyendo matriz
    for j in range(0,coupon.size):
        coupon_matrix.iloc[np.in1d(days_matrix, days_complete[j]),j] = cpn_complete[j]
    coupon_matrix = np.array(coupon_matrix) 
    
    dict_out = ({
            'coupon_matrix':coupon_matrix,
            'days_matrix':days_matrix
            })
    return dict_out



def coupon_m6(parametros):
    """
    coupon_m: Calcula la matriz de cupones para titulos Act/365, FS, cupon anual
    :parametros: entrar como arrays
    """
    val_date = np.array(parametros['val_date'])
    ini_date = np.array(parametros['ini_date'])
    fin_date = np.array(parametros['fin_date'])
    coupon = np.array(parametros['coupon'])
    frec_pag = np.array(parametros['frec_pag']) ###added
    #print(fin_date)
    #print(coupon)
    #print(frec_pag)
    cpn_complete = [[0]]*coupon.size
    days_complete = [[0]]*coupon.size
    for i in range(0,coupon.size):
        ## Calcular las fechas de cupones
        maturity = dt.datetime.strptime(str(fin_date[i]),'%Y-%m-%d').date()
        valuation = dt.datetime.strptime(str(val_date),'%Y-%m-%d').date()
        issue = dt.datetime.strptime(str(ini_date[i]),'%Y-%m-%d').date()
        days_diff = maturity-issue
        
        flow = days_diff.days/(frec_pag[i]*(365/12))+2 #Se suma 2 para garantizar
                            # que se generan todos los flujos requeridos.
        #coupon_dates = np.array([issue + relativedelta(years=x,month=+x*frec_pag[i]) for x in range(int(flow))])
        coupon_dates= np.array([])
        for x in range(int(flow)):
            if x==0:
                years_0=0
            elif (frec_pag[i]*x)%12==0:
                years_0=(frec_pag[i]*x)/12
            months_0=(frec_pag[i]*x)%12
            respuesta=issue + relativedelta(years=years_0) +relativedelta(months=months_0) 
            coupon_dates = np.append(coupon_dates, respuesta)   
        print(coupon_dates.shape)
        coupon_dates[coupon_dates>maturity] = maturity
        coupon_dates = np.unique(coupon_dates)
        future_coupon_dates = coupon_dates[coupon_dates>valuation]
        future_coupon_dates = np.sort(future_coupon_dates)
        try:
            prev_coupon = coupon_dates[coupon_dates<=valuation][-1]
        except:
            prev_coupon = issue
        #Calcular el delta de los cupones
        coupon_dates_adj = coupon_dates[coupon_dates>=prev_coupon]
        coupon_days_diff = [np.diff(coupon_dates_adj)[i].days for i in\
                            range(0,coupon_dates_adj.size-1)]
        delta = np.array([coupon_days_diff[-future_coupon_dates.size:][i]/365 \
                          for i in range(0,future_coupon_dates.size)])
    
        coupon_pays = np.array([(1+coupon[i]/100)**delta[x]-1 for x in range(0,delta.size)])
        ## nominal o nocional o principal
        coupon_pays[-1] = coupon_pays[-1]+ 1.0
        #calcular los dias al descuento
        days_count = (future_coupon_dates-valuation)
        days_count = np.array([days_count[x].days for x in range(0,days_count.size)])
        ### Agregar informacion 
        cpn_complete[i] = coupon_pays
        days_complete[i] = days_count
    
    days_matrix = np.sort(np.unique(list(itertools.chain.from_iterable(days_complete))))
    instruments = coupon.size
    coupon_matrix = np.zeros((days_matrix.size,instruments))
    coupon_matrix =pd.DataFrame(coupon_matrix)
    #Construyendo matriz
    for j in range(0,coupon.size):
        coupon_matrix.iloc[np.in1d(days_matrix, days_complete[j]),j] = cpn_complete[j]
    coupon_matrix = np.array(coupon_matrix) 
    
    dict_out = ({
            'coupon_matrix':coupon_matrix,
            'days_matrix':days_matrix
            })
    return dict_out


def discount_factor_m(parametros):
    discount_rates = parametros['discount_rates']
    delta_t = parametros['delta_t']
    d_r = np.array(discount_rates)
    d_r = d_r.reshape(d_r.size,1)
    disc = (1/(1+d_r/100)**delta_t).T
    return disc


def dirty_price_m(parametros):
    disc_factors = discount_factor_m(parametros)
    coupon_dict = parametros['cpn_dict']
    cpn_m = coupon_dict['coupon_matrix']
    dirty_prices =  np.multiply(disc_factors,cpn_m).sum(axis=0)*100
    return np.round(dirty_prices,3)


#### Unidimensional con loop

def dirty_price_1i(parametros,inst_pos):
    """
    Calculo de precio sucio para un titulo en la ubicacion inst_pos 
        de la matriz de cupones
    :param parametros: Diccionario de parametros
    :param inst_pos: Posicion del titulo en la matriz de cupones (columna)
    """
    ytm_1d = np.array(parametros['discount_rates'])
    coupon_dict = parametros['cpn_dict']
    cpn_m = coupon_dict['coupon_matrix']
    days_m = coupon_dict['days_matrix']
    
    delta_t = days_m/365    
    inst_coupon = cpn_m[:,inst_pos]
    delta_t = np.array(delta_t).reshape(len(delta_t),1)
    ytm = ytm_1d.reshape(ytm_1d.size,1)
    # Calculando curva  
    discount_rate = 1/(1+ytm/100)**delta_t
    # Calculando precios con TIR de los titulos
    dirty_price_1d = np.matmul(discount_rate.T,inst_coupon)*100    
    return np.round(dirty_price_1d,3)

def ytm_1d_obj_func(x,i,params,prices):
    """
    Funcion objetivo multidimensional de calculo de TIR
    :param x: (array) TIRs objetivo.
    :param i: Posicion del titulo.
    :param parametros: Diccionario de parametros
    :param precios: (array) precios de los titulos.
    """
    params['discount_rates'] = x
    #diff= (prices[i] - dirty_price_1i(params,i)[0])**2
    diff= (prices[i] - dirty_price_1i(params,i)[0])*1000
    return(diff)

        
def find_ytm_loop(parametros,prices):
    """
    Encontrar las TIRs de multiples titulos a traves de un loop con scipy.optimize.brentq
    :param parametros: Diccionario de parametros
    :param precios: (array) precios de los titulos.
    :return: (array) TIRs de los titulos.
    """
    inst = len(parametros['coupon'])
    up_limit = parametros['limits'][1]
    dw_limit = parametros['limits'][0]
    #up_limit = 1
    #dw_limit = -1
    params = parametros.copy()
    ## Escritura
    ytm_total = np.zeros(inst)
    for i in range(0,len(prices)):    
        if np.int(np.sign(ytm_1d_obj_func(dw_limit,i,params,prices))*np.sign(ytm_1d_obj_func(up_limit,i,params,prices))) ==1:
            continue
        ytm_total[i] = opt.brentq(ytm_1d_obj_func, dw_limit,up_limit,args = (i, params,prices))
    return ytm_total

### multidimensional con root

def ytm_obj_func(x,params,prices):
    """
    Funcion objetivo multidimensional de calculo de TIR
    :param x: (array) TIRs objetivo.
    :param parametros: Diccionario de parametros
    """
    params['discount_rates'] = x
    diff= prices.reshape(prices.size,) - dirty_price_m(params)
    #diff= (prices.reshape(prices.size,) - dirty_price_m(params))*1000
    return(diff)

def find_ytm_m(parametros,prices):
    """
    Encontrar las TIRs de multiples titulos simultaneamente con scipy.optimize.root.
    :param parametros: Diccionario de parametros
    :param precios: (array) precios de los titulos.
    :return: (array) TIRs de los titulos.
    """
    inst = len(parametros['coupon'])
    init_cond = np.array([0.0]*inst)
    params = parametros.copy()
    result = opt.root(ytm_obj_func,init_cond,args = (params,prices)) ## Usar el algoritmo root
    ytm_total = result.x
    return ytm_total

##############3 Clase 5: Calculo de N&S
    

def NelsonSiegel_1d(params, ven):
    """
    Calcular la curva de tasas de interes con el modelo de Nelson & Siegel, 
    para un vector de parametros.
    :params: vector de parametros. Array de 4 valores.
    :ven: Dias anualizados. 
    :return: Curva tasas de interes. 
    """
    try:
        y = params[0]+params[1]*((1-np.exp(-ven/params[3]))/(ven/params[3]))+\
                      params[2]*((1-np.exp(-ven/params[3]))/(ven/params[3])- \
                            np.exp(-ven/params[3]))
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion NelsonSiegel_1d')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion NelsonSiegel_1d')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion NelsonSiegel_1d')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion NelsonSiegel_1d')
        print(e)
        raise Exception 
    return y.T


def NelsonSiegel_m(params, ven):
    """
    Calcular la curva de tasas de interes con el modelo de Nelson & Siegel, 
    para una matriz de parametros.
    :params: vector de parametros. Array de 4 valores.
    :ven: Dias anualizados. 
    :return: Matriz de curvas de tasas de interes. 
    """
    try:
        ven = ven.reshape(ven.size,1)
        params_len = int(params.size/4) #(p,r)
        ven_len = ven.shape[0] #(v,1)
        beta_0 = np.tile(params[0],ven_len).reshape(ven_len,params_len)
        y = beta_0+params[1]*((1-np.exp(-ven/params[3]))/(ven/params[3]))+\
                    params[2]*((1-np.exp(-ven/params[3]))/(ven/params[3])- \
                            np.exp(-ven/params[3]))
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion NelsonSiegel_m')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion NelsonSiegel_m')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion NelsonSiegel_m')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion NelsonSiegel_m')
        print(e)
        raise Exception 
    return y



def dirty_price_m_ns(betas,parametros):
    """
    Calcular el precio sucio de varios titulos por medio de un descuento con la curva
        generada por el modelo de Nelson & Siegel
    :param betas: (array) parametros del modelo
    :param parametros: Diccionario de parametros
    :return: precios
    """
    try:
        coupon_dict = parametros['cpn_dict']
        cpn_m = coupon_dict['coupon_matrix']
        venc = coupon_dict['days_matrix']/365
        venc = venc.reshape(1,venc.size)
        model_curve = NelsonSiegel_m(betas, venc)
        disc_factors = np.exp(np.multiply(-model_curve.T, venc)).T
        # Calculando precios nelson siegel de los titulos
        dirty_prices = np.matmul(cpn_m.T,disc_factors)
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion dirty_price_m_ns')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion dirty_price_m_ns')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion dirty_price_m_ns')
        print(e)
        raise Exception 
    return np.round(dirty_prices*100,3)


def objetive_function(betas,parametros):
    """
    Suma de los errores cuadraticos entre las TIRs de mercado y las TIRs provenientes
        del descuento con una curva generada por el modelo de Nelson & Siegel
    :param betas: (array) parametros del modelo
    :param parametros: Diccionario de parametros
    :return: (float) Diferencia
    """
    try:
        tires_merc = np.array(parametros['discount_rates'])
        prices = dirty_price_m_ns(betas,parametros)
        #ns_tires = find_ytm_loop(parametros,prices) #calculo desde funciones 
        ns_tires = find_ytm_m(parametros,prices) #calculo desde funciones 
        diff = ((tires_merc-ns_tires)**2).sum()            
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion dirty_price_m_ns')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion dirty_price_m_ns')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion dirty_price_m_ns')
        print(e)
        raise Exception                                          
    return diff

def betas_optimization(parametros):
    """
    Optimizacion por TIRs con un descuento con el modelo de Nelson & Siegel, a traves de
        scipy.optimize.minimize.
    :return: Diccionario con restultados del optimizador
    """
    try:
        initial_ = np.array([ 1, 2,  3.9,  5   ])
        #initial_ = np.array([ 0.07962154, -0.0653882,  0.07219587,  10   ])
        #optimiz_result = opt.minimize(objetive_function,initial_,args=(parametros),method='L-BFGS-B')
        optimiz_result = opt.minimize(objetive_function,initial_,args=(parametros),method='Nelder-Mead', tol=1e-6)
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion betas_optimization')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion betas_optimization')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion betas_optimization')
        print(e)
        raise Exception 
    return optimiz_result

def betas_optimization_bound(parametros): #con ytm_mult y condicion no cuadratica
    """
    Optimizacion por TIRs con un descuento con el modelo de Nelson & Siegel, a traves de
        scipy.optimize.brute.
    :return: Los parametros optimos del modelo
    """
    try:
        up_limit = parametros['limits'][1]
        dw_limit = parametros['limits'][0]
        bnds = [(dw_limit, up_limit)]*4
        optimiz_result = opt.brute(objetive_function,bnds,args=(parametros,))
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion betas_optimization_bound')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion betas_optimization_bound')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion betas_optimization_bound')
        print(e)
        raise Exception 
    return optimiz_result
    