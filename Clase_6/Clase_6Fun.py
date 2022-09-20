import numpy as np
import scipy.optimize as opt
import Clase_6.Funciones as funciones


def Macaulay_duration_mat(parametros,prices):
    """
    Calcula la duracion de Macaulay de manera matricial y simultanea para un
    numero de titulos
    :return: Vector con duracion de Macaulay para cada titulo
    """
    try:
        coupon_dict = parametros['cpn_dict']
        cpn_m = coupon_dict['coupon_matrix']
        delta_t = parametros['delta_t']
        delta_t = delta_t.reshape(delta_t.size,1)
        discount_factors = parametros['discount_factors']
        ### Calculo de los cupones ponderados
        cpn_pond = np.multiply(cpn_m,delta_t)
        price_pond = np.multiply(cpn_pond,discount_factors).sum(axis=0)
        mac_dur = price_pond/prices
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat')
        print(e)
        raise Exception 
    return mac_dur



def Modified_duration_mat(parametros,prices):
    """
    Calcula la duracion de modificada de manera matricial y simultanea para un
    numero de titulos, utilizando la funcion Macaulay_duration_mat 
    :return: Vector con duracion de modificada para cada titulo
    """
    try:
        discount_rates = np.array(parametros['discount_rates'])
        discount_tirs = 1/(1+discount_rates/100)
        #Calcularse con otra funcion
        mac_dur = Macaulay_duration_mat(parametros,prices)
        mod_dur = np.multiply(mac_dur,discount_tirs)
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Modified_duration_mat')
        print(e)
        raise Exception 
    return mod_dur


def Modified_duration_mat_2(parametros,prices,macaulay_duration):
    """
    Calcula la duracion de modificada de manera matricial y simultanea para un
    numero de titulos, utilizando el insumo de duracion de macaulay 
    :return: Vector con duracion de modificada para cada titulo
    """
    try:
        discount_rates = np.array(parametros['discount_rates'])
        discount_tirs = 1/(1+discount_rates/100)
        #Calcularse con otra funcion
        mod_dur = np.multiply(macaulay_duration,discount_tirs)
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_2')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_2')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_2')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_2')
        print(e)
        raise Exception 
    return mod_dur


def Convexity_mat(parametros,prices):
    """
    Calcula la convexidad de manera matricial y simultanea para un
    numero de titulos
    :return: Vector con convexidad para cada titulo
    """
    try:
        coupon_dict = parametros['cpn_dict']
        cpn_m = coupon_dict['coupon_matrix']
        delta_t = parametros['delta_t']
        delta_t = delta_t.reshape(delta_t.size,1)
        discount_rates = np.array(parametros['discount_rates'])
        discount_factors = parametros['discount_factors']
        ### Calculo de denominadores
        tir_discount = 1/(1+discount_rates/100)**2
        price_denom = 1/prices
        ### Calculo de los cupones ponderados
        time_pond = np.multiply(delta_t,delta_t+1)
        cpn_pond = np.multiply(cpn_m,time_pond)
        price_pond = np.multiply(cpn_pond,discount_factors).sum(axis=0)
        conv = price_denom*tir_discount*price_pond
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Convexity_mat')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Convexity_mat')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Convexity_mat')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Convexity_mat')
        print(e)
        raise Exception 
    return conv


####### Funciones matriciales con Loop
    

def Macaulay_duration_mat_loop(parametros,prices):
    """
    Calcula la duracion de Macaulay de manera vectorizada y a traves de un loop para un
    numero de titulos
    :return: Vector con duracion de Macaulay para cada titulo
    """
    try:
        coupon_dict = parametros['cpn_dict']
        cpn_m = coupon_dict['coupon_matrix']
        delta_t = parametros['delta_t']
        delta_t = delta_t.reshape(delta_t.size,1)
        discount_factors = parametros['discount_factors']
        mac_dur = []
        for j in range(len(prices)):
            ### Calculo de los cupones ponderados
            cpn_pond = np.multiply(cpn_m[:,j].reshape(delta_t.size,1),delta_t)
            price_pond = np.multiply(cpn_pond,discount_factors[:,j].reshape(cpn_pond.size,1)).sum(axis=0)
            mac_dur_j = price_pond[0]/prices[j]
            mac_dur.append(mac_dur_j)
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat_loop')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat_loop')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat_loop')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Macaulay_duration_mat_loop')
        print(e)
        raise Exception 
    return np.array(mac_dur)


def Modified_duration_mat_loop(parametros,prices):
    """
    Calcula la duracion de modificada de manera vectorizada  y a traves de un loop  para un
    numero de titulos, utilizando la funcion Macaulay_duration_mat_loop 
    :return: Vector con duracion de modificada para cada titulo
    """
    try:
        discount_rates = np.array(parametros['discount_rates'])
        discount_tirs = 1/(1+discount_rates/100)
        #Calcularse con otra funcion
        mac_dur = Macaulay_duration_mat_loop(parametros,prices)
        mod_dur = []
        for j in range(len(prices)):
            mod_dur_j = np.multiply(mac_dur[j],discount_tirs[j])
            mod_dur.append(mod_dur_j)
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop')
        print(e)
        raise Exception 
    return np.array(mod_dur)


def Modified_duration_mat_loop_2(parametros,prices,macaulay_duration):
    """
    Calcula la duracion de modificada de manera matricial y a traves de un loop para un
    numero de titulos, utilizando el insumo de duracion de macaulay 
    :return: Vector con duracion de modificada para cada titulo
    """
    try:
        discount_rates = np.array(parametros['discount_rates'])
        discount_tirs = 1/(1+discount_rates/100)
        #Calcularse con otra funcion
        mod_dur = []
        for j in range(len(prices)):
            mod_dur_j = np.multiply(macaulay_duration[j],discount_tirs[j])
            mod_dur.append(mod_dur_j)
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop_2')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop_2')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop_2')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Modified_duration_mat_loop_2')
        print(e)
        raise Exception 
    return np.array(mod_dur)


def Convexity_mat_loop(parametros,prices):
    """
    Calcula la convexidad de manera vectorizada y a traves de un loop para un
    numero de titulos
    :return: Vector con duracion de Macaulay para cada titulo
    """
    try:
        coupon_dict = parametros['cpn_dict']
        cpn_m = coupon_dict['coupon_matrix']
        delta_t = parametros['delta_t']
        delta_t = delta_t.reshape(delta_t.size,1)
        discount_rates = np.array(parametros['discount_rates'])
        discount_factors = parametros['discount_factors']
        ### Calculo de denominadores
        tir_discount = 1/(1+discount_rates/100)**2
        price_denom = 1/prices
        ### Calculo de los cupones ponderados
        time_pond = np.multiply(delta_t,delta_t+1)
        conv = []
        for j in range(len(prices)):
            cpn_pond = np.multiply(cpn_m[:,j].reshape(time_pond.size,1),time_pond)
            price_pond = np.multiply(cpn_pond,discount_factors[:,j].reshape(cpn_pond.size,1)).sum(axis=0)
            conv_j = price_denom[j]*tir_discount[j]*price_pond[0]
            conv.append(conv_j)
    except TypeError as ty:
        print('Hubo un error en el calculo de la funcion Convexity_mat_loop')
        print(ty)
        raise TypeError
    except AttributeError as t:
        print('Hubo un error en el calculo de la funcion Convexity_mat_loop')
        print(t)
        raise AttributeError
    except ValueError as v:
        print('Hubo un error en el calculo de la funcion Convexity_mat_loop')
        print(v)
        raise ValueError
    except Exception as e: 
        print('Hubo un error en el calculo de la funcion Convexity_mat_loop')
        print(e)
        raise Exception 
    return np.array(conv)


def objetive_function_prices(betas,parametros):
    """
    Suma de los errores cuadraticos entre los precios de mercado y los precios provenientes
        del descuento con una curva generada por el modelo de Nelson & Siegel
    :param betas: (array) parametros del modelo
    :param parametros: Diccionario de parametros
    :return: (float) Diferencia
    """
    try:
        mkt_prices = parametros['mkt_prices']
        mkt_prices = mkt_prices.reshape(mkt_prices.size,1)
        prices = funciones.dirty_price_m_ns(betas,parametros) 
        diff = ((mkt_prices-prices)**2).sum()            
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


def betas_optimization_bound_prices(parametros): 
    """
    Optimizacion por TIRs con un descuento con el modelo de Nelson & Siegel, a traves de
        scipy.optimize.brute.
    :param parametros: Diccionario de parametros
    :return: Los parametros optimos del modelo
    """
    try:
        up_limit = parametros['limits'][1]
        dw_limit = parametros['limits'][0]
        bnds = [(dw_limit, up_limit)]*4
        optimiz_result = opt.brute(objetive_function_prices,bnds,args=(parametros,))
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


