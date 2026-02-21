import numpy
import math
import statistics

def produto (x):
    result = 1 # Inicializamos o resultado como 1 (identidade da multiplicação)
    for valor in x:
        result *= valor  # Multiplicamos os valores dentro da lista

    return result
                    
def minimo (x):
    result = x[0]
    for valor in x[1:]:
        result = min(result, valor)  # Atualiza o resultado com o mínimo
    return result

def lukasiewicz (x):
    result = x[0]
    for valor in x[1:]:
        result = max(1e-12, result + valor - 1)
    return result

def produto_drastico (x):
    result = x[0]
    for valor in x[1:]:
        if result == 1.0:
            result = valor
        elif valor == 1.0:
            continue  # result permanece o mesmo
        else:
            result = 1e-12
    return result

def nilpotente_min(x): 
    result = x[0]
    for valor in x[1:]:
        if (result + valor) > 1:
            result = min(result, valor)
        else:
            result = 1e-12
    return result

# ========= t-normas com lambda ===========

def hamacher_prod (x, lambda_param): 
    if lambda_param >= 1500:
        result = produto_drastico(x)
    else: 
        result = x[0]
        for valor in x[1:]:
            if (lambda_param == result == valor) == 1e-12:
                result = 1e-12
            else:
                result = result * valor / ( lambda_param + (1 - lambda_param) * (result + valor - result * valor) )

    return result

def frank (x, lambda_param):      
    if lambda_param == 0:
        result = minimo(x)

    elif lambda_param == 1:
        result = produto(x)

    elif lambda_param >= 1050:
        result =  lukasiewicz(x)

    else: 
        result = x[0]
        for valor in x[1:]:
            numerador = ( lambda_param**result - 1 ) * ( lambda_param**valor - 1 )
            result = math.log(1 + (numerador / ( lambda_param - 1 ) ), lambda_param)

    return result

def sugeno_weber (x, lambda_param):
    if lambda_param == -1:
        result = produto_drastico(x)

    elif lambda_param == 10000:
        result = produto(x)

    else:
        result = x[0]
        for valor in x[1:]:
            numerador = result + valor - 1 + lambda_param * result * valor
            denominador = 1 + lambda_param
            result = max( 1e-12, numerador / denominador)
    return result

def yager (x, lambda_param):
    if lambda_param == 0:
        result = produto_drastico(x)
    
    elif lambda_param == 10000:
        result = minimo(x)
    
    else:
        result = x[0]
        for valor in x[1:]:
            temp = 1 - ( (1 - result)**lambda_param + (1 - valor)**lambda_param )**(1/lambda_param)
            result = max(1e-12, temp )
    return result

def dombi(x, lambda_param):
    if lambda_param == 0:
        result = produto_drastico(x)

    elif lambda_param == 10000:
        result = minimo(x)

    else:
        result = x[0]
        for valor in x[1:]:
            denominador = ( ((1-result)/result)**lambda_param + ((1-valor)/valor)**lambda_param )**(1/lambda_param)
            result = 1 / (1 + denominador)
    return result

def schweizer_skar (x, lambda_param):
    if lambda_param == -10000:
        result = minimo(x)

    elif lambda_param == 0:
        result = produto(x)
    
    elif lambda_param == 10000:
        result = produto_drastico(x)
        
    else:
        result = x[0]
        for valor in x[1:]:
            result = max(1e-12, ( result**lambda_param + valor**lambda_param - 1 ) )**(1/lambda_param)
    
    return result

# ========= prof pediu 06-08-25 que gerasse a t-norma AA ===========

def t_norma_AA(x, lambda_param):
    # A T-norma AA é definida como: e^-( (-log(x1))^lambda + (-log(x2))^lambda )^(1/lambda)
    
    if lambda_param == 0:
        result = produto_drastico(x)

    elif lambda_param == 1500:
        result = minimo(x)

    else:
        result = x[0]
        for valor in x[1:]:
            x1 = (-math.log(result))**lambda_param 
            x2 = (-math.log(valor))**lambda_param
            result = math.e **  (-( x1 + x2 )**(1/lambda_param))
    
    return result

#==================== MEDIAS ===================================

def media_aritmetic(x):
    n = len(x)
    result = x[0]
    for valor in x:
        result += valor
    result = result * (1/n)
    return result
    # m = 1/n * somatorio de x0 a xn

def media_geometrica(x):
    n = len(x)
    result = 1
    for valor in x:
        result *= valor
    result = result ** (1/n)
    return result
    # m = raiz N(ézima) do produtório de x (x0 a xn)

def media_harmonica(x):
    n = len(x)
    result = 0
    for valor in x:
        result += 1/valor
    result = n/result
    return result
    # m = n / somatorio de 1/xi, onde x vai de x0 a xn

def media_quadratica(x):
    n = len(x)
    result = 0
    for valor in x:
        result += valor ** 2
    result = (result/n) ** 1/2
    return result
    # m = raiz quadrada do somatorio de xi ao quadrado dividido por n, onde x vai de x0 a xn.

def mediana(x):
    result = statistics.median(x)
    return result

def moda(x): # não funciona !!!!!
    result = statistics.multimode(x)
    return result




## ======================== FUNÇÕES MISTAS =======================================


def mycin(x):
    if len(x) == 0:
        return None
    result = x[0]

    for valor in x[1:]:
        if min(result,valor) >= 0.5:
            result = 2 * (result + valor - result * valor) - 1

        elif (min(result,valor) < 0.5) and  (max(result, valor)) > 0.5:
            numerador = result + valor - 1
            denominador = 1 - min(abs(2 * result - 1), abs(2 * valor - 1))
 
            result = (numerador / denominador) + 0.5
            
        elif max(result, valor) <= 0.5:
            result = 2 * result * valor

    return result

def prospector(x): 
    result = x[0]
    for valor in x[1:]:
        result = (result + valor) / (result * valor + (1-result) * (1-valor))
    return result

def example_9_1 (x):
    result = x[0]
    for valor in x[1:]:
        result = max(0, min(1, result + valor - 0.1))  # rodar com  com 0,1 .. 0,5 ... 0,9

    return result

def example_9_5 (x):
    result = x[0]
    for valor in x[1:]:
        result = max(0, min(1, result + valor - 0.5))  # rodar com  com 0,1 .. 0,5 ... 0,9

    return result

def example_9_9 (x):
    result = x[0]
    for valor in x[1:]:
        result = max(0, min(1, result + valor - 0.9))  # rodar com  com 0,1 .. 0,5 ... 0,9

    return result

def example_10 (x):
    result = x[0]
    for valor in x[1:]:
        if (result * valor) > 0.5:
            result = result * valor
        elif (1 - result) * (1 - valor) > 0.5:
            result = result + valor - result * valor
        else:
            result = 0.5
    return result

def example_11 (x):
    result = x[0]
    for valor in x[1:]:
        result = 0.4*result + 0.4*valor + 0.2*result*valor
    return result

def example_12 (x):
    result = x[0]
    for valor in x[1:]:
        numerador = 2*result*valor * (result + valor - result*valor)
        result = numerador / (result + valor)
    return result

