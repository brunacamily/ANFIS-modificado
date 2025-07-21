import numpy
import math
import statistics

def produto (x):
    result = 1
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

#========= MEDIAS ===========

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