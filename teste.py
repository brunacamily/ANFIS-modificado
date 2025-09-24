def _mycin_bivariada(x, y):
    """
    Função auxiliar que aplica a regra MYCIN para apenas duas entradas.
    (Definição completa na resposta anterior)
    """
    if (x == 0 and y == 1) or (x == 1 and y == 0):
        return 0
    if max(x, y) <= 0.5:
        return 2 * x * y
    elif min(x, y) >= 0.5:
        return 2 * (x + y - x * y) - 1
    else:
        numerador = x + y - 1
        denominador = 1 - min(abs(2 * x - 1), abs(2 * y - 1))
        return (numerador / denominador) + 0.5

def mycin_(valores):
    """
    Implementa a função de agregação mista MYCIN para um vetor de valores.
    """
    if not valores:
        return None
    if len(valores) == 1:
        return valores[0]
    
    resultado_agregado = valores[0]

    for i in range(1, len(valores)):
        proximo_valor = valores[i]
        resultado_agregado = _mycin_bivariada(resultado_agregado, proximo_valor)
        print(f"  Após agregar {proximo_valor:.2f}, o resultado parcial é: {resultado_agregado:.4f}")
    return resultado_agregado

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
            
        else:
            result = 2 * result * valor
    return result

# --- Exemplo de Uso ---
vetor_x = [0.7, 0.8, 0.4]
print(f"Iniciando agregação para o vetor: {vetor_x}")
print(f"Valor inicial: {vetor_x[0]}")

resultado_final = mycin(vetor_x)

print(f"\nO resultado final da agregação é: {resultado_final:.4f}")