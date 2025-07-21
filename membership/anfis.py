# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 07:30:34 2014

@author: tim.meggs
"""
import itertools
import numpy as np
from membership import mfDerivs
import copy

import matplotlib.pyplot as plt
class ANFIS:
    """Class to implement an Adaptive Network Fuzzy Inference System: ANFIS"

    Attributes:
        X
        Y
        XLen
        memClass
        memFuncs
        memFuncsByVariable
        rules
        consequents
        errors
        memFuncsHomo
        trainingType


    """

    def __init__(self, X, Y, memFunction, cm2):
        self.X = np.array(copy.copy(X))       
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))    
        #print(f"\n num regra:{self.memFuncsByVariable} \nLista de regras    ========== \n{self.rules} \nFim lista de regras =========")
        
        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)
        self.errors = np.empty(0)
        self.memFuncsHomo = all(len(i)==len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'
        self.cm2 = cm2


    def plot_gaussians_with_iris(self, x_values):
        for input_var in range(len(self.memFuncs)):
            plt.figure(figsize=(10, 6))
            
            # Plota apenas as três primeiras funções gaussianas para a variável de entrada atual
            for mf_index, (mf_type, mf_params) in enumerate(self.memFuncs[input_var]):
                if mf_index < 3:  # Verifica se é uma das três primeiras funções
                    if mf_type == 'gaussmf':
                        # Certifique-se de que a função de pertinência está correta
                        y_values = self.memClass.funcDict[mf_type](x_values, **mf_params)
                        
                        # Verifique se y_values está retornando valores esperados
                        plt.plot(x_values, y_values, label=f'Função Gaussiana {mf_index+1} para X{input_var+1}')
            
            # Adiciona os pontos das amostras para o atributo correspondente do dataset Iris
            input_data_points = self.X[:, input_var]
            plt.scatter(input_data_points, np.zeros_like(input_data_points), color='red', label='Pontos do Dataset Iris')

            plt.title(f'Funções Gaussianas e Pontos do Dataset Iris para X{input_var+1}')
            plt.xlabel(f'X{input_var+1}')
            plt.ylabel('Pertinência')
            plt.ylim(-0.1, 1.1)  # Ajusta os limites do eixo y para melhor visualização
            plt.legend()
            plt.grid(True)
            plt.show()


    def printRulesParaDUAS_entradas(self, x_values, y_layerFive, y_pred, y_true, pert):
        print("\nRegras do ANFIS:")
        # Itera sobre cada linha de x_values
        for i, current_x in enumerate(x_values):
            sumNumerador = 0
            sumDenominador = 0
            
            ps = 0
            pr = 0
            x1 = current_x[0]
            x2 = current_x[1]
            pert0 = pert[i]

            conj = ['Pequeno', 'Médio', 'Grande']

            print(f"\nPara entrada_{i+1} = [{x1}, {x2}],  class_pred = {y_pred[i]} , class_true = [{y_true.iloc[i]}]:")

            # (r1.pretsr1 + r2.pertsr2 + r3.pertsr3 + r4.pertsr4) / (r1 + r2 + r3 + r4)   
            for j, rule in enumerate(self.rules):
                pert1 = pert0[j][0]
                pert2 = pert0[j][1]           
                
                # Exibir a parte SE (antecedentes)
                antecedents = []
                for k, mf_idx in enumerate(rule):
                    antecedents.append(f"X{k} é {conj[mf_idx]}")

                antecedents_str = " e ".join(antecedents)

                # Exibir a parte ENTÃO (consequentes)
                num_inputs = self.X.shape[1]
                start_idx = j * (num_inputs + 1)
                end_idx = start_idx + num_inputs

                # Pegando os valores numéricos corretamente para exibir
                consequents = self.consequents[start_idx:end_idx].flatten()
                #
                p = consequents[0]
                q = consequents[1]
                
                bias = self.consequents[end_idx]  # Termo constante (bias)

                # Corrigir formatação para números individuais
                consequent_str = " + ".join([f"{float(consequents[k])}*x{k}" for k in range(len(consequents))])
                consequent_str += f" + {float(bias)}"  # Termo constante (bias)
                
                regra = x1 * p + x2 * q + bias
                multRegraPert = regra * (pert1 * pert2)         # pert1*pert2 = w[correspondente]
                sumNumerador += multRegraPert
                sumDenominador += (pert1 * pert2)       
            
                # Printar a regra formatada
                print(f"Regra {j+1}: Se ({antecedents_str}), então ({consequent_str}) = {regra}\n(pert_1: {pert1}, pert_2: {pert2})\nPeso da regra = pret_1 * pert_2 = {pert1*pert2}\tRegra*Peso da Regra = {regra*(pert1*pert2)}\n")
                ps += pert1 * pert2
                pr += regra * (pert1 * pert2)
                # regra = x0*p + x1*q + b
                
                 
            print(f"( Somatório das regras*peso:{pr} ) / ( Somatório dos pesos das regras:[{ps}] ) = {pr/ps}")
            print(f"Takage-Sugeno: {sumNumerador/sumDenominador} , layerfive: {y_layerFive[i]}")
        print()
        
#-----------------------------------------------------------------------------------------------------

    def printRules(self, x_values, y_layerFive, y_pred, y_true, pert):
        print("\nRegras do ANFIS:")
        contagem = 0
        tksg = [x_values[0], x_values[1], x_values[2]]
        
        # Itera sobre cada linha de x_values
        for i, current_x in enumerate(x_values):
            sumNumerador = 0
            sumDenominador = 0
            
            ps = 0
            pr = 0
            x1 = current_x[0]
            x2 = current_x[1]
            x3 = current_x[2]
            x4 = current_x[3]
            print('x1,x2,x3,x4')
            print(x1,x2,x3,x4)
            pert0 = pert[i]
            
            # duas funções de pertinencia
            conj = ['Pequeno', 'Grande']
            input = ['SL', 'SW', 'PL', 'PW']

            print(f"\nPara entrada_{i+1} = [{input[0]}: {x1}, {input[1]}: {x2}, {input[2]}: {x3}, {input[3]}: {x4}],  class_pred = {y_pred[i]} , class_true = [{y_true.iloc[i]}]:")

            # (r1.pretsr1 + r2.pertsr2 + r3.pertsr3 + r4.pertsr4) / (r1 + r2 + r3 + r4)   
            
            for j, rule in enumerate(self.rules):
                pert1 = pert0[j][0]
                pert2 = pert0[j][1]  
                pert3 = pert0[j][2]
                pert4 = pert0[j][3]  
                
                print('pert1,pert2,pert3,pert4')
                print(pert1,pert2,pert3,pert4)

                # Exibir a parte SE (antecedentes)
                antecedents = []
                for k, mf_idx in enumerate(rule):
                    antecedents.append(f"{input[k]} é {conj[mf_idx]}")

                antecedents_str = " e ".join(antecedents)

                # Exibir a parte ENTÃO (consequentes)
                num_inputs = self.X.shape[1]
                start_idx = j * (num_inputs + 1)
                end_idx = start_idx + num_inputs

                # Pegando os valores numéricos corretamente para exibir
                consequents = self.consequents[start_idx:end_idx].flatten()
                #
                p = consequents[0]
                q = consequents[1]
                r = consequents[2]
                t = consequents[3]

                print('p,q,r,t')
                print(p,q,r,t)

                #print(f"\n BIAS:{self.consequents[end_idx]}, {consequents.shape}")
                
                bias = self.consequents[end_idx]  # Termo constante (bias)

                # Corrigir formatação para números individuais
                consequent_str = " + ".join([f"{float(consequents[k])}*{input[k]}" for k in range(len(consequents))])
                consequent_str += f" + {float(bias)}"  # Termo constante (bias)
                
                regra = x1 * p + x2 * q + x3 * r + x4 * t + bias

                print('regra')
                print(regra)

                print('(pert1 * pert2 * pert3 * pert4)')
                print(pert1 * pert2 * pert3 * pert4)


                multRegraPert = regra * (pert1 * pert2 * pert3 * pert4)         # pert1*pert2 = w[correspondente]
                sumNumerador += multRegraPert
                sumDenominador += (pert1 * pert2 * pert3 * pert4)       
            
                print('multRegraPert')
                print(multRegraPert)
                print('sumNumerador')
                print(sumNumerador)
                print("sumDenominador")
                print(sumDenominador)

                # Printar a regra formatada
                print(f"Regra {j+1}: Se ({antecedents_str}), então ({consequent_str}) = {regra}\n(pert_1: {pert1}, pert_2: {pert2}, pert_3: {pert3}, pert_4: {pert4})\nPeso da regra = pert_1 * pert_2 * pert_3 * pert_4 = { pert1 * pert2 * pert3 * pert4 }\tRegra*Peso da Regra = {regra*(pert1 * pert2 * pert3 * pert4)}\n")
                ps += (pert1 * pert2 * pert3 * pert4)
                pr += regra * (pert1 * pert2 * pert3 * pert4)
               
                print("ps e pr")
                print(ps,pr)
                # regra = x0*p + x1*q + b
                
            print(f"( Somatório das regras*peso:{pr} ) / ( Somatório dos pesos das regras:[{ps}] ) = {pr/ps}")
            print(f"Takage-Sugeno: {sumNumerador/sumDenominador} , layerfive: {y_layerFive[i]}")
            if round((sumNumerador / sumDenominador).item(), 8) == round(y_layerFive[i].item(), 8):

                contagem = contagem + 1
                print("acertou")
            
        print()
        
        print(f"acertos: {contagem}, total de entradas: {i+1}")      

#------------------------------------------------------------------------------------------------

    def printRules2(self, x_values, y_layerFive, y_pred, y_true, pert, conj, input):
        print("\nRegras do ANFIS:")
        contagem = 0
        tksg = [x_values[0], x_values[1], x_values[2]]

        quantityofMFs = len(conj)
        quantityofInputParam = len(input)

        arquivo = open("saida.txt", "w")
        # Itera sobre cada linha de x_values
        for i, current_x in enumerate(x_values):
            listofInputs = []

            sumNumerador = 0
            sumDenominador = 0

            ps = 0
            pr = 0


            #print('x1,x2,x3,x4')
            for j in range(quantityofInputParam):
                listofInputs.append(current_x[j])
            #print(listofInputs)

            pert0 = pert[i]
            #print(listofInputs)

            middleMFs = ""
            for j in range(quantityofInputParam):
                #print(j)
                #print(listofInputs)
                #print(listofInputs[j])
                middleMFs += f"{input[j]}: {listofInputs[j]},"
                #Para entrada_{i+1} = [{input[0]}: {x1}, {input[1]}: {x2}, {input[2]}: {x3}, {input[3]}: {x4}]

            middleMFs = middleMFs.rstrip(", ")
            print(f"\nPara entrada_{i+1} = [{middleMFs}],  class_pred = {y_pred[i]} , class_true = [{y_true.iloc[i]}]:")

            # (r1.pretsr1 + r2.pertsr2 + r3.pertsr3 + r4.pertsr4) / (r1 + r2 + r3 + r4)   
            
            for j, rule in enumerate(self.rules):
                pertValueList = []
                for k in range(quantityofInputParam):
                    #print(pert0[j])
                    pertValueList.append(pert0[j][k])
                
                #print('pert1,pert2,pert3,pert4')
                #print(pertValueList)

                # Exibir a parte SE (antecedentes)
                antecedents = []
                for k, mf_idx in enumerate(rule):
                    antecedents.append(f"{input[k]} é {conj[mf_idx]}")

                antecedents_str = " e ".join(antecedents)

                # Exibir a parte ENTÃO (consequentes)
                num_inputs = self.X.shape[1]
                start_idx = j * (num_inputs + 1)
                end_idx = start_idx + num_inputs

                # Pegando os valores numéricos corretamente para exibir
                consequents = self.consequents[start_idx:end_idx].flatten()
                #

                #print(f"\n BIAS:{self.consequents[end_idx]}, {consequents.shape}")
                
                bias = self.consequents[end_idx]  # Termo constante (bias)

                # Corrigir formatação para números individuais
                consequent_str = " + ".join([f"{float(consequents[k])}*{input[k]}" for k in range(len(consequents))])
                consequent_str += f" + {float(bias)}"  # Termo constante (bias)

                #para não iniciar com regra = 0
                #print('p,q,r,t')
                #print(consequents[0])
                regra = listofInputs[0] * consequents[0]
                if(len(consequents) > 1):
                    for k in range(len(consequents)-1):
                        #print(consequents[k+1])
                        regra += listofInputs[k+1] * consequents[k+1]
                regra = regra + bias

                #print('regra')
                #print(regra)
                pertValueMult = 1
                
                #print("pert1 * pert2 * pert3 * pert4 nas próximas 2 linhas:")
                for k in range(quantityofInputParam):
                    #print(pertValueList[k])
                    pertValueMult *= pertValueList[k]
                #print(pertValueMult)

                multRegraPert = regra * (pertValueMult)         # pert1*pert2 = w[correspondente]
                sumNumerador += multRegraPert
                sumDenominador += (pertValueMult)       
            
                #print('multRegraPert')
                #print(multRegraPert)
                #print('sumNumerador')
                #print(sumNumerador)
                #print("sumDenominador")
                #print(sumDenominador)

                middleMFs = ""
                for k in range(quantityofInputParam):
                    middleMFs += f"pert_{k+1}: {pertValueList[k]},"
                    #Regra {j+1}: Se ({antecedents_str}), então ({consequent_str}) = {regra}\n(pert_1: {pert1}, pert_2: {pert2}, pert_3: {pert3}, pert_4: {pert4}

                middleMFs = middleMFs.rstrip(", ")

                middleMFs = ""
                for k in range(quantityofMFs):
                    middleMFs += f"pert_{k+1}*"
                    #Peso da regra = pert_1 * pert_2 * pert_3 * pert_4

                middleMFs = middleMFs.rstrip("* ")

                # Printar a regra formatada
                print(f"Regra {j+1}: Se ({antecedents_str}), então ({consequent_str}) = {regra}\n({middleMFs})\nPeso da regra = {middleMFs} = {pertValueMult}\tRegra*Peso da Regra = {regra*(pertValueMult)}\n")
                arquivo.write(f"Regra {j+1}: Se ({antecedents_str}), então ({consequent_str}) = {regra}\n({middleMFs})\nPeso da regra = {middleMFs} = {pertValueMult}\tRegra*Peso da Regra = {regra*(pertValueMult)}\n")
                ps += (pertValueMult)
                pr += regra * (pertValueMult)
                # regra = x0*p + x1*q + b
                
            print(f"( Somatório das regras*peso:{pr} ) / ( Somatório dos pesos das regras:[{ps}] ) = {pr/ps}")
            print(f"Takage-Sugeno: {sumNumerador/sumDenominador} , layerfive: {y_layerFive[i]}")
            if round((sumNumerador / sumDenominador).item(), 8) == round(y_layerFive[i].item(), 8):

                contagem = contagem + 1
                print("acertou")
            
        print()
        
        print(f"acertos: {contagem}, total de entradas: {i+1}")
        arquivo.close()

#------------------------------------------------------------------------------------------------

    def LSE(self, A, B, initialGamma = 1000.):
        coeffMat = A
        rhsMat = B
        S = np.eye(coeffMat.shape[1])*initialGamma
        x = np.zeros((coeffMat.shape[1],1)) # need to correct for multi-dim B
        for i in range(len(coeffMat[:,0])):
            a = coeffMat[i,:]
            b = np.array(rhsMat[i])
            S = S - (np.array(np.dot(np.dot(np.dot(S,np.matrix(a).transpose()),np.matrix(a)),S)))/(1+(np.dot(np.dot(S,a),a)))
            x = x + (np.dot(S,np.dot(np.matrix(a).transpose(),(np.matrix(b)-np.dot(np.matrix(a),x)))))
        return x

    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):

        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (convergence is not True):

            #layer four: forward pass
            [layerFour, wSum, w] = forwardHalfPass(self, self.X)

            #layer five: least squares estimate
            layerFive = np.array(self.LSE(layerFour,self.Y,initialGamma))
            self.consequents = layerFive
            #if (epoch == 1):
            #    print('Camada 4: ' + str(layerFour))
            #    print('Camada 5: ' + str(layerFive))
            layerFive = np.dot(layerFour,layerFive)
            #if (epoch == 1):
            #    print('Saída Final: ' + str(layerFive))

            #error
            #if (epoch == 1):
            #    print('Calculo do erro: ' + str(self.Y) + '-' + str(layerFive.T) )
            error = np.sum((self.Y-layerFive.T)**2)
            print('Epoch: [' + str(epoch) + '] current error: '+ str(error))
            average_error = np.average(np.absolute(self.Y-layerFive.T))
            self.errors = np.append(self.errors,error)

            if len(self.errors) != 0:
                if self.errors[len(self.errors)-1] < tolerance:
                    convergence = True

            # back propagation
            if convergence is not True:
                cols = range(len(self.X[0,:]))
                dE_dAlpha = list(backprop(self, colX, cols, wSum, w, layerFive) for colX in range(self.X.shape[1]))


            if len(self.errors) >= 4:
                if (self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9

            ## handling of variables with a different number of MFs
            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            eta = k / np.abs(np.sum(t))

            if(np.isinf(eta)):
                eta = k

            ## handling of variables with a different number of MFs
            dAlpha = copy.deepcopy(dE_dAlpha)
            if not(self.memFuncsHomo):
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * np.array(dE_dAlpha)


            for varsWithMemFuncs in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[varsWithMemFuncs])):
                    paramList = sorted(self.memFuncs[varsWithMemFuncs][MFs][1])
                    for param in range(len(paramList)):
                        self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]] = self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]] + dAlpha[varsWithMemFuncs][MFs][param]
            epoch = epoch + 1
            
            #if(epoch == 20):
                #(self, x_values, y_layerFive, y_pred, y_true, pert)
                #self.printRulesParaDUAS_entradas(self.X, self.consequents, layerFive.T, self.Y, self.memFuncsByVariable)

        self.fittedValues = predict(self,self.X)
        self.residuals = self.Y - self.fittedValues[:,0]
        
        return self.fittedValues


    def plotErrors(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.errors)),self.errors,'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()

    def plotMF(self, x, inputVar):
        import matplotlib.pyplot as plt
        from skfuzzy import gaussmf, gbellmf, sigmf

        for mf in range(len(self.memFuncs[inputVar])):
            if self.memFuncs[inputVar][mf][0] == 'gaussmf':
                y = gaussmf(x,**self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'gbellmf':
                y = gbellmf(x,**self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'sigmf':
                y = sigmf(x,**self.memClass.MFList[inputVar][mf][1])

            plt.plot(x,y,'r')

        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.fittedValues)),self.fittedValues,'r', label='trained')
            plt.plot(range(len(self.Y)),self.Y,'b', label='original')
            plt.legend(loc='upper left')
            plt.show()



def forwardHalfPass(ANFISObj, Xs):
    layerFour = np.empty(0,)
    wSum = []

    for pattern in range(len(Xs[:,0])):
        #layer one
        layerOne = ANFISObj.memClass.evaluateMF(Xs[pattern,:])
        #layer two
        miAlloc = [[layerOne[x][ANFISObj.rules[row][x]] for x in range(len(ANFISObj.rules[0]))] for row in range(len(ANFISObj.rules))]

        #print(f"miAlloc: {miAlloc}")
        
        # faz o produto dos das pertinencias por regra - que será o peso da regra 
        
        # layerTwo = np.array([np.product(x) for x in miAlloc])

        #=============================
        option =  ANFISObj.cm2 # Pode ser "produto", "minimo" ou "media_geometrica"           

        layerTwo = [] # soma das pertinencias por regra

        # Iteramos sobre cada lista em miAlloc, que possui todas as pertinencias
        
        for x in miAlloc:
            
            if option == "produto":
                result = 1
                for valor in x:
                    result *= valor  # Multiplicamos os valores dentro da lista
                    
            elif option == "minimo":
                result = x[0]
                for valor in x[1:]:
                    result = min(result, valor)  # Atualiza o resultado com o mínimo

            elif option == "lukasiewicz":
                result = x[0]
                for valor in x[1:]:
                    result = max(1e-6, result + valor - 1)

            elif option == "produto_drastico":
                result = x[0]
                for valor in x[1:]:
                    if result == 1.0:
                        result = valor
                    elif valor == 1.0:
                        continue  # result permanece o mesmo
                    else:
                        result = 1e-12


            elif option == "nilpotente_min": 
                result = x[0]
                for valor in x[1:]:
                    if (result + valor) > 1:
                        result = min(result, valor)
                    else:
                        result = 1e-12


            elif option == "hamacher_prod": 
                result = x[0]
                for valor in x[1:]:
                    if (result == valor) == 0:
                        result = 1e-12
                    else:
                        result = result * valor / (result + valor - (result * valor))


            elif option == "frank":
                lambda_param = 10 # apartir de 1050 gera overflow
                    # apartir de 4 melhora a acuracia
                    # 10 acuracia igual (0.98), mas valor otimo para o erro(< 19)
                result = x[0]
                for valor in x[1:]:
                    numerador = ( lambda_param**result - 1 ) * ( lambda_param**valor - 1 )
                    result = np.log2(1 + numerador / ( lambda_param - 1 ) )
            

            elif option == "sugeno_weber":
                lambda_param = 0  # Deve ser ≥ -1
                # qlqr valor maior que 0 aumenta o erro e não a acuracia
                result = x[0]
                for valor in x[1:]:
                    numerador = result + valor - 1 + lambda_param * result * valor
                    denominador = 1 + lambda_param
                    result = max( 1e-12, numerador / denominador)


            elif option == "yager":
                lambda_param = 1
                result = x[0]
                for valor in x[1:]:

                    result = max(1e-12,  1-( (1 - result)**lambda_param + (1 - valor)**lambda_param ) **(1/lambda_param) )

            elif option == "dombi":
                lambda_param = 1
                result = x[0]
                for valor in x[1:]:
                    denominador = ( ((1-result)/result)**lambda_param + ((1-valor)/valor)**lambda_param )**(1/lambda_param)
                    result = 1 / (1 + denominador)

            elif option == "schweizer_skar":
                lambda_param = 3 # otimo, melhor acuracia e erro)         
                result = x[0]
                for valor in x[1:]:
                    result = max(1e-12, ( result**lambda_param + valor**lambda_param - 1 ) )**(1/lambda_param)
            
            # ========== medias ==========

            elif option == "media_geometrica":
                # Função para calcular a média geométrica de uma lista de valores
                result = 1
                for valor in x:
                    result *= valor
                    result =  result ** 0.5  # Raiz quadrada do produto
            
            # ========== antigas T-NORMAS ==========         

                    
            elif option == "einstein":
                if not x:
                    result = 0.0
                else:
                    result = x[0]
                    for valor in x[1:]:
                        epsilon = 1e-12
                        numerador = result * valor
                        denominador = 2 - (result + valor - numerador) + epsilon
                        result = numerador / denominador
                    result = np.clip(result, 0.0, 1.0)
                    
            elif option == "dubois_prade":
                if not x:
                    result = 0.0
                else:
                    alpha = 0.5
                    result = x[0]
                    for valor in x[1:]:
                        denominator = max(result, valor, alpha) + 1e-12
                        result = (result * valor) / denominator
                        
                        
            # ========== FIM ==========
            
            else:
                raise ValueError(f"Operação não reconhecida: {option}")
            
            # Garantia final contra valores inválidos
            result = np.nan_to_num(result, nan=0.0)
            result = np.clip(result, 0.0, 1.0)
            
            layerTwo.append(result)            
        #==============================================
        
        if pattern == 0:
            w = layerTwo
        else:
            w = np.vstack((w,layerTwo))

        #layer three
        wSum.append(np.sum(layerTwo))
        if pattern == 0:
            wNormalized = layerTwo/wSum[pattern]
        else:
            wNormalized = np.vstack((wNormalized,layerTwo/wSum[pattern]))
            
        #prep for layer four (bit of a hack)
        layerThree = layerTwo/wSum[pattern] # pesos das regras normalizados ex: w1/w1+w2
        rowHolder = np.concatenate([x*np.append(Xs[pattern,:],1) for x in layerThree])
        layerFour = np.append(layerFour,rowHolder) # regra*seu_peso, ex: w1*f1 = w1(p1x+q1y+r1)
    w = w.T # w guardava as pertinencias: entradas x regras, agora regras x entradas = 81 regras x 45 entradas 
    wNormalized = wNormalized.T
    layerFour = np.array(np.array_split(layerFour,pattern + 1))

    return layerFour, wSum, w


def backprop(ANFISObj, columnX, columns, theWSum, theW, theLayerFive):

    paramGrp = [0]* len(ANFISObj.memFuncs[columnX])
    for MF in range(len(ANFISObj.memFuncs[columnX])):

        parameters = np.empty(len(ANFISObj.memFuncs[columnX][MF][1]))
        timesThru = 0
        for alpha in sorted(ANFISObj.memFuncs[columnX][MF][1].keys()):

            bucket3 = np.empty(len(ANFISObj.X))
            for rowX in range(len(ANFISObj.X)):
                varToTest = ANFISObj.X[rowX,columnX]
                tmpRow = np.empty(len(ANFISObj.memFuncs))
                tmpRow.fill(varToTest)

                bucket2 = np.empty(ANFISObj.Y.ndim)
                for colY in range(ANFISObj.Y.ndim):

                    rulesWithAlpha = np.array(np.where(ANFISObj.rules[:,columnX]==MF))[0]
                    adjCols = np.delete(columns,columnX)

                    senSit = mfDerivs.partial_dMF(ANFISObj.X[rowX,columnX],ANFISObj.memFuncs[columnX][MF],alpha)
                    # produces d_ruleOutput/d_parameterWithinMF
                    dW_dAplha = senSit * np.array([np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha])

                    bucket1 = np.empty(len(ANFISObj.rules[:,0]))
                    for consequent in range(len(ANFISObj.rules[:,0])):
                        fConsequent = np.dot(np.append(ANFISObj.X[rowX,:],1.),ANFISObj.consequents[((ANFISObj.X.shape[1] + 1) * consequent):(((ANFISObj.X.shape[1] + 1) * consequent) + (ANFISObj.X.shape[1] + 1)),colY])
                        acum = 0
                        if consequent in rulesWithAlpha:
                            acum = dW_dAplha[np.where(rulesWithAlpha==consequent)] * theWSum[rowX]

                        acum = acum - theW[consequent,rowX] * np.sum(dW_dAplha)
                        acum = acum / theWSum[rowX]**2
                        bucket1[consequent] = fConsequent * acum

                    sum1 = np.sum(bucket1)

                    if ANFISObj.Y.ndim == 1:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX]-theLayerFive[rowX,colY])*(-2)
                    else:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX,colY]-theLayerFive[rowX,colY])*(-2)

                sum2 = np.sum(bucket2)
                bucket3[rowX] = sum2

            sum3 = np.sum(bucket3)
            parameters[timesThru] = sum3
            timesThru = timesThru + 1

        paramGrp[MF] = parameters

    return paramGrp


def predict(ANFISObj, varsToTest):

    [layerFour, wSum, w] = forwardHalfPass(ANFISObj, varsToTest)

    #layer five
    layerFive = np.dot(layerFour,ANFISObj.consequents)
    
    return layerFive


if __name__ == "__main__":
    print("I am main!")



def gaussmf(mf):
    """
    Plota funções gaussianas a partir dos parâmetros fornecidos.

    Parâmetros:
    - mf: lista de listas, onde cada sublista contém funções gaussianas com seus parâmetros.
    """
    
    # Definindo o intervalo de valores para x
    x = np.linspace(-3, 13, 100)  # Aumenta o intervalo para capturar melhor as gaussianas

    # Loop para plotar cada conjunto separadamente
    for idx, group in enumerate(mf):  # Itera sobre cada grupo de gaussianas
        plt.figure(figsize=(10, 6))  # Cria uma nova figura para cada grupo

        for params in group:  # Itera sobre cada função gaussiana dentro do grupo
            mf_type, mf_params = params  # Extrai o tipo e os parâmetros da função
            if mf_type == 'gaussmf':  # Verifica se o tipo é 'gaussmf'
                y = np.exp(-((x - mf_params['mean']) ** 2) / (2 * mf_params['sigma'] ** 2))  # Calcula a gaussiana
                plt.plot(x, y, label=f'Gaussian (mean={mf_params["mean"]}, sigma={mf_params["sigma"]})')

        # Configurando o gráfico
        plt.title(f"Funções Gaussianas - Entrada {idx + 1} \n Entrada 1: Petal Length (cm), Entrada 2: Petal Width (cm)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(-0.1, 1.1)  # Ajusta os limites do eixo y para melhor visualização
        plt.legend()
        plt.grid(True)
        plt.show()