from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import numpy
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


def acuracia(y_test, y_pred):
    # Ajusta as previsões ao espaço de classes
    y_pred_class = numpy.clip(numpy.round(y_pred).astype(int), a_min=min(y_test), a_max=max(y_test))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, y_pred_class)
    #print(f"\nAcurácia do modelo ANFIS: {accuracy:.2%}")
 
    return accuracy


def calcular_metricas_basicas(y_test, y_pred):
    # Ajusta as previsões ao espaço de classes
    y_pred_class = np.clip(np.round(y_pred).astype(int), a_min=min(y_test), a_max=max(y_test))
    
    # Calcula a matriz de confusão
    cm = confusion_matrix(y_test, y_pred_class)

    print("Matriz de Confusão:")
    print(cm)

    n_classes = cm.shape[0]

    # Calcula TPR e FPR para cada classe no formato one-vs-rest
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        print(f"\nClasse {i}:")
        print(f"  Verdadeiros Positivos (TP): {TP}")
        print(f"  Falsos Negativos (FN): {FN}")
        print(f"  Falsos Positivos (FP): {FP}")
        print(f"  Verdadeiros Negativos (TN): {TN}")
        print(f"  TPR (Sensibilidade): {TPR:.2f}")
        print(f"  FPR: {FPR:.2f}")







def plot_multiclass_roc(y_test, y_scores, classes):
    """
    Plota as curvas ROC para cada classe em um problema multiclasse.
    
    Parâmetros:
      y_test : array-like, rótulos reais (ex.: [0, 1, 2, 0, ...])
      y_scores : array-like, matriz de escores com shape (n_amostras, n_classes)
      classes : list, lista com os rótulos das classes (ex.: [0,1,2])
    """
    # Binariza os rótulos
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    plt.figure(figsize=(8,6))
    colors = ['blue', 'red', 'green']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'Classe {classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='AUC aleatória')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC Multiclasse')
    plt.legend(loc='lower right')
    plt.show()



def nova(y_true, y_scores):

    # Converta y_scores para um array numpy
    y_scores = np.array(y_scores).flatten()
    y_scores_transformed = transform_to_class_scores(y_scores)

    # Binarize os rótulos verdadeiros (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    # Calcule a curva ROC e a AUC para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):  # 3 classes
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plote a curva ROC para cada classe
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.show()
    

def transform_to_class_scores1(y_scores, num_classes=3):
    # Garantir que y_scores seja um array unidimensional
    y_scores = np.array(y_scores).flatten()
    
    y_scores_transformed = np.zeros((len(y_scores), num_classes))
    for i, score in enumerate(y_scores):
        # Atribuir um score alto para a classe mais próxima do valor contínuo
        predicted_class = int(round(score))
        y_scores_transformed[i, predicted_class] = 1.0  # Score máximo para a classe predita
        # Atribuir scores baixos para as outras classes
        for j in range(num_classes):
            if j != predicted_class:
                y_scores_transformed[i, j] = 0.0  # Score mínimo para as outras classes
    return y_scores_transformed


def transform_to_class_scores(y_scores, num_classes=3):
    # Garantir que y_scores seja um array unidimensional
    y_scores = np.array(y_scores).flatten()
    
    # Limitar os valores de y_scores ao intervalo das classes (0 a num_classes-1)
    y_scores = np.clip(y_scores, 0, num_classes - 1)
    
    y_scores_transformed = np.zeros((len(y_scores), num_classes))
    for i, score in enumerate(y_scores):
        # Atribuir um score alto para a classe mais próxima do valor contínuo
        predicted_class = int(round(score))
        y_scores_transformed[i, predicted_class] = 1.0  # Score máximo para a classe predita
        # Atribuir scores baixos para as outras classes
        for j in range(num_classes):
            if j != predicted_class:
                y_scores_transformed[i, j] = 0.0  # Score mínimo para as outras classes
    return y_scores_transformed


def roc(y_true, y_scores):
    
    # Transformar a saída do ANFIS em scores para cada classe
    y_scores_transformed = transform_to_class_scores(y_scores)
  
    # Binarize os rótulos verdadeiros (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    # Calcule a curva ROC e a AUC para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):  # 3 classes
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_transformed[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plote a curva ROC para cada classe
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.show()
    
    


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def rocSemPlot(y_true, y_scores):
    """
    Calcula e imprime as AUCs (Área Sob a Curva ROC) para cada classe.
    
    Parâmetros:
    - y_true: Rótulos verdadeiros (ground truth).
    - y_scores: Escores preditos pelo modelo (probabilidades ou scores de decisão).
    """
    # Transformar a saída do ANFIS em scores para cada classe
    y_scores_transformed = transform_to_class_scores(y_scores)
  
    # Binarizar os rótulos verdadeiros (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    # Calcular a curva ROC e a AUC para cada classe
    roc_auc = dict()
    
    for i in range(3):  # 3 classes
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores_transformed[:, i])
        roc_auc[i] = auc(fpr, tpr)
        print(f"AUC da Classe {i}: {roc_auc[i]:.2f}")
    
    return roc_auc