import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import metricas
import anfis
import membership.membershipfunction

import time

def encontrar_linha_data(arquivo):
    """Encontra a linha onde começa a seção @data no arquivo."""
    with open(arquivo, 'r') as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith('@data'):
                return i + 1  # Linha seguinte ao @data
    return 0

def ler_e_preprocessar(arquivo):
    """
    Lê o arquivo ARFF:
    - Detecta automaticamente o início dos dados (@data).
    - Remove vírgulas e espaços dos valores.
    - Converte atributos para float.
    """
    skiprows = encontrar_linha_data(arquivo)
    df = pd.read_csv(arquivo, header=None, skiprows=skiprows, sep=',\s*', engine='python')
    
    # Remover caracteres especiais e converter para float (exceto a última coluna)
    for col in range(df.shape[1] - 1):
        df[col] = df[col].astype(str).str.replace(r'[",]', '', regex=True).astype(float)
    
    return df

def converter_classes(y, dataset):

    y_str = [str(val).strip() for val in y]
    class_mapping = {

        'iris': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2},  
         
    }.get(dataset, None)
    if class_mapping is None:
        raise ValueError(f"Dataset {dataset} não suportado.")
    return np.array([class_mapping[val] for val in y_str])

def validacao_cruzada(dataset, abordagem, n_splits=5):
    """
    Executa a validação cruzada para um dataset usando uma abordagem específica.
    Para cada fold:
      - Arquivo de teste: Datasets/{dataset}/{dataset}-5-{i}tst.dat
      - Arquivos de treino: os demais (Datasets/{dataset}/{dataset}-5-{j}tra.dat, j ≠ i)
    Retorna a acurácia média das n_splits rodadas.
    """
    accuracies = []
    print(f"\nDataset: {dataset} | Abordagem: {abordagem}")
    for i in range(1, n_splits+1):
        print(f"  Rodada {i}:")
        try:
            test_file = f"Datasets/{dataset}/{dataset}-5-{i}tst.dat"
            test_data = ler_e_preprocessar(test_file)
            X_test = test_data.iloc[:, :-1].values
            y_test = converter_classes(test_data.iloc[:, -1].values, dataset)
            
            treino_list = []
            for j in range(1, n_splits+1):
                if j == i:
                    continue
                train_file = f"Datasets/{dataset}/{dataset}-5-{j}tra.dat"
                df_temp = ler_e_preprocessar(train_file)
                treino_list.append(df_temp)
            train_data = pd.concat(treino_list, ignore_index=True)
            X_train = train_data.iloc[:, :-1].values
            y_train = converter_classes(train_data.iloc[:, -1].values, dataset)
            
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}],
                   ['gaussmf', {'mean': 5., 'sigma': 1.}]] for _ in range(n_features)]
            mfc = membership.membershipfunction.MemFuncs(mf)
            
            anf = anfis.ANFIS(X_train_scaled, y_train, mfc, abordagem)



            # ⏱️ Medição do tempo de treinamento
            start_time = time.time()

            anf.trainHybridJangOffLine(epochs=2)    # rodanndo treino da rede por 2 épocas
            y_pred = anfis.predict(anf, X_test_scaled)  # rodando teste na rede

            # ⏱️ Fim da medição
            training_time = time.time() - start_time


            acc = metricas.acuracia(y_test, y_pred)
            accuracies.append(acc)

            print(f"    Acurácia: {acc:.4f} | Tempo de treino: {training_time:.2f} segundos")

        except Exception as e:
            print(f"    Erro na rodada {i}: {e}")
            accuracies.append(0)
    media = np.mean(accuracies)
    print(f"  Acurácia média para {dataset} com {abordagem}: {media:.4f}\n")

    # fim tempo total
    time_total = time.time() - start_time_total
    print(f"\nTempo total de execução: {time_total:.2f} segundos")

    return media


#contador de tempo total
start_time_total = time.time()


# Abordagens desejadas
abordagens = {
   "dombi": "dombi",
}

# Lista de datasets a serem utilizados
datasets = [
    'iris', 
]

# Executa a validação cruzada e organiza os resultados em uma tabela pivot:
# Linhas = datasets, Colunas = abordagens
resultados_lista = []
for ds in datasets:
    for nome_abordagem, abordagem in abordagens.items():
        media_acc = validacao_cruzada(ds, abordagem)
        resultados_lista.append({
            'Dataset': ds,
            'Abordagem': nome_abordagem,
            'Acurácia Média': media_acc
        })
df_resultados = pd.DataFrame(resultados_lista)
pivot_df = df_resultados.pivot(index='Dataset', columns='Abordagem', values='Acurácia Média')

# Adiciona uma última linha com a média de cada coluna (por abordagem)
pivot_df.loc['Média'] = pivot_df.mean()

print("Tabela de Resultados:")
print(pivot_df)

# Salva os resultados em Excel com formatação:
# Cada linha (exceto a última) terá o maior valor destacado em negrito.
output_file = "resultados_8_datasets.xlsx"
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    pivot_df.to_excel(writer, sheet_name="Resultados")
    workbook  = writer.book
    worksheet = writer.sheets["Resultados"]
    
    bold_format = workbook.add_format({'bold': True})
    
    # Obter dimensões da tabela (incluindo o cabeçalho)
    (max_row, max_col) = pivot_df.shape
    # O Excel inicia as linhas em 1 (linha 0 é o cabeçalho) e a coluna 0 contém o índice.
    # Itera sobre as linhas de dados (exceto o cabeçalho e a última linha 'Média')
    for row in range(1, max_row):  # linhas 1 a max_row-1 (já que pivot_df.index[-1] é 'Média')
        # Ler os valores da linha (colunas 1 a max_col)
        row_values = [pivot_df.iloc[row-1, col] for col in range(max_col)]
        max_val = max(row_values)
        for col in range(max_col):
            if pivot_df.iloc[row-1, col] == max_val:
                # As colunas do Excel começam na coluna 1 (coluna 0 é o índice)
                worksheet.write(row, col+1, max_val, bold_format)
    
print(f"\nResultados salvos em '{output_file}'.")
 