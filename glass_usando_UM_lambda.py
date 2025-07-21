import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import metricas
import anfis
import membership.membershipfunction
import time
import os

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
    # O separador agora é mais robusto para lidar com espaços variáveis
    df = pd.read_csv(arquivo, header=None, skiprows=skiprows, sep=',\s*', engine='python', skipinitialspace=True)
    
    # Remove aspas e converte para float (exceto a última coluna, que é a classe)
    for col in range(df.shape[1] - 1):
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(r'[",]', '', regex=True).astype(float)
    
    return df

def converter_classes(y, dataset):
    """Converte as classes de string para valores numéricos."""
    y_str = [str(val).strip() for val in y]
    class_mapping = {
         'glass': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6},
    }.get(dataset, None)
    
    if class_mapping is None:
        raise ValueError(f"Dataset {dataset} não suportado. Mapeamento de classe não encontrado.")
    
    return np.array([class_mapping[val] for val in y_str])

def validacao_cruzada(dataset, abordagem, lambda_param, n_splits=5):
    """
    Executa a validação cruzada para um dataset usando uma abordagem específica e um lambda_param.
    Para cada fold:
      - Arquivo de teste: Datasets/{dataset}/{dataset}-5-{i}tst.dat
      - Arquivos de treino: os demais (Datasets/{dataset}/{dataset}-5-{j}tra.dat, j ≠ i)
    Retorna a acurácia média das n_splits rodadas.
    """
    accuracies = []
    print(f"\nDataset: {dataset} | Abordagem: {abordagem} | Lambda: {lambda_param}")
    for i in range(1, n_splits + 1):
        print(f"  Rodada {i}:")
        try:
            # Carrega e processa o conjunto de teste
            test_file = f"Datasets/{dataset}/{dataset}-5-{i}tst.dat"
            test_data = ler_e_preprocessar(test_file)
            X_test = test_data.iloc[:, :-1].values
            y_test = converter_classes(test_data.iloc[:, -1].values, dataset)
            
            # Carrega e concatena os conjuntos de treino
            treino_list = []
            for j in range(1, n_splits + 1):
                if j == i:
                    continue
                train_file = f"Datasets/{dataset}/{dataset}-5-{j}tra.dat"
                df_temp = ler_e_preprocessar(train_file)
                treino_list.append(df_temp)
            train_data = pd.concat(treino_list, ignore_index=True)
            X_train = train_data.iloc[:, :-1].values
            y_train = converter_classes(train_data.iloc[:, -1].values, dataset)
            
            # Normalização dos dados
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Definição das funções de pertinência
            n_features = X_train.shape[1]
            mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}],
                   ['gaussmf', {'mean': 5., 'sigma': 1.}]] for _ in range(n_features)]
            mfc = membership.membershipfunction.MemFuncs(mf)
            
            # Criação, treino e predição com o modelo ANFIS
            anf = anfis.ANFIS(X_train_scaled, y_train, mfc, abordagem)

            # ⏱️ Medição do tempo de treinamento
            start_time = time.time()
            
            anf.trainHybridJangOffLine(epochs=2, lambda_param=lambda_param)
            y_pred = anfis.predict(anf, X_test_scaled, lambda_param)

            # ⏱️ Fim da medição
            training_time = time.time() - start_time
            
            # Cálculo e armazenamento da acurácia
            acc = metricas.acuracia(y_test, y_pred)
            accuracies.append(acc)

            print(f"    Acurácia: {acc:.4f} | Tempo de treino: {training_time:.2f} segundos")

        except Exception as e:
            print(f"    Erro na rodada {i}: {e}")
            accuracies.append(0) # Adiciona 0 se ocorrer um erro para não quebrar a média
            
    media = np.mean(accuracies)
    print(f"  Acurácia média para {dataset} com {abordagem} (Lambda: {lambda_param}): {media:.4f}\n")
    
    # fim tempo total
    time_total = time.time() - start_time_total
    print(f"\nTempo total de execução: {time_total:.2f} segundos")

    return media

# --- Bloco Principal de Execução ---

#contador de tempo total
start_time_total = time.time()

# Abordagens desejadas
abordagens = {
    "produto": "produto", "minimo": "minimo", "lukasiewicz": "lukasiewicz",
    "produto_drastico": "produto_drastico", "nilpotente_min": "nilpotente_min",
    "hamacher_prod": "hamacher_prod", "frank": "frank", "sugeno_weber": "sugeno_weber",
    "yager": "yager", "dombi": "dombi", "schweizer_skar": "schweizer_skar",
}

# Lista de datasets a serem utilizados
datasets = [ 'glass']

# Definindo o valor único para lambda
lambda_valor = 0

# Executa a validação cruzada e armazena os resultados
resultados_lista = []
for ds in datasets:
    for nome_abordagem, abordagem in abordagens.items():
        media_acc = validacao_cruzada(ds, abordagem, lambda_valor)
        resultados_lista.append({
            'Dataset': ds,
            'Abordagem': nome_abordagem,
            'Lambda': lambda_valor,
            'Acurácia Média': media_acc
        })

# Cria o DataFrame com todos os resultados
df_resultados = pd.DataFrame(resultados_lista)

# Cria a tabela pivot para visualização
print(f"\n--- Tabela de Resultados para Lambda = {lambda_valor} ---")
pivot_df = df_resultados.pivot(index='Dataset', columns='Abordagem', values='Acurácia Média')

# Adiciona uma última linha com a média de cada coluna (por abordagem)
if not pivot_df.empty:
    pivot_df.loc['Média'] = pivot_df.mean()

print(pivot_df)

# Salva os resultados em um único arquivo Excel com formatação
output_dir = "lamb_0"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
output_file = f"{output_dir}/teste_lambda_{lambda_valor}.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    pivot_df.to_excel(writer, sheet_name=f"Resultados_Lambda_{lambda_valor}")
    workbook  = writer.book
    worksheet = writer.sheets[f"Resultados_Lambda_{lambda_valor}"]
    
    bold_format = workbook.add_format({'bold': True})
    
    # Destaca o maior valor em cada linha de dados (exceto a linha 'Média')
    (max_row, max_col) = pivot_df.shape
    # Itera sobre as linhas de dados (de 0 até a penúltima)
    for row_num in range(max_row - 1):
        # Pega os valores da linha no DataFrame
        row_values = pivot_df.iloc[row_num, :].tolist()
        if row_values: # Verifica se a lista não está vazia
            max_val = max(row_values)
            # Encontra as colunas com o valor máximo
            for col_num, current_val in enumerate(row_values):
                if current_val == max_val:
                    # Escreve na célula correspondente no Excel com formato de negrito
                    # +1 porque as linhas/colunas do xlsxwriter são baseadas em 0
                    worksheet.write(row_num + 1, col_num + 1, current_val, bold_format)
    
print(f"\nResultados salvos em '{output_file}'.")
