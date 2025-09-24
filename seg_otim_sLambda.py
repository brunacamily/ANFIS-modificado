import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Módulos internos
sys.path.append(str(Path(__file__).resolve().parent.parent))
import metricas
import anfis
import membership.membershipfunction as membership
import json

# === Carregar configurações dos datasets ===
CONFIG_PATH = Path("datasets.json")
with open(CONFIG_PATH, "r") as f:
    DATASET_CONFIGS = json.load(f)

# --- Funções Utilitárias ---
def converter_classes(y, classes_map):
    """Mapeia classes string para inteiros, de acordo com o mapeamento."""
    y_str = [str(val).strip() for val in y]
    return np.array([classes_map[val] for val in y_str])

def encontrar_linha_data(arquivo):
    """Retorna a linha de início dos dados em um ARFF."""
    with open(arquivo, "r") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("@data"):
                return i + 1
    return 0

def ler_e_preprocessar(arquivo):
    """Carrega e converte dados ARFF para DataFrame."""
    skiprows = encontrar_linha_data(arquivo)
    df = pd.read_csv(
        arquivo,
        header=None,
        skiprows=skiprows,
        sep=r",\s*",
        engine="python",
        skipinitialspace=True,
    )
    for col in range(df.shape[1] - 1):
        if df[col].dtype == "object":
            df[col] = df[col].str.replace(r'[",]', "", regex=True).astype(float)
    return df

def carregar_dataset(dataset_cfg, i_teste):
    """Carrega conjunto de teste e treino para um fold específico."""
    dataset_name = dataset_cfg["name"]
    n_splits = dataset_cfg["n_splits"]
    classes_map = dataset_cfg["classes"]

    test_file = Path(f"Datasets/{dataset_name}/{dataset_name}-5-{i_teste}tst.dat")
    test_data = ler_e_preprocessar(test_file)
    X_test = test_data.iloc[:, :-1].values
    y_test = converter_classes(test_data.iloc[:, -1].values, classes_map)

    treino_list = []
    for j in range(1, n_splits + 1):
        if j == i_teste:
            continue
        train_file = Path(f"Datasets/{dataset_name}/{dataset_name}-5-{j}tra.dat")
        treino_list.append(ler_e_preprocessar(train_file))
    train_data = pd.concat(treino_list, ignore_index=True)
    X_train = train_data.iloc[:, :-1].values
    y_train = converter_classes(train_data.iloc[:, -1].values, classes_map)

    return X_train, y_train, X_test, y_test

def destacar_max_excel(workbook, worksheet, pivot_df):
    bold_format = workbook.add_format({"bold": True})
    max_row, _ = pivot_df.shape
    for row_num in range(max_row - 1):
        row_values = pivot_df.iloc[row_num].tolist()
        if row_values:
            max_val = max(row_values)
            for col_num, val in enumerate(row_values):
                if val == max_val:
                    worksheet.write(row_num + 1, col_num + 1, val, bold_format)

# --- Função Principal de Validação ---
def validacao_cruzada(dataset_cfg, abordagem):
    """Executa validação cruzada k-fold para o dataset."""
    accuracies = []
    dataset_name = dataset_cfg["name"]
    n_splits = dataset_cfg["n_splits"]
    print(f"\n=== Dataset: {dataset_name} | Abordagem: {abordagem} ===")

    for i in range(1, n_splits + 1):
        print(f"  Fold {i}/{n_splits}...")
        try:
            X_train, y_train, X_test, y_test = carregar_dataset(dataset_cfg, i)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Funções de pertinência
            mf = [[
                ["gaussmf", {"mean": 0.0, "sigma": 1.0}],
                ["gaussmf", {"mean": 5.0, "sigma": 1.0}],
            ] for _ in range(X_train.shape[1])]
            mfc = membership.MemFuncs(mf)

            anf = anfis.ANFIS(X_train_scaled, y_train, mfc, abordagem)

            start_train_time = time.time()
            anf.trainHybridJangOffLine(epochs=2)
            y_pred = anfis.predict(anf, X_test_scaled)
            train_time = time.time() - start_train_time

            acc = metricas.acuracia(y_test, y_pred)
            accuracies.append(acc)
            print(f"    Acurácia: {acc:.4f} | Tempo treino: {train_time:.2f}s")

        except Exception as e:
            print(f"    Erro no fold {i}: {e}")
            accuracies.append(0)

    media = np.mean(accuracies)
    print(f"\n\nDataset: {dataset_name} | Abordagem: {abordagem} | Média final: {media:.4f}")
    return media

# --- Execução ---
if __name__ == "__main__":
    start_time_total = time.time()

    if len(sys.argv) < 3:
        print("Uso: python seg_otim.py <abordagem> <dataset_id>")
        sys.exit(1)

    abordagem_input = sys.argv[1]
    dataset_id = sys.argv[2]

    if dataset_id not in DATASET_CONFIGS:
        print(f"Dataset id '{dataset_id}' não encontrado no arquivo JSON.")
        sys.exit(1)

    dataset_cfg = DATASET_CONFIGS[dataset_id]
    OUTPUT_DIR = Path("faltantes/planilhas")

    resultados = []
    media_acc = validacao_cruzada(dataset_cfg, abordagem_input)
    resultados.append({
        "Dataset": dataset_cfg["name"],
        "Abordagem": abordagem_input,
        "Acurácia Média": media_acc
    })

    df_resultados = pd.DataFrame(resultados)
    pivot_df = df_resultados.pivot(index="Dataset", columns="Abordagem", values="Acurácia Média")

    if not pivot_df.empty:
        pivot_df.loc["Média"] = pivot_df.mean()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{dataset_cfg['name']}.xlsx"

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        pivot_df.to_excel(writer, sheet_name="Resultados")
        workbook = writer.book
        worksheet = writer.sheets["Resultados"]
        destacar_max_excel(workbook, worksheet, pivot_df)

    print(f"\n--- Tempo total: {time.time() - start_time_total:.2f}s ---")
    print(f"Resultados salvos em '{output_file}'.")

    # pra executar: python3 dombi_teste/seg_otim.py t_norma_AA 1 100
    # onde "t_norma_AA" é a abordagem, "1" é o dataset_id e "100" é o valor de lambda.