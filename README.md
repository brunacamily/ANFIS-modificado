# Como executar o A2-ANFIS

## Execução rápida

- Para agregadores **sem parâmetro lambda**:
  
  ```
  python3 seg_otim_sLambda.py produto 9
  ```
  - Onde `produto` é o agregador (veja todos em `agregadores.py`)
  - `9` é o número do dataset (veja a lista em `datasets.json`)

- Para agregadores **com parâmetro lambda**:
  
  ```
  python3 seg_otim_cLambda.py produto 9 0
  ```
  - `produto` é o agregador
  - `9` é o número do dataset
  - `0` é o valor do lambda a ser usado

- Também é possível rodar ambos automaticamente com:
  
  ```
  bash roda.sh
  ```

## Localização dos arquivos

- **Datasets disponíveis:**  
  Veja o arquivo `datasets.json` para os nomes e índices dos datasets.
- **Agregadores disponíveis:**  
  Veja o arquivo `agregadores.py` para a lista de agregadores implementados e quais aceitam parâmetro lambda.

## Observações

- O resultado de cada execução será salvo em arquivos `.txt` conforme definido nos scripts.
- Para alterar agregador, dataset ou lambda, basta mudar os argumentos.



## Créditos

Este projeto é uma extensão e automação do modelo ANFIS desenvolvido originalmente por [twmeggs](https://github.com/twmeggs). 

O repositório base pode ser encontrado em: [twmeggs/anfis](https://github.com/twmeggs/anfis)
