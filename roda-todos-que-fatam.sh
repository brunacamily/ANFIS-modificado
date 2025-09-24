#!/usr/bin/env bash

python3 seg_otim_sLambda.py produto 13 >> faltantes/results/saida-pima_produto.txt &
python3 seg_otim_sLambda.py produto 15 >> faltantes/results/saida-yeast_produto.txt &

python3 seg_otim_cLambda.py dombi 6 0 >> faltantes/results/saida-glass_dombi.txt &
python3 seg_otim_cLambda.py hamacher_prod 6 0 >> faltantes/results/saida-glass_hamacher_prod.txt &
python3 seg_otim_cLambda.py frank 6 0 >> faltantes/results/saida-glass_frank.txt &
python3 seg_otim_cLambda.py sugeno_weber 6 0 >> faltantes/results/saida-glass_sugeno_weber.txt &
python3 seg_otim_cLambda.py yager 6 0 >> faltantes/results/saida-glass_yager.txt &
python3 seg_otim_cLambda.py schweizer_skar 6 0 >> faltantes/results/saida-glass_schweizer_skar.txt &

python3 seg_otim_cLambda.py dombi 12 0 >> faltantes/results/saida-phoneme_dombi.txt &
python3 seg_otim_cLambda.py hamacher_prod 12 0 >> faltantes/results/saida-phoneme_hamacher_prod.txt &
python3 seg_otim_cLambda.py frank 12 0 >> faltantes/results/saida-phoneme_frank.txt &
python3 seg_otim_cLambda.py sugeno_weber 12 0 >> faltantes/results/saida-phoneme_sugeno_weber.txt &
python3 seg_otim_cLambda.py yager 12 0 >> faltantes/results/saida-phoneme_yager.txt &
python3 seg_otim_cLambda.py schweizer_skar 12 0 >> faltantes/results/saida-phoneme_schweizer_skar.txt &

python3 seg_otim_cLambda.py dombi 15 0 >> faltantes/results/saida-yeast_dombi.txt &
python3 seg_otim_cLambda.py hamacher_prod 15 0 >> faltantes/results/saida-yeast_hamacher_prod.txt &
python3 seg_otim_cLambda.py frank 15 0 >> faltantes/results/saida-yeast_frank.txt &
python3 seg_otim_cLambda.py sugeno_weber 15 0 >> faltantes/results/saida-yeast_sugeno_weber.txt &
python3 seg_otim_cLambda.py yager 15 0 >> faltantes/results/saida-yeast_yager.txt &
python3 seg_otim_cLambda.py schweizer_skar 15 0 >> faltantes/results/saida-yeast_schweizer_skar.txt &

python3 seg_otim_sLambda.py media_aritmetic 8 >> faltantes/results/saida-Hayes-roth_media_aritmetic.txt &
python3 seg_otim_sLambda.py media_geometrica 8 >> faltantes/results/saida-Hayes-roth_media_geometrica.txt &
python3 seg_otim_sLambda.py media_harmonica 8 >> faltantes/results/saida-Hayes-roth_media_harmonica.txt &
python3 seg_otim_sLambda.py media_quadratica 8 >> faltantes/results/saida-Hayes-roth_media_quadratica.txt &
python3 seg_otim_sLambda.py mediana 8 >> faltantes/results/saida-Hayes-roth_mediana.txt &

python3 seg_otim_sLambda.py media_aritmetic 13 >> faltantes/results/saida-pima_media_aritmetic.txt &
python3 seg_otim_sLambda.py media_geometrica 13 >> faltantes/results/saida-pima_media_geometrica.txt &
python3 seg_otim_sLambda.py media_harmonica 13 >> faltantes/results/saida-pima_media_harmonica.txt &
python3 seg_otim_sLambda.py media_quadratica 13 >> faltantes/results/saida-pima_media_quadratica.txt &
python3 seg_otim_sLambda.py mediana 13 >> faltantes/results/saida-pima_mediana.txt &

python3 seg_otim_sLambda.py media_aritmetic 15 >> faltantes/results/saida-yeast_media_aritmetic.txt &
python3 seg_otim_sLambda.py media_geometrica 15 >> faltantes/results/saida-yeast_media_geometrica.txt &
python3 seg_otim_sLambda.py media_harmonica 15 >> faltantes/results/saida-yeast_media_harmonica.txt &
python3 seg_otim_sLambda.py media_quadratica 15 >> faltantes/results/saida-yeast_media_quadratica.txt &
python3 seg_otim_sLambda.py mediana 15 >> faltantes/results/saida-yeast_mediana.txt &

python3 seg_otim_sLambda.py minimo 15 >> faltantes/results/saida-yeast_minimo.txt &
python3 seg_otim_sLambda.py lukasiewicz 15 >> faltantes/results/saida-yeast_lukasiewicz.txt &
python3 seg_otim_sLambda.py produto_drastico 15 >> faltantes/results/saida-yeast_produto_drastico.txt &
python3 seg_otim_sLambda.py nilpotente_min 15 >> faltantes/results/saida-yeast_nilpotente_min.txt &
