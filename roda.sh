#!/usr/bin/env bash

python3 seg_otim_sLambda.py produto 9 >> produto.txt &

python3 seg_otim_cLambda.py produto 9 0 >> prod.txt &