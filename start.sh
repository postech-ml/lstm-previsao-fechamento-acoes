#!/bin/bash

echo "Iniciando treinamento do modelo..."
python criacao_modelo.py

if [ $? -eq 0 ]; then
    echo "Treinamento concluído com sucesso. Iniciando a API Flask..."
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
else
    echo "Erro durante o treinamento do modelo. Encerrando o container."
    exit 1
fi
