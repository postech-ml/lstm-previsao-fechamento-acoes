FROM python:3.11-slim

WORKDIR /app

# Copiar os requisitos e instalar dependências do sistema
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos do projeto para o container
COPY . /app

# Adicionar o script de inicialização e torná-lo executável
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Configurar variáveis de ambiente para resolver warnings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV GIT_PYTHON_REFRESH=quiet

# Expor a porta para a API
EXPOSE 5000

# Usar o script como ponto de entrada
ENTRYPOINT ["/app/start.sh"]
