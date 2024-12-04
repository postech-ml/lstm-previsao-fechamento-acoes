# Usar imagem base Python 3.11 slim
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    GIT_PYTHON_REFRESH=quiet \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema e ferramentas de monitoramento
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    gnupg2 \
    apt-transport-https \
    software-properties-common \
    prometheus \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Copiar arquivos de requisitos e instalar dependências Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Criar diretórios necessários
RUN mkdir -p /app/logs /app/data /app/monitoring

# Configurar Prometheus
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Criar diretórios de dados para Prometheus
RUN mkdir -p /prometheus /var/lib/prometheus \
    && chown -R nobody:nogroup /prometheus /var/lib/prometheus

# Copiar arquivos da aplicação
COPY . /app/

# Configurar permissões
RUN chmod +x /app/start.sh \
    && chmod -R 777 /app/logs \
    && chmod -R 777 /app/data

# Configurar volumes
VOLUME ["/app/data", "/app/logs", "/var/lib/prometheus"]

# Expor portas
EXPOSE 5000 8000 9090

# Configurar healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Configurar entrypoint
ENTRYPOINT ["/app/start.sh"]

# Adicionar labels
LABEL maintainer="" \
      version="1.0" \
      description="Container para modelo LSTM com monitoramento" \
      org.opencontainers.image.source="https://github.com/postech-ml/lstm-previsao-fechamento-acoes"

# Definir argumentos para build
ARG BUILD_DATE
ARG VCS_REF

# Adicionar labels de metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/postech-ml/lstm-previsao-fechamento-acoes" \
      org.label-schema.schema-version="1.0"