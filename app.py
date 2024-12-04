from flask import Flask, render_template, request, jsonify, send_file
from flasgger import Swagger, swag_from
from previsao_fechamento_acao import prepare_data_for_prediction, make_prediction, get_latest_model
from inf_acao import get_stock_info, plot_recent_prices
import sys
import os
import mlflow
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import threading
import subprocess
from datetime import datetime
import shutil
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from monitoramento import (
    ModelMonitor, 
    monitor_prediction, 
    get_resource_usage,
    PREDICTION_COUNTER, 
    PREDICTION_LATENCY,
    MODEL_ACCURACY
)
import logging
from functools import wraps
import time

# Configurar logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Instanciar monitor
model_monitor = ModelMonitor()

# Métricas adicionais
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP request latency', ['endpoint'])
ERROR_COUNTER = Counter('http_request_errors_total', 'Total HTTP request errors', ['endpoint'])
ACTIVE_REQUESTS = Gauge('http_requests_active', 'Number of active HTTP requests')

# Caminho da pasta que será zipada
FOLDER_TO_ZIP = 'mlruns'
FOLDER_TO_SAVE_ZIP = 'Modelos_Grupo_60'
ZIP_FILE_NAME = FOLDER_TO_SAVE_ZIP + '.zip'

# Configuração do Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

template = {
    "swagger": "2.0",
    "info": {
        "title": "API de Treinamento e Previsão de Ações",
        "description": "API para treinamento de modelo LSTM e previsão de preços de ações",
        "version": "1.0.0",
        "contact": {
            "email": "seu.email@exemplo.com"
        }
    },
    "tags": [
        {
            "name": "ações",
            "description": "Operações relacionadas a ações"
        },
        {
            "name": "treinamento",
            "description": "Operações relacionadas ao treinamento do modelo"
        },
        {
            "name": "monitoramento",
            "description": "Endpoints de monitoramento e métricas"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=template)

# Decorator para monitorar endpoints
def monitor_endpoint(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        endpoint = request.endpoint
        
        try:
            response = f(*args, **kwargs)
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
            return response
        except Exception as e:
            ERROR_COUNTER.labels(endpoint=endpoint).inc()
            logger.error(f"Erro no endpoint {endpoint}: {str(e)}")
            raise
        finally:
            ACTIVE_REQUESTS.dec()
            
    return decorated_function

# Status do treinamento
training_status = {
    "is_running": False,
    "start_time": None,
    "end_time": None,
    "run_id": None,
    "error": None,
    "metrics": None
}

@app.route('/health')
@monitor_endpoint
def health_check():
    """Verificar saúde da aplicação"""
    try:
        # Verificar conexão com MLflow
        mlflow.get_tracking_uri()
        
        # Verificar recursos do sistema
        resources = get_resource_usage()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'resources': resources,
            'active_requests': ACTIVE_REQUESTS._value.get(),
        })
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/metrics/model')
@monitor_endpoint
def model_metrics():
    """Retornar métricas do modelo"""
    try:
        metrics = model_monitor.calculate_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Erro ao obter métricas do modelo: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics/system')
@monitor_endpoint
def system_metrics():
    """Retornar métricas do sistema"""
    try:
        return jsonify(get_resource_usage())
    except Exception as e:
        logger.error(f"Erro ao obter métricas do sistema: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
@monitor_endpoint
def pagina_inicial():
    """Renderizar a página inicial"""
    return render_template('index.html')

@app.route('/obter_info_acao', methods=['POST'])
@monitor_endpoint
@swag_from({
    'tags': ['ações'],
    'summary': 'Obtém informações de uma ação',
    'parameters': [
        {
            'name': 'ticker',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'default': 'AMBA'
        }
    ]
})
def obter_informacoes_acao():
    try:
        ticker = request.form.get('ticker', 'AMBA')
        stock_info, dados_recentes = get_stock_info(ticker)
        
        if stock_info is None:
            raise ValueError('Não foi possível obter informações da ação')
        
        plt = plot_recent_prices(dados_recentes, 
                               f"Preços Recentes - {stock_info['Nome Empresa']} ({ticker})")
        
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'stock_info': stock_info,
            'graph': graph_url
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter informações da ação: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/fazer_previsao', methods=['POST'])
@monitor_endpoint
@monitor_prediction
@swag_from({
    'tags': ['ações'],
    'summary': 'Realiza previsão de preço'
})
def fazer_previsao_acao():
    try:
        prediction, ultimo_preco, variacao = make_prediction()
        
        if prediction is None:
            raise ValueError('Erro ao fazer previsão')
        
        # Registrar previsão no monitor
        prediction_data = {
            'prediction': prediction,
            'timestamp': datetime.now(),
            'latency': time.time() - request.start_time,
            'memory_usage': psutil.Process().memory_info().rss,
            'cpu_usage': psutil.cpu_percent()
        }
        model_monitor.log_prediction(prediction_data)
        
        return jsonify({
            'prediction': f"${prediction:.2f}",
            'ultimo_preco': f"${ultimo_preco:.2f}",
            'variacao': f"{variacao:.2f}%"
        })
        
    except Exception as e:
        logger.error(f"Erro ao fazer previsão: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/treinamentomodelo')
@monitor_endpoint
def painel_treinamento():
    """Renderizar o painel de treinamento"""
    return render_template('treinamento_modelo.html')

@app.route('/treinamentomodelo/treinar', methods=['POST'])
@monitor_endpoint
@swag_from({
    'tags': ['treinamento'],
    'summary': 'Inicia treinamento do modelo'
})
def treinar_modelo():
    global training_status
    
    if training_status["is_running"]:
        return jsonify({
            "status": "erro",
            "message": "Já existe um treinamento em andamento"
        }), 400
    
    try:
        thread = threading.Thread(target=execute_model_training)
        thread.start()
        
        return jsonify({
            "status": "iniciado",
            "message": "Treinamento iniciado com sucesso",
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Erro ao iniciar treinamento: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Iniciar servidor de métricas do Prometheus
def start_metrics_server():
    start_http_server(8000)
    logger.info("Servidor de métricas iniciado na porta 8000")

if __name__ == '__main__':
    # Iniciar servidor de métricas em thread separada
    metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
    metrics_thread.start()
    
    # Certificar-se de que a pasta que você quer zipar existe
    if not os.path.exists(FOLDER_TO_ZIP):
        os.makedirs(FOLDER_TO_ZIP)
    
    # Iniciar aplicação Flask
    app.run(host='0.0.0.0', port=5000, debug=False)