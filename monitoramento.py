import time
import psutil
import logging
from functools import wraps
from datetime import datetime
import pandas as pd
import os
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configurar logging
logging.basicConfig(
    filename='model_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Métricas Prometheus
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')
PREDICTION_COUNTER = Counter('prediction_total', 'Total number of predictions')
PREDICTION_ERROR_COUNTER = Counter('prediction_errors_total', 'Total number of prediction errors')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Current memory usage')
CPU_USAGE = Gauge('cpu_usage_percent', 'Current CPU usage')

class ModelMonitor:
    def __init__(self):
        self.predictions_log = []
        self.performance_metrics = []
        
    def log_prediction(self, prediction_data):
        """Registra dados de uma previsão"""
        timestamp = datetime.now()
        self.predictions_log.append({
            'timestamp': timestamp,
            'prediction': prediction_data['prediction'],
            'actual_value': prediction_data.get('actual_value'),
            'latency': prediction_data['latency'],
            'memory_usage': prediction_data['memory_usage'],
            'cpu_usage': prediction_data['cpu_usage']
        })
        
        # Atualizar métricas Prometheus
        PREDICTION_COUNTER.inc()
        MEMORY_USAGE.set(prediction_data['memory_usage'])
        CPU_USAGE.set(prediction_data['cpu_usage'])
        
        # Salvar logs em CSV
        self._save_logs()
        
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.predictions_log:
            return {}
            
        df = pd.DataFrame(self.predictions_log)
        
        metrics = {
            'avg_latency': df['latency'].mean(),
            'max_latency': df['latency'].max(),
            'avg_memory_usage': df['memory_usage'].mean(),
            'avg_cpu_usage': df['cpu_usage'].mean(),
            'total_predictions': len(df),
            'timestamp': datetime.now()
        }
        
        # Calcular acurácia se houver valores reais
        if 'actual_value' in df.columns and not df['actual_value'].isna().all():
            mse = ((df['prediction'] - df['actual_value'])**2).mean()
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            
            # Atualizar métrica de acurácia no Prometheus
            MODEL_ACCURACY.set(1 - mse)
            
        self.performance_metrics.append(metrics)
        return metrics
    
    def _save_logs(self):
        """Salva logs em arquivo CSV"""
        df = pd.DataFrame(self.predictions_log)
        df.to_csv('prediction_logs.csv', index=False)
        
        metrics_df = pd.DataFrame(self.performance_metrics)
        metrics_df.to_csv('performance_metrics.csv', index=False)

# Criar instância global do monitor
model_monitor = ModelMonitor()

def monitor_prediction(func):
    """Decorator para monitorar previsões"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Executar previsão
            result = func(*args, **kwargs)
            
            # Coletar métricas
            end_time = time.time()
            latency = end_time - start_time
            
            prediction_data = {
                'prediction': result[0] if isinstance(result, tuple) else result,
                'latency': latency,
                'memory_usage': psutil.Process().memory_info().rss,
                'cpu_usage': psutil.cpu_percent()
            }
            
            # Registrar métricas
            model_monitor.log_prediction(prediction_data)
            PREDICTION_LATENCY.observe(latency)
            
            return result
            
        except Exception as e:
            PREDICTION_ERROR_COUNTER.inc()
            logging.error(f"Erro na previsão: {str(e)}")
            raise
            
    return wrapper

def start_monitoring_server(port=8000):
    """Inicia servidor Prometheus"""
    start_http_server(port)
    logging.info(f"Servidor de monitoramento iniciado na porta {port}")

def get_resource_usage():
    """Retorna uso atual de recursos"""
    return {
        'memory_usage': psutil.Process().memory_info().rss,
        'cpu_usage': psutil.cpu_percent(),
        'disk_usage': psutil.disk_usage('/').percent
    }