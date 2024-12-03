import mlflow
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_latest_model():
    """Encontrar o modelo mais recente no MLflow"""
    mlflow.set_tracking_uri('file:' + os.path.join(os.getcwd(), 'mlruns'))
    client = mlflow.tracking.MlflowClient()
    
    # Listar todos os experimentos
    experiments = client.search_experiments()
    latest_run = None
    latest_timestamp = 0
    
    for experiment in experiments:
        # Buscar as runs do experimento
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs and runs[0].info.start_time > latest_timestamp:
            latest_timestamp = runs[0].info.start_time
            latest_run = runs[0]
    
    if latest_run is None:
        raise Exception("Nenhum modelo encontrado")
        
    return latest_run.info.run_id

def prepare_data_for_prediction(ticker, sequence_length=60):
    """Preparar dados para previsão"""
    try:
        # Baixar dados históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=sequence_length + 30)
        
        # Configurar ticker
        ticker_obj = yf.Ticker(ticker)
        dados = ticker_obj.history(start=start_date, end=end_date)
        
        if len(dados) < sequence_length:
            raise ValueError(f"Dados insuficientes. Necessário {sequence_length} dias.")
        
        # Pegar os últimos 60 dias
        close_prices = dados['Close'].values[-sequence_length:].reshape(-1, 1)
        
        # Normalizar dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Preparar input para o modelo
        X = scaled_data.reshape(1, sequence_length, 1)
        
        return X, scaler, dados
        
    except Exception as e:
        print(f"Erro ao preparar dados: {e}")
        raise

def make_prediction():
    try:
        # Configurações
        ticker = 'AMBA'
        sequence_length = 60
        
        # Obter o modelo mais recente
        run_id = get_latest_model()
        print(f"Usando modelo do run_id: {run_id}")
        
        # Carregar o modelo
        model_path = f"runs:/{run_id}/modelo_lstm"
        model = mlflow.keras.load_model(model_path)
        
        # Preparar dados
        X, scaler, dados = prepare_data_for_prediction(ticker, sequence_length)
        
        # Fazer previsão
        prediction_scaled = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Obter último preço conhecido
        ultimo_preco = float(dados['Close'].iloc[-1])
        
        # Calcular variação percentual
        variacao = ((prediction - ultimo_preco) / ultimo_preco) * 100
        
        # Plotar gráfico
        plt.figure(figsize=(15, 7))
        
        # Plotar histórico recente
        plt.plot(dados.index[-30:], dados['Close'][-30:], 
                label='Histórico Recente', color='blue')
        
        # Plotar previsão
        proxima_data = dados.index[-1] + timedelta(days=1)
        plt.scatter(proxima_data, prediction, 
                   color='red', s=100, label='Previsão')
        
        plt.title(f'Previsão de Preço para {ticker}', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço ($)', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Adicionar informações
        info_text = (f'Último preço: ${ultimo_preco:.2f}\n'
                    f'Previsão: ${prediction:.2f}\n'
                    f'Variação: {variacao:.2f}%')
        
        plt.figtext(0.01, 0.01, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Salvar gráfico
        plt.savefig('previsao_atual.png')
        
        # Imprimir resultados
        print("\nResultados da Previsão:")
        print(f"Data da previsão: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Último preço conhecido: ${ultimo_preco:.2f}")
        print(f"Previsão para próximo dia útil: ${prediction:.2f}")
        print(f"Variação esperada: {variacao:.2f}%")
        
        # Mostrar gráfico
        plt.show()
        
        return prediction, ultimo_preco, variacao
        
    except Exception as e:
        print(f"Erro ao fazer previsão: {e}")
        return None, None, None

def save_prediction_results(prediction, ultimo_preco, variacao):
    """Salvar resultados da previsão"""
    try:
        data_atual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        resultados = {
            'Data': [data_atual],
            'Último Preço': [f"${ultimo_preco:.2f}"],
            'Previsão': [f"${prediction:.2f}"],
            'Variação (%)': [f"{variacao:.2f}%"]
        }
        
        df = pd.DataFrame(resultados)
        
        # Salvar em CSV
        df.to_csv('historico_previsoes.csv', 
                 mode='a', 
                 header=not os.path.exists('historico_previsoes.csv'),
                 index=False)
        
        print("\nResultados salvos em 'historico_previsoes.csv'")
        
    except Exception as e:
        print(f"Erro ao salvar resultados: {e}")

if __name__ == "__main__":
    try:
        # Fazer previsão
        prediction, ultimo_preco, variacao = make_prediction()
        
        if all(v is not None for v in [prediction, ultimo_preco, variacao]):
            # Salvar resultados
            save_prediction_results(prediction, ultimo_preco, variacao)
            
            # Mostrar histórico
            if os.path.exists('historico_previsoes.csv'):
                historico = pd.read_csv('historico_previsoes.csv')
                print("\nHistórico de Previsões:")
                print(historico.to_string(index=False))
                
    except Exception as e:
        print(f"Erro na execução: {e}")