import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

def create_model():
    """Criar modelo LSTM com janela de 60 dias"""
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, sequence_length=60):
    """Preparar dados com janela de 60 dias"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def evaluate_period(ticker, start_date, end_date=datetime.now().strftime('%Y-%m-%d')):
    """Avaliar performance do modelo para um período específico"""
    try:
        # Baixar dados
        dados = yf.download(ticker, start=start_date, end=end_date)
        
        if len(dados) == 0:
            print(f"Nenhum dado encontrado para o período {start_date} a {end_date}")
            return None
            
        print(f"\nPeríodo: {start_date} até {end_date}")
        print(f"Número de dias: {len(dados)}")
        print(f"Primeiro preço: ${float(dados['Close'].iloc[0]):.2f}")
        print(f"Último preço: ${float(dados['Close'].iloc[-1]):.2f}")
        
        # Verificar se há dados suficientes
        if len(dados) < 120:
            print(f"Dados insuficientes para o período {start_date} a {end_date}")
            return None
        
        # Preparar dados
        data = dados['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Dividir em treino e teste
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Preparar sequências
        X_train, y_train = prepare_data(train_data)
        X_test, y_test = prepare_data(test_data)
        
        # Verificar se há dados suficientes após a preparação
        if len(X_train) < 60 or len(X_test) < 1:
            print(f"Sequências insuficientes para o período {start_date} a {end_date}")
            return None
        
        print("Treinando modelo...")
        # Treinar modelo
        model = create_model()
        history = model.fit(X_train, y_train, 
                          epochs=10, 
                          batch_size=32, 
                          validation_split=0.1,
                          verbose=0)
        
        # Fazer previsões
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)
        
        # Desnormalizar
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_inv = scaler.inverse_transform([y_train])
        y_test_inv = scaler.inverse_transform([y_test])
        
        # Calcular métricas
        metrics = {
            'period': start_date,
            'train_mae': mean_absolute_error(y_train_inv.T, train_predict),
            'train_rmse': np.sqrt(mean_squared_error(y_train_inv.T, train_predict)),
            'test_mae': mean_absolute_error(y_test_inv.T, test_predict),
            'test_rmse': np.sqrt(mean_squared_error(y_test_inv.T, test_predict)),
            'data_points': len(dados),
            'primeiro_preco': float(dados['Close'].iloc[0]),
            'ultimo_preco': float(dados['Close'].iloc[-1])
        }
        
        print("Avaliação concluída com sucesso")
        return metrics
        
    except Exception as e:
        print(f"Erro ao avaliar período {start_date} a {end_date}: {str(e)}")
        return None

def compare_periods(ticker='AMBA'):
    """Comparar diferentes períodos históricos"""
    data_atual = datetime.now().strftime('%Y-%m-%d')
    
    # Definir períodos com datas fixas
    periods = {
        '2 anos': '2022-01-01',
        '3 anos': '2021-01-01',
        '4 anos': '2020-01-01',
        '5 anos': '2019-01-01'
    }
    
    print(f"Análise para {ticker}")
    print(f"Data final: {data_atual}")
    
    results = {}
    for period_name, start_date in periods.items():
        print(f"\nAvaliando período: {period_name}")
        metrics = evaluate_period(ticker, start_date, data_atual)
        if metrics is not None:
            results[period_name] = metrics
    
    if not results:
        print("Nenhum período forneceu dados suficientes para análise")
        return
    
    # Plotar resultados
    plt.figure(figsize=(15, 10))
    
    # Preparar dados para plotagem
    periods = list(results.keys())
    train_mae = [results[p]['train_mae'] for p in periods]
    test_mae = [results[p]['test_mae'] for p in periods]
    train_rmse = [results[p]['train_rmse'] for p in periods]
    test_rmse = [results[p]['test_rmse'] for p in periods]
    
    # Plot MAE
    plt.subplot(2, 1, 1)
    plt.plot(periods, train_mae, 'b-o', label='Treino MAE')
    plt.plot(periods, test_mae, 'r-o', label='Teste MAE')
    plt.title(f'MAE por Período Histórico - {ticker}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot RMSE
    plt.subplot(2, 1, 2)
    plt.plot(periods, train_rmse, 'b-o', label='Treino RMSE')
    plt.plot(periods, test_rmse, 'r-o', label='Teste RMSE')
    plt.title('RMSE por Período Histórico')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Imprimir resultados
    print("\nResultados Detalhados:")
    print("\nPeríodo    | Dias  | Primeiro $ | Último $ | Treino MAE | Teste MAE | RMSE")
    print("-" * 80)
    for period in results:
        metrics = results[period]
        print(f"{period:10} | {metrics['data_points']:5d} | "
              f"${metrics['primeiro_preco']:9.2f} | "
              f"${metrics['ultimo_preco']:8.2f} | "
              f"{metrics['train_mae']:9.2f} | "
              f"{metrics['test_mae']:8.2f} | "
              f"{metrics['test_rmse']:8.2f}")
    
    # Identificar melhor período
    best_period = min(results.items(), key=lambda x: x[1]['test_mae'])
    print(f"\nMelhor período baseado no Teste MAE: {best_period[0]}")
    print(f"MAE do teste: ${best_period[1]['test_mae']:.2f}")
    
    plt.savefig('comparacao_periodos_amba.png')
    plt.show()

if __name__ == "__main__":
    compare_periods()