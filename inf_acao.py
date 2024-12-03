import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

def get_stock_info(ticker):
    """Obter informações detalhadas da ação"""
    try:
        # Criar objeto Ticker
        stock = yf.Ticker(ticker)
        
        # Obter dados recentes
        dados_recentes = stock.history(period='5d')
        
        if dados_recentes.empty:
            raise ValueError("Não foi possível obter dados")
            
        # Obter último preço
        ultimo_preco = float(dados_recentes['Close'].iloc[-1])
        data_ultimo_preco = dados_recentes.index[-1]
        
        # Obter informações adicionais
        info = stock.info
        
        # Criar dicionário com informações relevantes
        stock_info = {
            'Ticker': ticker,
            'Data Último Preço': data_ultimo_preco.strftime('%Y-%m-%d'),
            'Último Preço': f"${ultimo_preco:.2f}",
            'Variação 5d': f"{((ultimo_preco - dados_recentes['Close'].iloc[0]) / dados_recentes['Close'].iloc[0] * 100):.2f}%",
            'Volume Médio (5d)': f"{dados_recentes['Volume'].mean():,.0f}",
            'Preço Máximo (5d)': f"${dados_recentes['High'].max():.2f}",
            'Preço Mínimo (5d)': f"${dados_recentes['Low'].min():.2f}"
        }
        
        # Adicionar informações do Yahoo Finance (se disponíveis)
        try:
            stock_info.update({
                'Nome Empresa': info.get('longName', 'N/A'),
                'Setor': info.get('sector', 'N/A'),
                'Indústria': info.get('industry', 'N/A'),
                'Market Cap': f"${info.get('marketCap', 0):,.2f}",
                'Volume Médio (3m)': f"{info.get('averageVolume3months', 0):,.0f}"
            })
        except:
            pass
        
        return stock_info, dados_recentes
        
    except Exception as e:
        print(f"Erro ao obter informações da ação: {e}")
        return None, None

def plot_recent_prices(dados_recentes, titulo):
    """Plotar gráfico dos preços recentes"""
    plt.figure(figsize=(12, 6))
    
    # Plotar preços
    plt.plot(dados_recentes.index, dados_recentes['Close'], 
            label='Preço de Fechamento', color='blue')
    
    # Destacar último preço
    plt.scatter(dados_recentes.index[-1], dados_recentes['Close'].iloc[-1],
               color='red', s=100, label='Último Preço')
    
    # Configurar gráfico
    plt.title(titulo, fontsize=14)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Rotacionar datas
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def save_report(stock_info, filename='relatorio_acao.txt'):
    """Salvar relatório em arquivo texto"""
    with open(filename, 'w') as f:
        f.write(f"Relatório da Ação - Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in stock_info.items():
            f.write(f"{key}: {value}\n")

def main(ticker='AMBA'):
    """Função principal"""
    print(f"\nVerificando informações para {ticker}...")
    
    # Obter informações
    stock_info, dados_recentes = get_stock_info(ticker)
    
    if stock_info is None:
        print("Não foi possível obter informações da ação.")
        return
    
    # Mostrar informações
    print("\nInformações da Ação:")
    print("=" * 50)
    for key, value in stock_info.items():
        print(f"{key}: {value}")
    
    # Criar e mostrar gráfico
    plt = plot_recent_prices(dados_recentes, 
                           f"Preços Recentes - {stock_info['Nome Empresa']} ({ticker})")
    
    # Salvar gráfico
    plt.savefig('ultimos_precos.png')
    
    # Salvar relatório
    save_report(stock_info)
    
    print("\nArquivos gerados:")
    print("- ultimos_precos.png (gráfico)")
    print("- stock_report.txt (relatório)")
    
    # Mostrar gráfico
    plt.show()

if __name__ == "__main__":
    # Você pode especificar um ticker diferente aqui
    main('AMBA')