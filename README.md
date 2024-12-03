# Previsão de Preços de Ações usando LSTM

Um aplicativo web que utiliza redes neurais LSTM (Long Short-Term Memory) para prever preços de fechamento de ações. O projeto foi desenvolvido especificamente para análise e previsão das ações da Ambarella Inc. (AMBA).

## Sobre o Projeto

### Autoria
**Grupo 60 - Pós Tech FIAP - Engenharia de Machine Learning**

### Contexto Acadêmico
Este projeto foi desenvolvido como parte do curso de Pós-Graduação em Engenharia de Machine Learning da FIAP (Faculdade de Informática e Administração Paulista), representando a aplicação prática dos conhecimentos adquiridos na Fase 4 - Deep Learning e IA.

## Funcionalidades

- Consulta de informações detalhadas de ações via Yahoo Finance
- Visualização de gráficos de preços históricos
- Previsão de preços futuros usando modelo LSTM
- Interface web intuitiva para interação com o sistema
- Painel de controle para treinamento do modelo
- Registro de previsões e métricas de desempenho
- Documentação da API com Swagger
- Comparação de períodos históricos diferentes

## Tecnologias Utilizadas

### Backend
- Python 3.11
- TensorFlow/Keras para modelagem LSTM
- MLflow para gerenciamento de experimentos
- Flask para servidor web
- Flasgger para documentação da API
- yfinance para dados de mercado
- Pandas e NumPy para manipulação de dados
- Matplotlib para visualizações

### Frontend
- HTML5/CSS3
- Bootstrap 5
- jQuery
- Font Awesome
- Gráficos interativos

### Infraestrutura
- Docker para containerização
- Gunicorn como servidor WSGI
- Git para controle de versão

## Pré-requisitos e Considerações

### Verificação de Porta

Antes de iniciar o sistema, verifique se a porta 5000 não está em uso:

#### Windows
```bash
netstat -ano | findstr :5000
```

#### Linux/Mac
```bash
sudo lsof -i :5000
```

Se a porta 5000 estiver em uso, você pode:
1. Encerrar o processo que está usando a porta:
   - Windows: `taskkill /PID [número_do_processo] /F`
   - Linux/Mac: `sudo kill -9 [número_do_processo]`

2. Ou alterar a porta do aplicativo:
   - No arquivo `app.py`, modifique a linha:
     ```python
     app.run(host='0.0.0.0', port=5000, debug=False)
     ```
   - Para usar outra porta, por exemplo 5001:
     ```python
     app.run(host='0.0.0.0', port=5001, debug=False)
     ```
   - Se estiver usando Docker, modifique também o Dockerfile e o comando de execução do container.

### Outros Pré-requisitos

- Python 3.11 ou superior instalado
- Git para clonar o repositório
- Pip para instalação de dependências
- Docker (opcional, para execução containerizada)
- Memória RAM mínima recomendada: 4GB
- Espaço em disco: mínimo 1GB livre

## Instalação

### Usando Docker

1. Clone o repositório:
```bash
git clone https://github.com/cleberdevs/lstm-previsao-fechamento-acoes.git
cd lstm-previsao-fechamento-acoes
```

2. Construa a imagem Docker:
```bash
docker build -t lstm-previsao-acoes .
```

3. Execute o container:
```bash
docker run -p 5000:5000 lstm-previsao-acoes
```

### Instalação Local

1. Clone o repositório e instale as dependências:
```bash
git clone https://github.com/cleberdevs/lstm-previsao-fechamento-acoes.git
cd lstm-previsao-fechamento-acoes
pip install -r requirements.txt
```

2. Execute o treinamento do modelo:
```bash
python criacao_modelo.py
```

3. Inicie o servidor web:
```bash
python app.py
```

## Estrutura do Projeto

```
.
├── app.py                 # Servidor Flask e endpoints da API
├── criacao_modelo.py      # Script para treinar o modelo LSTM
├── previsao_fechamento_acao.py  # Lógica de previsão
├── inf_acao.py           # Funções para obter informações das ações
├── comparacao_periodos.py # Análise comparativa de períodos
├── requirements.txt      # Dependências do projeto
├── Dockerfile           # Configuração do container
├── start.sh            # Script de inicialização
└── templates/          # Templates HTML
    ├── index.html     # Página principal
    └── treinamento_modelo.html  # Painel de treinamento
```

## API Endpoints

- `GET /`: Página principal
- `GET /treinamentomodelo`: Painel de treinamento
- `GET /docs`: Documentação Swagger da API
- `POST /obter_info_acao`: Obtém informações da ação
- `POST /fazer_previsao`: Realiza previsão de preço
- `POST /treinamentomodelo/treinar`: Inicia treinamento
- `GET /treinamentomodelo/status`: Status do treinamento
- `GET /treinamentomodelo/zipar-pasta`: Zipar a pasta do modelo
- `GET /treinamentomodelo/download`: Download da pasta zipada do modelo

## Uso

1. Acesse a interface web principal em `http://localhost:5000`

2. Para acessar diferentes páginas do sistema:
   - Painel de Treinamento: `http://localhost:5000/treinamentomodelo`
   - Documentação da API: `http://localhost:5000/docs`

3. Na página principal:
   - Use o ticker "AMBA" para visualizar informações da ação
   - Acompanhe gráficos históricos
   - Faça previsões de preço

4. No painel de treinamento (`/treinamentomodelo`):
   - Inicie novo treinamento do modelo
   - Acompanhe o status do treinamento em tempo real
   - Visualize métricas de desempenho
   - Veja gráficos comparativos de treino e teste
   - Baixar a pasta zipado do modelo gerado

5. Na documentação da API (`/docs`):
   - Explore todos os endpoints disponíveis
   - Teste as APIs diretamente pela interface Swagger
   - Consulte parâmetros e respostas esperadas

## Limitações

- Modelo treinado especificamente para ações da AMBA
- Previsões baseadas apenas em preços históricos
- Desempenho pode variar em períodos de alta volatilidade
- Necessário retreinamento periódico do modelo

## Métricas e Avaliação

O modelo é avaliado usando:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Visualização de previsões vs valores reais
- Histórico de previsões pode ser monitorado através do arquivo `historico_previsoes.csv`


