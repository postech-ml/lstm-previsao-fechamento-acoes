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

app = Flask(__name__)

# Caminho da pasta que será zipada
FOLDER_TO_ZIP       = 'mlruns'
FOLDER_TO_SAVE_ZIP  = 'Modelos_Grupo_60'
ZIP_FILE_NAME       = FOLDER_TO_SAVE_ZIP + '.zip'

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
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=template)

# Status do treinamento
training_status = {
    "is_running": False,
    "start_time": None,
    "end_time": None,
    "run_id": None,
    "error": None,
    "metrics": None
}

def execute_model_training():
    """Executa o treinamento do modelo"""
    global training_status
    
    try:
        training_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_status["is_running"] = True
        
        result = subprocess.run(
            [sys.executable, 'criacao_modelo.py'],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if "O run_id é:" in line:
                training_status["run_id"] = line.split(":")[1].strip()
            elif "Métricas de Avaliação:" in line:
                metrics_lines = result.stdout.split('\n')[result.stdout.split('\n').index(line)+1:result.stdout.split('\n').index(line)+3]
                training_status["metrics"] = {
                    'train': metrics_lines[0].split("Treino -")[1].strip(),
                    'test': metrics_lines[1].split("Teste -")[1].strip()
                }
        
        if result.returncode != 0:
            raise Exception(f"Erro no treinamento: {result.stderr}")
        
        training_status["error"] = None
        
    except Exception as e:
        training_status["error"] = str(e)
    finally:
        training_status["is_running"] = False
        training_status["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route('/')
def pagina_inicial():
    """Renderiza a página inicial"""
    return render_template('index.html')

@app.route('/zipar-pasta', methods=['GET'])
def zip_folder():
    try:
        # Apaga o arquivo zip antigo, se existir
        if os.path.exists(ZIP_FILE_NAME):
            os.remove(ZIP_FILE_NAME)

        # Cria o arquivo zip
        shutil.make_archive(FOLDER_TO_SAVE_ZIP, 'zip', FOLDER_TO_ZIP)
        
        return jsonify({"zipFileName": ZIP_FILE_NAME}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/download/<file_name>', methods=['GET'])
def download_file(file_name):
    try:
        # Caminho completo do arquivo
        file_path = os.path.join(os.getcwd(), file_name)

        if not os.path.exists(file_path):
            return jsonify({"message": "Arquivo não encontrado"}), 404

        # Envia o arquivo para download
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/obter_info_acao', methods=['POST'])
@swag_from({
    'tags': ['ações'],
    'summary': 'Obtém informações de uma ação',
    'description': 'Retorna informações detalhadas e gráfico de uma ação específica',
    'parameters': [
        {
            'name': 'ticker',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'default': 'AMBA',
            'description': 'Código da ação'
        }
    ],
    'responses': {
        200: {
            'description': 'Informações obtidas com sucesso'
        },
        400: {
            'description': 'Erro na requisição'
        }
    }
})
def obter_informacoes_acao():
    ticker = request.form.get('ticker', 'AMBA')
    stock_info, dados_recentes = get_stock_info(ticker)
    
    if stock_info is None:
        return jsonify({'error': 'Não foi possível obter informações da ação'}), 400
    
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

@app.route('/fazer_previsao', methods=['POST'])
@swag_from({
    'tags': ['ações'],
    'summary': 'Realiza previsão de preço',
    'description': 'Faz uma previsão do próximo preço da ação usando o modelo LSTM',
    'parameters': [
        {
            'name': 'ticker',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'default': 'AMBA',
            'description': 'Código da ação'
        }
    ],
    'responses': {
        200: {
            'description': 'Previsão realizada com sucesso'
        },
        400: {
            'description': 'Erro na requisição'
        }
    }
})
def fazer_previsao_acao():
    try:
        ticker = request.form.get('ticker', 'AMBA')
        sequence_length = int(request.form.get('sequence_length', 60))
        
        prediction, ultimo_preco, variacao = make_prediction()
        
        if prediction is None:
            return jsonify({'error': 'Erro ao fazer previsão'}), 400
        
        img = BytesIO()
        matplotlib.pyplot.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        matplotlib.pyplot.close()
        
        return jsonify({
            'prediction': f"${prediction:.2f}",
            'ultimo_preco': f"${ultimo_preco:.2f}",
            'variacao': f"{variacao:.2f}%",
            'graph': graph_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/treinamentomodelo')
def painel_treinamento():
    """Renderiza o painel de treinamento"""
    return render_template('treinamento_modelo.html')

@app.route('/treinamentomodelo/treinar', methods=['POST'])
@swag_from({
    'tags': ['treinamento'],
    'summary': 'Inicia treinamento do modelo',
    'description': 'Inicia um novo processo de treinamento do modelo LSTM',
    'responses': {
        200: {
            'description': 'Treinamento iniciado com sucesso'
        },
        400: {
            'description': 'Já existe um treinamento em andamento'
        }
    }
})
def treinar_modelo():
    global training_status
    
    if training_status["is_running"]:
        return jsonify({
            "status": "erro",
            "message": "Já existe um treinamento em andamento",
            "start_time": training_status["start_time"]
        }), 400
    
    training_status = {
        "is_running": False,
        "start_time": None,
        "end_time": None,
        "run_id": None,
        "error": None,
        "metrics": None
    }
    
    thread = threading.Thread(target=execute_model_training)
    thread.start()
    
    return jsonify({
        "status": "iniciado",
        "message": "Treinamento iniciado com sucesso",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/treinamentomodelo/status', methods=['GET'])
@swag_from({
    'tags': ['treinamento'],
    'summary': 'Obtém status do treinamento',
    'description': 'Retorna o status atual do processo de treinamento',
    'responses': {
        200: {
            'description': 'Status obtido com sucesso'
        }
    }
})
def obter_status_treinamento():
    status_data = training_status.copy()
    
    # Se o treinamento foi concluído e o arquivo existe, adiciona o gráfico
    if not status_data["is_running"] and status_data["end_time"] and os.path.exists('previsoes_completas.png'):
        with open('previsoes_completas.png', 'rb') as f:
            graph_data = base64.b64encode(f.read()).decode()
            status_data['graph'] = graph_data
    else:
        status_data['graph'] = None
        
    return jsonify(status_data)

@app.route('/treinamentomodelo/saude', methods=['GET'])
@swag_from({
    'tags': ['treinamento'],
    'summary': 'Verifica saúde da API',
    'description': 'Verifica se a API está funcionando corretamente',
    'responses': {
        200: {
            'description': 'API está saudável'
        }
    }
})
def verificar_saude():
    return jsonify({
        "status": "saudavel",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    # Certifique-se de que a pasta que você quer zipar existe
    if not os.path.exists(FOLDER_TO_ZIP):
        os.makedirs(FOLDER_TO_ZIP)
        with open(os.path.join(FOLDER_TO_ZIP, "exemplo.txt"), "w") as f:
            f.write("Este é um exemplo de arquivo.")
    app.run(host='0.0.0.0', port=5000, debug=False)    