import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Definindo a estrutura do modelo bayesiano
model = BayesianNetwork([
    ('HistoricoVendas', 'Demanda'),
    ('ComportamentoConsumidor', 'Demanda'),
    ('InformacoesSazonais', 'Demanda'),
    ('Demanda', 'Estoque'),
    ('Demanda', 'Compra'),
    ('Estoque', 'Envio'),
    ('Estoque', 'Armazenamento'),
])

# Dados de treinamento fictícios
data = {
    'HistoricoVendas': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    'ComportamentoConsumidor': [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'InformacoesSazonais': [0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    'Demanda': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    'Estoque': [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'Compra': [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'Envio': [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    'Armazenamento': [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
}

# Convertendo o dicionário para DataFrame
df = pd.DataFrame(data)

# Estimando as probabilidades usando o estimador bayesiano
model.fit(df, estimator=BayesianEstimator)

# Consulta: Dado o histórico de vendas, comportamento do consumidor e informações sazonais, faça previsões
inference = VariableElimination(model)

# Função para consultar o modelo bayesiano
def fazer_previsao(historico_vendas, comportamento_consumidor, informacoes_sazonais, evidencias={}):
    evidencias['HistoricoVendas'] = historico_vendas
    evidencias['ComportamentoConsumidor'] = comportamento_consumidor
    evidencias['InformacoesSazonais'] = informacoes_sazonais
    
    # Print para debug
    print("Evidências:", evidencias)
    
    previsao = inference.query(variables=['Demanda', 'Compra'], evidence=evidencias)
    
    # Print para debug
    print("Resultado da Consulta:", previsao)
    
    return previsao

# Exemplo de previsão: histórico de vendas é alto (1), comportamento do consumidor é positivo (1), e há informações sazonais (1)
resultado_previsao = fazer_previsao(historico_vendas=1, comportamento_consumidor=1, informacoes_sazonais=1)

# Verifica se a consulta foi bem-sucedida antes de acessar os valores
if 'Demanda' in resultado_previsao.values and 'Compra' in resultado_previsao.values:
    print("Previsão de Demanda:", resultado_previsao.values['Demanda'])
    print("Previsão de Compra:", resultado_previsao.values['Compra'])
else:
    print("Consulta sem resultados ou estrutura inesperada.")