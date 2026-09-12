# Detecção de Fraude em Cartão de Crédito

Este projeto implementa um modelo de **Regressão Logística** para identificar transações fraudulentas utilizando a base de dados pública do TensorFlow.

## 🚀 Como Executar o Projeto

### 1. Instalar as Dependências
Abra o terminal na pasta do projeto e instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

### 2. Executar o Script
Rode o arquivo principal para treinar o modelo e visualizar os gráficos de avaliação (Matriz de Confusão e Curva ROC):
```bash
python credit_card_fraud_detection.py
```

## 📊 Métricas Avaliadas
O script calcula automaticamente:
- Acurácia (Accuracy)
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- Área sob a Curva ROC (AUC-ROC)
