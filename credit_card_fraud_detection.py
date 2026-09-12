import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
print(df.head())
print(df.info())

# Normalização
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df = df.drop(columns=['Amount', 'Time'])

# Dividindo os dados em treinamento e teste
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# modelo de regressão logística
model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
# Faz as previsões no conjunto de teste
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# metricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show() 

# curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Encontrando o Melhor Limiar com Matriz de Confusão
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 1. Varredura para encontrar o limiar que maximiza o F1-Score
limiares = np.linspace(0.5, 0.99, 50)
melhor_f1 = 0
melhor_limiar = 0.5

for limiar in limiares:
    preds = (y_prob >= limiar).astype(int)
    score = f1_score(y_test, preds, zero_division=0)
    if score > melhor_f1:
        melhor_f1 = score
        melhor_limiar = limiar

print(f"--- Ponto Ótimo Encontrado ---")
print(f"Melhor Limiar: {melhor_limiar:.3f}")
y_pred_otimo = (y_prob >= melhor_limiar).astype(int)
print(f"Precisão: {precision_score(y_test, y_pred_otimo):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_otimo):.4f}")
print(f"Novo F1-Score: {f1_score(y_test, y_pred_otimo):.4f}\n")

# 2. Plotagem lado a lado da Matriz de Confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz com corte padrão (0.50)
cm_padrao = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_padrao, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Limiar Padrão (0.50)')
axes[0].set_xlabel('Previsão')
axes[0].set_ylabel('Real')

# Matriz com corte otimizado
cm_otimo = confusion_matrix(y_test, y_pred_otimo)
sns.heatmap(cm_otimo, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title(f'Limiar Otimizado ({melhor_limiar:.2f})')
axes[1].set_xlabel('Previsão')
axes[1].set_ylabel('Real')

plt.tight_layout()
plt.show()