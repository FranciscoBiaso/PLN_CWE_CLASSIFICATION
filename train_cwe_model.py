import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import gensim
from gensim.models import Word2Vec
import seaborn as sns
import matplotlib.pyplot as plt

# ANSI escape codes for colors
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"

# Set up logging with colored output
class ColoredFormatter(logging.Formatter):
    """Custom formatter for adding color to logs."""
    def format(self, record):
        log_message = super().format(record)
        if record.levelno == logging.INFO:
            return f"{BLUE}{log_message}{RESET}"
        elif record.levelno == logging.WARNING:
            return f"{YELLOW}{log_message}{RESET}"
        elif record.levelno == logging.ERROR:
            return f"{RED}{log_message}{RESET}"
        return log_message

# Set up logger with colored output
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_data():
    with open("cwe_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("cwe_labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open("cwe_label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    logger.info(f"Dados carregados: {len(data)} exemplos")
    return data, labels, label_map

# Função para carregar o modelo Word2Vec previamente treinado
def load_word2vec_model(model_path="word2vec.model"):
    """
    Carrega o modelo Word2Vec treinado previamente.
    """
    model = Word2Vec.load(model_path)
    logger.info(f"{GREEN}Modelo Word2Vec carregado com sucesso!{RESET}")
    return model

# Função para vetorização dos dados utilizando o modelo Word2Vec carregado
def vectorize_data_with_trained_word2vec(data, model):
    """
    Vetoriza os dados usando um modelo Word2Vec carregado.
    """
    logger.info(f"{YELLOW}Vetorizando os dados com o modelo Word2Vec carregado...{RESET}")
    
    # Extrair sentenças de código processadas
    sentences = [entry["code"] for entry in data]

    # Criar vetores médios para cada trecho de código
    X = []
    for sentence in sentences:
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if vectors:
            X.append(np.mean(vectors, axis=0))
        else:
            X.append(np.zeros(100))  # Se não houver palavras no modelo, usa um vetor zero

    return np.array(X)

# Função para treinar e avaliar o modelo Random Forest
def train_and_evaluate(X, labels, label_map):
    logger.info(f"{YELLOW}Dividindo os dados em conjuntos de treino e teste...{RESET}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    logger.info(f"{YELLOW}Treinando o modelo Random Forest com 100 estimadores...{RESET}")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info(f"{GREEN}Treinamento concluído com sucesso!{RESET}")

    logger.info(f"{YELLOW}Realizando previsões no conjunto de teste...{RESET}")
    y_pred = model.predict(X_test)

    # Identificar classes únicas presentes no teste
    test_classes = np.unique(y_test)
    test_class_names = [name for name, val in label_map.items() if val in test_classes]

    # Relatório de classificação
    logger.info(f"\n{BLUE}Relatório de Classificação:{RESET}")
    print(
        classification_report(
            y_test, y_pred, labels=test_classes, target_names=test_class_names
        )
    )

    # Matriz de Confusão
    logger.info(f"{YELLOW}Gerando matriz de confusão...{RESET}")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=test_classes)

    # Plotando a matriz de confusão
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_class_names,
        yticklabels=test_class_names,
    )
    plt.title("Matriz de Confusão")
    plt.xlabel("Rótulos Preditos")
    plt.ylabel("Rótulos Reais")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Salvar a matriz de confusão em um arquivo
    plt.savefig("confusion_matrix.png")
    plt.close()
    logger.info(f"{GREEN}Matriz de confusão salva em 'confusion_matrix.png'{RESET}")

    return model

# Função para salvar o modelo Random Forest
def save_model(model, filename="random_forest_cwe_classifier.pkl"):
    joblib.dump(model, filename)
    logger.info(f"{GREEN}Modelo salvo como '{filename}'{RESET}")

# Execução do pipeline
if __name__ == "__main__":
    logger.info(f"{BLUE}Iniciando pipeline de treinamento...{RESET}")
    
    # 1. Carregar os dados e labels
    data, labels, label_map = load_data()

    # Argumento para selecionar o método de vetorização
    parser = argparse.ArgumentParser(description="Processar conjuntos de dados CWE e treinar modelo.")
    parser.add_argument("--embedding_method", type=str, choices=["word2vec"], default="word2vec", help="Escolha o método de embedding: 'word2vec'")
    args = parser.parse_args()

    # 2. Carregar o modelo Word2Vec treinado
    model = load_word2vec_model()

    # 3. Vetorização com o modelo carregado
    X = vectorize_data_with_trained_word2vec(data, model)

    # 4. Treinamento e Avaliação
    trained_model = train_and_evaluate(X, labels, label_map)

    # 5. Salvar o Modelo
    save_model(trained_model)
    
    logger.info(f"{GREEN}Pipeline concluído com sucesso!{RESET}")
