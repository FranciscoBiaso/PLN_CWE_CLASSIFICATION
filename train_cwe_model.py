import argparse
import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import gensim
from gensim.models import Word2Vec
import seaborn as sns
import matplotlib.pyplot as plt

# Define color codes for logging
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        if record.levelno == logging.INFO:
            return f"{BLUE}{log_message}{RESET}"
        elif record.levelno == logging.WARNING:
            return f"{YELLOW}{log_message}{RESET}"
        elif record.levelno == logging.ERROR:
            return f"{RED}{log_message}{RESET}"
        return log_message

# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_data(directory):
    """
    Carrega os dados e labels a partir de arquivos JSON no diretório especificado.
    """
    data_path = os.path.join(directory, "cwe_data.json")
    labels_path = os.path.join(directory, "cwe_labels.json")
    label_map_path = os.path.join(directory, "cwe_label_map.json")

    # Check if all required files exist
    for path in [data_path, labels_path, label_map_path]:
        if not os.path.isfile(path):
            logger.error(f"Arquivo não encontrado: {path}")
            exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    logger.info(f"Dados carregados: {len(data)} exemplos")
    return data, labels, label_map

def load_word2vec_model(model_path):
    """
    Carrega o modelo Word2Vec treinado a partir do caminho especificado.
    """
    if not os.path.isfile(model_path):
        logger.error(f"Modelo Word2Vec não encontrado: {model_path}")
        exit(1)
    model = Word2Vec.load(model_path)
    logger.info(f"{GREEN}Modelo Word2Vec carregado com sucesso!{RESET}")
    
    # Exibe a quantidade de palavras no vocabulário do modelo
    vocabulary_size = len(model.wv)
    logger.info(f"{BLUE}O modelo Word2Vec contém {vocabulary_size} palavras no vocabulário.{RESET}")
    
    return model

def vectorize_data_with_trained_word2vec(data, model, vector_size):
    """
    Vetoriza os dados processados usando um modelo Word2Vec treinado.
    """
    logger.info(f"{YELLOW}Vetorizando os dados com o modelo Word2Vec carregado...{RESET}")
    sentences = [entry["tokens"] for entry in data]
    X = []
    for idx, sentence in enumerate(sentences):
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if vectors:
            X.append(np.mean(vectors, axis=0))
        else:
            X.append(np.zeros(vector_size))
        if (idx + 1) % 10 == 0:
            logger.debug(f"Vetorizado {idx + 1}/{len(sentences)} exemplos")
    return np.array(X)

def train_and_evaluate(X, labels, label_map, test_size, random_state, n_estimators, output_dir):
    """
    Treina um classificador Random Forest e avalia seu desempenho.
    """
    logger.info(f"{YELLOW}Dividindo os dados em conjuntos de treino e teste...{RESET}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    logger.info(f"{YELLOW}Treinando o modelo Random Forest...{RESET}")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info(f"{GREEN}Treinamento concluído!{RESET}")

    y_pred = model.predict(X_test)

    label_to_cwe = {v: k for k, v in label_map.items()}
    test_classes = np.unique(y_test)
    test_class_names = [label_to_cwe[label] for label in test_classes]

    logger.info(f"\n{BLUE}Relatório de Classificação:{RESET}")
    report = classification_report(
        y_test, y_pred, labels=test_classes, target_names=test_class_names
    )
    print(report)

    logger.info(f"{YELLOW}Gerando matriz de confusão...{RESET}")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=test_classes)
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
    plt.tight_layout()

    # Passando o diretório de saída para salvar corretamente a matriz de confusão
    output_conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_conf_matrix_path)
    plt.close()
    logger.info(f"{GREEN}Matriz de confusão salva como '{output_conf_matrix_path}'!{RESET}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Pipeline de treinamento e avaliação de modelo Random Forest.")
    parser.add_argument("--data_dir", type=str, required=True, help="Caminho para o diretório contendo os arquivos JSON.")
    parser.add_argument("--output_dir", type=str, default=".", help="Diretório para salvar os arquivos de saída.")
    parser.add_argument("--word2vec_model_path", type=str, required=True, help="Caminho para o modelo Word2Vec treinado.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Proporção dos dados para o conjunto de teste.")
    parser.add_argument("--random_state", type=int, default=42, help="Semente para aleatoriedade.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Número de árvores no Random Forest.")
    parser.add_argument("--vector_size", type=int, default=300, help="Tamanho dos vetores gerados pelo Word2Vec.")
    args = parser.parse_args()

    data, labels, label_map = load_data(args.data_dir)
    model = load_word2vec_model(args.word2vec_model_path)
    X = vectorize_data_with_trained_word2vec(data, model, args.vector_size)
    trained_model = train_and_evaluate(X, labels, label_map, args.test_size, args.random_state, args.n_estimators, args.output_dir)

    # Verifica se o diretório de saída existe, caso contrário cria
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, "random_forest_cwe_classifier.pkl")
    joblib.dump(trained_model, output_path)
    logger.info(f"Modelo salvo em {output_path}.")

if __name__ == "__main__":
    main()
