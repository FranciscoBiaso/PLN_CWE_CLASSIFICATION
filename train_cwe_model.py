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
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        if record.levelno == logging.INFO:
            return f"{BLUE}{BOLD}{log_message}{RESET}"
        elif record.levelno == logging.WARNING:
            return f"{YELLOW}{BOLD}{log_message}{RESET}"
        elif record.levelno == logging.ERROR:
            return f"{RED}{BOLD}{log_message}{RESET}"
        elif record.levelno == logging.DEBUG:
            return f"{CYAN}{BOLD}{log_message}{RESET}"
        return log_message

# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Suppress verbose gensim logs
gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.WARNING)

def load_data(directory):
    data_path = os.path.join(directory, "cwe_data.json")
    label_map_path = os.path.join(directory, "cwe_label_map.json")

    if not os.path.isfile(data_path):
        logger.error(f"❌ Arquivo não encontrado: {data_path}")
        exit(1)
    if not os.path.isfile(label_map_path):
        logger.error(f"❌ Arquivo não encontrado: {label_map_path}")
        exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    labels = [entry["label"] for entry in data]

    logger.info(f"📂 Dados carregados: {len(data)} exemplos")
    logger.info(f"📂 Rótulos carregados: {len(labels)} rótulos")

    if len(data) != len(labels):
        logger.error(f"❌ O número de exemplos ({len(data)}) não coincide com o número de rótulos ({len(labels)}).")
        exit(1)

    return data, labels, label_map

def load_word2vec_model(model_path):
    if not os.path.isfile(model_path):
        logger.error(f"❌ Modelo Word2Vec não encontrado: {model_path}")
        exit(1)
    model = Word2Vec.load(model_path)
    logger.info(f"✅ {GREEN}Modelo Word2Vec carregado com sucesso!{RESET}")
    logger.info(f"📖 O modelo contém {len(model.wv)} palavras no vocabulário.")
    return model

def vectorize_data_with_trained_word2vec(data, model, vector_size):
    logger.info(f"🔄 Vetorizando os dados com o modelo Word2Vec carregado...")
    sentences = [entry["code"] for entry in data]
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

def train_and_evaluate(X, labels, label_map, test_size, random_state, n_estimators, output_dir, stratify):
    logger.info(f"📊 Dividindo os dados em conjuntos de treino e teste...")
    if stratify is not None:
        logger.info(f"🔍 Estratificação está {GREEN}ativada{RESET}.")
    else:
        logger.info(f"⚠️ Estratificação está {RED}desativada{RESET}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=stratify
    )
    logger.info(f"🧠 Treinando o modelo Random Forest...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
    model.fit(X_train, y_train)
    logger.info(f"✅ {GREEN}Treinamento concluído!{RESET}")

    y_pred = model.predict(X_test)

    label_to_cwe = {v: k for k, v in label_map.items()}
    test_classes = np.unique(y_test)
    test_class_names = [label_to_cwe[label] for label in test_classes]

    logger.info(f"\n📊 {BLUE}Relatório de Classificação:{RESET}")
    report = classification_report(
        y_test, y_pred, labels=test_classes, target_names=test_class_names
    )
    print(report)

    logger.info(f"📈 Gerando matriz de confusão...")
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

    output_conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_conf_matrix_path)
    plt.close()
    logger.info(f"✅ {GREEN}Matriz de confusão salva em: {output_conf_matrix_path}{RESET}")
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
    parser.add_argument("--stratification", type=str, choices=['yes', 'no'], default='yes', help="Ativar ou desativar a estratificação (yes ou no).")
    args = parser.parse_args()

    data, labels, label_map = load_data(args.data_dir)
    model = load_word2vec_model(args.word2vec_model_path)

    X = vectorize_data_with_trained_word2vec(data, model, args.vector_size)

    # Determinar se deve usar estratificação
    if args.stratification.lower() == 'yes':
        stratify = labels
    else:
        stratify = None

    trained_model = train_and_evaluate(
        X, labels, label_map, args.test_size, args.random_state, args.n_estimators, args.output_dir, stratify
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, "random_forest_cwe_classifier.pkl")
    joblib.dump(trained_model, output_path)
    logger.info(f"✅ {GREEN}Modelo salvo em: {output_path}{RESET}")

if __name__ == "__main__":
    main()