import os
import json
import joblib
import argparse
import numpy as np
from gensim.models import Word2Vec

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Carregar o label map
def load_label_map(label_map_path):
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Label map carregado com sucesso!")
        return label_map
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Arquivo label map não encontrado: {label_map_path}")
        exit(1)

# Carregar modelo RandomForest
def load_random_forest_model(model_path="random_forest_cwe_classifier.pkl"):
    try:
        model = joblib.load(model_path)
        print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Modelo Random Forest carregado com sucesso!")
        return model
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Modelo Random Forest não encontrado: {model_path}")
        exit(1)

# Carregar modelo Word2Vec
def load_word2vec_model(model_path="word2vec.model"):
    try:
        model = Word2Vec.load(model_path)
        print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Modelo Word2Vec carregado com sucesso!")
        return model
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Modelo Word2Vec não encontrado: {model_path}")
        exit(1)

# Função para extrair o prefixo CWE do nome do arquivo (ex: "CWE775_")
def extract_cwe_prefix(file_name):
    return file_name.split('_')[0]  # Exemplo: 'CWE775' de 'CWE775_Missing_Release_of_File_Descriptor_or_Handle.cpp'

def classify_with_random_forest(directory, model, label_map, word2vec_model):
    results = []
    correct_predictions = 0
    total_files = 0

    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith((".c", ".cpp")):
                file_path = os.path.join(root, file_name)
                total_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                        code = file.read()

                    tokens = code.split()
                    vectors = [
                        word2vec_model.wv[token]
                        for token in tokens
                        if token in word2vec_model.wv
                    ]

                    # Verificar se algum vetor foi encontrado
                    if vectors:
                        feature_vector = np.mean(vectors, axis=0).reshape(1, -1)
                    else:
                        # Caso nenhum vetor tenha sido encontrado, gerar um vetor de zeros
                        feature_vector = np.zeros((1, word2vec_model.vector_size))

                    prediction = model.predict(feature_vector)[0]
                    # Extraímos o prefixo da previsão (sem o sufixo)
                    predicted_cwe_prefix = extract_cwe_prefix(next((cwe for cwe, label in label_map.items() if label == prediction), None))

                    # Extraímos o prefixo esperado a partir do nome do arquivo
                    expected_cwe_prefix = extract_cwe_prefix(file_name)

                    # Verifica se a previsão está correta comparando os prefixos
                    is_correct = (predicted_cwe_prefix == expected_cwe_prefix)

                    if is_correct:
                        results.append((file_path, predicted_cwe_prefix, True))
                        correct_predictions += 1
                    else:
                        results.append((file_path, predicted_cwe_prefix, False))
                except Exception as e:
                    results.append((file_path, f"Erro ao processar: {e}", False))

    print(f"{bcolors.HEADER}[RESULTADOS]{bcolors.ENDC}")
    for file_path, classification, is_correct in results:
        if is_correct:
            print(f"{bcolors.OKBLUE}[FILE]{bcolors.ENDC} {file_path} -> {bcolors.OKGREEN}{classification}{bcolors.ENDC}")
        else:
            expected_cwe_prefix = extract_cwe_prefix(os.path.basename(file_path))
            print(f"{bcolors.OKBLUE}[FILE]{bcolors.ENDC} {file_path} -> {bcolors.FAIL}{classification}{bcolors.ENDC} (Esperado: {expected_cwe_prefix})")

    # Calcular o percentual de acerto
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    print(f"\n{bcolors.OKCYAN}[INFO]{bcolors.ENDC} Percentual de acerto: {accuracy:.2f}%")

# Função principal
def main():
    parser = argparse.ArgumentParser(description="Classificador de vulnerabilidades CWE em arquivos C++")
    parser.add_argument("--file", type=str, required=True, help="Diretório contendo os arquivos C++ para classificação")
    parser.add_argument("--model", type=str, default="random_forest_cwe_classifier.pkl", help="Caminho para o modelo treinado")
    parser.add_argument("--label_map", type=str, default="cwe_label_map.json", help="Caminho para o mapeamento de labels")
    parser.add_argument("--word2vec", type=str, default="word2vec.model", help="Caminho para o modelo Word2Vec")
    args = parser.parse_args()

    label_map = load_label_map(args.label_map)
    model = load_random_forest_model(args.model)
    word2vec_model = load_word2vec_model(args.word2vec)

    classify_with_random_forest(args.file, model, label_map, word2vec_model)

if __name__ == "__main__":
    main()
