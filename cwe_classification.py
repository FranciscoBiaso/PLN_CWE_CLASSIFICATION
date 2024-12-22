import os
import json
import joblib
import argparse
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

# Definir cores para mensagens
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

# Função para carregar os dados e rótulos
def load_data():
    with open("cwe_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("cwe_labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open("cwe_label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    print(f"Dados carregados: {len(data)} exemplos")
    return data, labels, label_map

# Carregar modelo RandomForest
def load_random_forest_model(model_path="random_forest_cwe_classifier.pkl"):
    """
    Carrega o modelo Random Forest treinado previamente.
    """
    model = joblib.load(model_path)
    print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Modelo Random Forest carregado com sucesso!")
    return model

# Carregar modelo CodeBERT e tokenizador
def load_codebert_model(model_path="microsoft/codebert-base"):
    """
    Carrega o modelo CodeBERT para classificação de código.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Função para classificar usando RandomForest
def classify_with_random_forest(file_path, model, label_map):
    """
    Lê um arquivo C++, pré-processa e classifica usando o modelo Random Forest.
    """
    try:
        if not os.path.isfile(file_path):
            print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} O arquivo '{file_path}' não existe.")
            return

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            code = file.read()

        # Pré-processamento usando tokens simples (exemplo, você pode substituir com sua técnica de vetorização)
        tokens = code.split()
        vector = np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0).reshape(1, -1)

        # Previsão com Random Forest
        prediction = model.predict(vector)[0]
        predicted_cwe = next((cwe for cwe, label in label_map.items() if label == prediction), None)

        if predicted_cwe:
            print(f"{bcolors.OKGREEN}[RESULT]{bcolors.ENDC} O arquivo '{file_path}' foi classificado como: {predicted_cwe}")
        else:
            print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Classificação desconhecida.")
    except Exception as e:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Falha ao processar o arquivo: {e}")

# Função para classificar usando CodeBERT
def classify_with_codebert(file_path, tokenizer, model, label_map):
    """
    Lê um arquivo C++, pré-processa e classifica usando o modelo CodeBERT.
    """
    try:
        if not os.path.isfile(file_path):
            print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} O arquivo '{file_path}' não existe.")
            return

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            code = file.read()

        # Pré-processamento e tokenização usando tokenizer do CodeBERT
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Previsão do modelo
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

        predicted_cwe = next((cwe for cwe, label in label_map.items() if label == prediction), None)

        if predicted_cwe:
            print(f"{bcolors.OKGREEN}[RESULT]{bcolors.ENDC} O arquivo '{file_path}' foi classificado como: {predicted_cwe}")
        else:
            print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Classificação desconhecida.")
    except Exception as e:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Falha ao processar o arquivo: {e}")

# Função principal para executar a aplicação
def main():
    parser = argparse.ArgumentParser(description="Classificador de vulnerabilidades CWE em arquivos C++")
    parser.add_argument("--file", type=str, required=True, help="Caminho do arquivo C++ para classificação")
    parser.add_argument("--model_type", type=str, choices=["random_forest", "codebert"], required=True, help="Escolha o tipo de modelo: 'random_forest' ou 'codebert'")
    parser.add_argument("--model", type=str, default="random_forest_cwe_classifier.pkl", help="Caminho para o modelo treinado (para random_forest)")
    parser.add_argument("--label_map", type=str, default="cwe_label_map.json", help="Caminho para o mapeamento de labels")
    args = parser.parse_args()

    # Carregar dados e label_map
    data, labels, label_map = load_data()

    # Carregar o modelo selecionado
    if args.model_type == "random_forest":
        model = load_random_forest_model(args.model)
        # Para Random Forest, normalmente o vetor de entrada seria pré-processado para usar o modelo.
        classify_with_random_forest(args.file, model, label_map)
    elif args.model_type == "codebert":
        tokenizer, model = load_codebert_model()
        classify_with_codebert(args.file, tokenizer, model, label_map)

if __name__ == "__main__":
    main()
