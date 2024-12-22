import os
import json
import joblib
import argparse
import numpy as np
import re  # Adicione esta linha
from tree_sitter import Language, Parser
import gensim
from gensim.models import Word2Vec

# Carregar a biblioteca combinada de linguagens
C_LANGUAGE = Language('tree-sitter-dir/tree_sitter.so', 'c')

CPP_RESERVED_WORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "delete", "do", "double", "else", 
    "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline", 
    "int", "long", "mutable", "namespace", "new", "noexcept", "not", "nullptr", "operator", "private", 
    "protected", "public", "register", "reinterpret_cast", "return", "short", "signed", "sizeof", 
    "static", "static_assert", "static_cast", "struct", "switch", "template", "this", "throw", "true", 
    "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", 
    "wchar_t", "while"
}

LIBRARY_FUNCTIONS = {
    "strcpy", "printf", "scanf", "malloc", "free", "memset", "memcpy", "strlen", "strcat", "fopen", 
    "fclose", "fprintf", "fscanf", "std::cout", "std::cin", "std::endl", "std::string"
}

# Função para carregar a gramática C/C++ do tree-sitter
def load_tree_sitter_grammar():
    return C_LANGUAGE

# Função de pré-processamento usando o Tree-Sitter
def preprocess_code(code, debug=False):
    """
    Pré-processa o código fornecido usando tree-sitter para análise e tokenização.
    """
    language = C_LANGUAGE
    parser = Parser()
    parser.set_language(language)

    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception as e:
        print(f"Erro ao parsear o código! Erro: {e}")
        return None

    # Função para coletar os tokens
    def collect_tokens(node):
        tokens = []
        if node.type == 'identifier':  # Captura identificadores
            tokens.append(code[node.start_byte:node.end_byte])
        for child in node.children:
            tokens.extend(collect_tokens(child))
        return tokens

    # Coletar tokens do AST
    tokens = collect_tokens(tree.root_node)

    # Debug: Se ativado, imprime os tokens processados
    if debug:
        print("Tokens processados:", tokens)

    # Retorna o código como string para uso com o tokenizador
    return " ".join(tokens)  # Aqui, a saída será uma string, não uma lista de tokens.

# Função para carregar os conjuntos de dados CWE com manipulação de subpastas aninhadas
def load_all_cwe_datasets(base_path, min_files, max_files=None):
    """
    Carrega todos os arquivos de código (C/C++) das subpastas em 'base_path', pré-processa-os
    e atribui rótulos com base nos nomes das pastas CWE, ignorando arquivos 'main.cpp' e 'main_linux.cpp'.
    """
    data = []
    labels = []
    label_map = {}
    current_label = 0

    for root, dirs, _ in os.walk(base_path):
        for dir_name in sorted(dirs):
            if dir_name.startswith("CWE"):
                cwe_path = os.path.join(root, dir_name)
                print(f"[INFO] Processando pasta CWE: {cwe_path}")

                # Conta todos os arquivos .c e .cpp no diretório e subdiretórios
                file_count = sum(
                    1 for sub_root, _, files in os.walk(cwe_path)
                    for file in files if file.endswith((".c", ".cpp")) and "_good" not in file and file not in ["main.cpp", "main_linux.cpp"]
                )

                if file_count < min_files:
                    print(f"[AVISO] Ignorando pasta '{dir_name}' devido a arquivos insuficientes ({file_count}/{min_files})")
                    continue

                # Adiciona a pasta CWE ao mapa de rótulos
                label_map[dir_name] = current_label
                current_label += 1

                loaded_files = 0
                for sub_root, _, files in os.walk(cwe_path):
                    for file in files:
                        # Ignora 'main.cpp' e 'main_linux.cpp'
                        if file.endswith((".c", ".cpp")) and file not in ["main.cpp", "main_linux.cpp"]:
                            if max_files and loaded_files >= max_files:
                                break
                            file_path = os.path.join(sub_root, file)
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                code = f.read()

                                # Pré-processa o código
                                processed_code = preprocess_code(code)
                                if processed_code is None:  # Ignorar trechos vazios
                                    print(f"[AVISO] Ignorando arquivo processado vazio: {file_path}")
                                    continue

                                # Usa o valor do label_map para example_type
                                example_type = label_map[dir_name]

                                # Tokeniza o código em palavras para o Word2Vec
                                tokens = processed_code.split()

                                # Armazena como um dicionário
                                data.append({"code": tokens, "example_type": example_type})
                                labels.append(label_map[dir_name])
                                loaded_files += 1

    print(f"[RESUMO] Processados {len(data)} arquivos em {len(label_map)} categorias CWE.")
    return data, labels, label_map

# Função para treinar o modelo Word2Vec
def train_word2vec(data):
    """
    Treina o modelo Word2Vec usando os dados fornecidos.
    """
    # Treina o modelo Word2Vec com 100 dimensões
    model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model

# Função para converter tokens em vetores utilizando o modelo Word2Vec
def convert_to_vectors(data, model):
    """
    Converte os tokens em vetores de acordo com o modelo Word2Vec.
    """
    vectors = []
    for item in data:
        vector = np.mean([model.wv[token] for token in item["code"] if token in model.wv], axis=0)
        vectors.append(vector)
    return vectors

# Função principal para executar a aplicação
def main():
    parser = argparse.ArgumentParser(description="Processar conjuntos de dados CWE e treinar modelo.")
    parser.add_argument("--base_path", type=str, help="Caminho para a pasta principal contendo os diretórios CWE.")
    parser.add_argument("--minFiles", type=int, default=10, help="Número mínimo de arquivos necessários em uma pasta CWE para processamento.")
    parser.add_argument("--maxFiles", type=int, default=100, help="Número máximo de arquivos a serem carregados por pasta CWE.")
    args = parser.parse_args()

    # Carregar todos os dados e rótulos
    data, labels, label_map = load_all_cwe_datasets(args.base_path, args.minFiles, args.maxFiles)

    # Salvar cwe_data.json
    with open("cwe_data.json", "w", encoding="utf-8") as data_file:
        json.dump(data, data_file, ensure_ascii=False, indent=4)
    print("[INFO] Dados salvos em cwe_data.json")

    # Treinar o modelo Word2Vec
    word2vec_model = train_word2vec([item["code"] for item in data])

    # Converter os dados para vetores
    vectors = convert_to_vectors(data, word2vec_model)

    # Salvar em arquivos JSON
    with open("cwe_vectors.json", "w", encoding="utf-8") as vectors_file:
        vectors_list = [vector.tolist() for vector in vectors]  # Converte para listas
        json.dump(vectors_list, vectors_file, ensure_ascii=False, indent=4)

    with open("cwe_labels.json", "w", encoding="utf-8") as labels_file:
        json.dump(labels, labels_file, ensure_ascii=False, indent=4)

    with open("cwe_label_map.json", "w", encoding="utf-8") as label_map_file:
        json.dump(label_map, label_map_file, ensure_ascii=False, indent=4)

    print("[INFO] Dados vetoriais, rótulos e mapeamento de rótulos foram salvos com sucesso em JSON!")

if __name__ == "__main__":
    main()
