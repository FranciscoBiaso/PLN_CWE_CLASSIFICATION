import os
import json
import argparse
from tree_sitter import Language, Parser
from gensim.models import Word2Vec

# Carrega a biblioteca de linguagens combinadas para tree-sitter
C_LANGUAGE = Language('tree-sitter-dir/tree_sitter.so', 'c')

def collect_tokens(node, code):
    """
    Coleta identificadores e processa os nós filhos, excluindo o conteúdo do nó raiz.

    Args:
        node: Nó atual da AST (Árvore de Sintaxe Abstrata).
        code: Código-fonte como uma string.

    Returns:
        List[str]: Lista de tokens processados.
    """
    tokens = []

    # Ignora o processamento do conteúdo bruto do nó raiz
    if node.type == 'identifier':  # Captura identificadores
        tokens.append(code[node.start_byte:node.end_byte].strip())
    elif node.type.startswith('preproc_'):  # Trata diretivas de pré-processador
        tokens.append(code[node.start_byte:node.end_byte].strip())
    elif node.type == 'string_literal':  # Trata literais de string
        tokens.append(code[node.start_byte:node.end_byte].strip())

    # Processa recursivamente os nós filhos
    for child in node.children:
        tokens.extend(collect_tokens(child, code))

    return tokens

def collect_relevant_tokens(node, code, debug=False, depth=0):
    """
    Coleta tokens relevantes da árvore de sintaxe.

    Args:
        node: Nó atual da AST.
        code: Código-fonte original como string.
        debug: Booleano para ativar logs detalhados.
        depth: Profundidade atual na árvore para depuração.

    Returns:
        List[dict]: Lista de tokens relevantes com metadados.
    """
    tokens = []
    indent = "  " * depth

    if debug:
        print(f"{indent}[DEBUG] Visitando nó: {node.type}")

    if node.type in {"string_literal", "number_literal"}:  # Literais
        token = {
            "text": code[node.start_byte:node.end_byte],
            "type": node.type,
            "category": "literal"
        }
        tokens.append(token)
        if debug:
            print(f"{indent}[INFO] Literal encontrado: {token}")
    elif node.type in {"binary_operator", "unary_operator"}:  # Operadores
        token = {
            "text": code[node.start_byte:node.end_byte],
            "type": node.type,
            "category": "operator"
        }
        tokens.append(token)
        if debug:
            print(f"{indent}[INFO] Operador encontrado: {token}")
    elif node.type in {"if_statement", "for_statement", "while_statement", "return_statement"}:  # Estruturas de controle
        token = {
            "text": node.type,
            "type": node.type,
            "category": "control_structure"
        }
        tokens.append(token)
        if debug:
            print(f"{indent}[INFO] Estrutura de controle encontrada: {token}")
        for child in node.children:
            tokens.extend(collect_relevant_tokens(child, code, debug, depth + 1))
    elif node.type in {"type_identifier", "identifier"}:  # Identificadores e tipos
        token = {
            "text": code[node.start_byte:node.end_byte],
            "type": node.type,
            "category": "identifier"
        }
        tokens.append(token)
        if debug:
            print(f"{indent}[INFO] Identificador encontrado: {token}")
    elif node.type.startswith("preproc_"):  # Diretivas de pré-processador
        token = {
            "text": code[node.start_byte:node.end_byte],
            "type": node.type,
            "category": "preprocessor"
        }
        tokens.append(token)
        if debug:
            print(f"{indent}[INFO] Diretiva de pré-processador encontrada: {token}")

    for child in node.children:
        tokens.extend(collect_relevant_tokens(child, code, debug, depth + 1))

    return tokens

def preprocess_code(code, debug=False, mode="relevant"):
    """
    Preprocessa o código fornecido usando tree-sitter para análise e tokenização.

    Args:
        code: Código-fonte a ser processado.
        debug: Ativa logs detalhados.
        mode: Modo de processamento ("relevant" ou "identifiers").
    """
    parser = Parser()
    parser.set_language(C_LANGUAGE)

    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception as e:
        print(f"Erro ao analisar o código! Erro: {e}")
        return None

    tokens = []
    if mode == "identifiers":
        tokens = collect_tokens(tree.root_node, code)
    elif mode == "relevant":
        tokens = collect_relevant_tokens(tree.root_node, code, debug)
    else:
        print(f"[ERRO] Modo desconhecido: {mode}")
        return None
    # Remove o primeiro token, se existir
    if tokens:
        tokens.pop(0)

    # Filtra e achata tokens em strings
    token_texts = [token["text"] if isinstance(token, dict) else token for token in tokens]

    if debug:
        print("[DEBUG] Tokens processados:", token_texts)

    return " ".join(token_texts)

def debug_file(file_path, debug, mode):
    """
    Depura um arquivo específico, processando-o.

    Args:
        file_path: Caminho para o arquivo a ser depurado.
        debug: Booleano para ativar logs detalhados.
        mode: Modo de processamento ("relevant" ou "identifiers").
    """
    print(f"[INFO] Depurando arquivo: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
            processed_code = preprocess_code(code, debug, mode)
            if processed_code:
                print("[DEBUG] Saída do código processado:")
                print(processed_code)
            else:
                print("[AVISO] O arquivo não pôde ser processado.")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Processa datasets CWE e treina um modelo.")
    parser.add_argument("--base_path", type=str, help="Caminho para o diretório principal contendo subdiretórios CWE.")
    parser.add_argument("--minFiles", type=int, default=10, help="Número mínimo de arquivos necessários em uma pasta CWE para processamento.")
    parser.add_argument("--maxFiles", type=int, default=100, help="Número máximo de arquivos a serem processados por pasta CWE.")
    parser.add_argument("--debug", action="store_true", help="Ativa modo de depuração com logs detalhados.")
    parser.add_argument("--debug_file", type=str, help="Especifica um arquivo para depuração.")
    parser.add_argument("--token_mode", type=str, choices=["relevant", "identifiers"], default="relevant",
                        help="Escolha a função para processar tokens: 'relevant' para collect_relevant_tokens, 'identifiers' para collect_tokens.")
    args = parser.parse_args()

    if args.debug and args.debug_file:
        debug_file(args.debug_file, args.debug, args.token_mode)
    elif args.base_path:
        # Processa todo o dataset (não mostrado neste script para brevidade)
        pass
    else:
        print("[ERRO] Você deve especificar --base_path ou --debug_file para processamento.")

if __name__ == "__main__":
    main()
