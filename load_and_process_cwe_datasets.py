import os
import json
import argparse
import numpy as np
from tree_sitter import Language, Parser
from gensim.models import Word2Vec
import logging

# Load the combined language library
C_LANGUAGE = Language('tree-sitter-cpp/libtree-sitter-cpp.so', 'cpp')

# Define color codes for logs
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

def collect_relevant_tokens(node, code):
    tokens = []
    if node.type in {"string_literal", "number_literal"}:
        tokens.append(code[node.start_byte:node.end_byte])
    elif node.type in {"binary_operator", "unary_operator"}:
        tokens.append(code[node.start_byte:node.end_byte])
    elif node.type in {"if_statement", "for_statement", "while_statement", "return_statement"}:
        tokens.append(node.type)
    elif node.type in {"type_identifier", "identifier"}:
        tokens.append(code[node.start_byte:node.end_byte])
    elif node.type.startswith("preproc_"):
        tokens.append(code[node.start_byte:node.end_byte])
    for child in node.children:
        tokens.extend(collect_relevant_tokens(child, code))
    return tokens

def preprocess_code(code, file_path):
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception as e:
        logger.error(f"❌ Error parsing file {file_path}: {e}")
        return None
    tokens = collect_relevant_tokens(tree.root_node, code)
    return tokens

def collect_files_in_subfolder(subfolder_path, pattern, min_files, max_files):
    eligible_files = []
    for root, _, files in os.walk(subfolder_path):
        for file in files:
            if file.endswith((".c", ".cpp")) and (not pattern or pattern in file):
                eligible_files.append(os.path.join(root, file))

    if len(eligible_files) < min_files:
        return []
    
    return eligible_files[:max_files]

def process_files_by_subfolder(base_path, pattern, max_files, min_files, labels, label_map):
    logger.info(f"🔍 {BOLD}Scanning for files in subfolders of {base_path}...{RESET}")
    data = []
    total_files_processed = 0

    label_id = 0
    for subfolder in sorted(os.listdir(base_path)):
        subfolder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        if subfolder not in label_map:
            label_map[subfolder] = label_id
            labels.append(label_id)
            logger.info(f"📂 {CYAN}Assigned label {label_id} to category: {subfolder}{RESET}")
            label_id += 1

        eligible_files = collect_files_in_subfolder(subfolder_path, pattern, min_files, max_files)

        if not eligible_files:
            logger.warning(f"⚠️ {YELLOW}No files found in subfolder: {subfolder}.{RESET}")
            continue

        logger.info(f"📂 {CYAN}Processing {len(eligible_files)} files in subfolder: {subfolder}.{RESET}")
        file_count = 0
        for file_path in eligible_files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                    processed_tokens = preprocess_code(code, file_path)
                    if processed_tokens:
                        data.append({"subfolder": subfolder, "code": processed_tokens, "label": label_map[subfolder]})
            except Exception as e:
                logger.error(f"⚠️ Error processing file {file_path}: {e}")
            file_count += 1

        total_files_processed += file_count

    if total_files_processed == 0:
        logger.warning(f"⚠️ {YELLOW}No files were processed.{RESET}")
    else:
        logger.info(f"✅ {GREEN}Finished processing {total_files_processed} files.{RESET}")
    
    return data

def train_word2vec(corpus):
    logger.info(f"🧠 {BOLD}Training Word2Vec model...{RESET}")
    tokenized_corpus = [tokens for tokens in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    logger.info(f"✅ {GREEN}Word2Vec model training completed and saved as 'word2vec.model'.{RESET}")
    return model

def convert_to_vectors(data, word2vec_model):
    logger.info(f"🔄 {BOLD}Converting processed code to vectors...{RESET}")
    vectors = []
    for item in data:
        tokens = item["code"]
        token_vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
        if token_vectors:
            vectors.append(np.mean(token_vectors, axis=0))
        else:
            vectors.append(np.zeros(word2vec_model.vector_size))
    logger.info(f"✅ {GREEN}Vectorization complete.{RESET}")
    return np.array(vectors)

def main():
    parser = argparse.ArgumentParser(description="Process CWE datasets and generate required files.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the main directory containing code files.")
    parser.add_argument("--pattern", type=str, default="", help="Pattern to match in file names.")
    parser.add_argument("--maxFiles", type=int, default=100, help="Maximum number of files to process per subfolder.")
    parser.add_argument("--minFiles", type=int, default=0, help="Minimum number of files required in a subfolder.")
    args = parser.parse_args()

    if not os.path.isdir(args.base_path):
        logger.error(f"❌ {RED}The base path does not exist or is not a directory: {args.base_path}{RESET}")
        return

    labels = []
    label_map = {}

    logger.info(f"🔍 {BOLD}Starting processing for base path: {args.base_path}...{RESET}")
    data = process_files_by_subfolder(args.base_path, args.pattern, args.maxFiles, args.minFiles, labels, label_map)

    if labels:
        with open("cwe_labels.json", "w", encoding="utf-8") as labels_file:
            json.dump(labels, labels_file, ensure_ascii=False, indent=4)
        logger.info(f"💾 {BOLD}Labels saved to 'cwe_labels.json'.{RESET}")

    if label_map:
        with open("cwe_label_map.json", "w", encoding="utf-8") as label_map_file:
            json.dump(label_map, label_map_file, ensure_ascii=False, indent=4)
        logger.info(f"💾 {BOLD}Label map saved to 'cwe_label_map.json'.{RESET}")

    if data:
        logger.info(f"💾 {BOLD}Saving processed data to 'cwe_data.json'...{RESET}")
        with open("cwe_data.json", "w", encoding="utf-8") as data_file:
            json.dump(data, data_file, ensure_ascii=False, indent=4)

        word2vec_model = train_word2vec([item["code"] for item in data])

        vectors = convert_to_vectors(data, word2vec_model)
        logger.info(f"💾 {BOLD}Saving vectors to 'cwe_vectors.json'...{RESET}")
        with open("cwe_vectors.json", "w", encoding="utf-8") as vectors_file:
            json.dump(vectors.tolist(), vectors_file, ensure_ascii=False, indent=4)

        logger.info(f"✅ {GREEN}Processing complete. All files successfully processed!{RESET}")
    else:
        logger.warning(f"⚠️ {YELLOW}No files matched the pattern or were processed.{RESET}")

if __name__ == "__main__":
    main()
