import os
import json
import argparse
import numpy as np
from tree_sitter import Language, Parser
from gensim.models import Word2Vec

# Load the combined language library
# Ensure you have built the tree-sitter library correctly and the path is accurate
# Example build command:
# Language.build_library(
#     'tree-sitter-dir/tree_sitter.so',
#     ['tree-sitter-c']
# )
C_LANGUAGE = Language('tree-sitter-dir/tree_sitter.so', 'c')

def collect_relevant_tokens(node, code):
    """
    Collect relevant tokens from the syntax tree.

    Args:
        node: Current AST node.
        code: Original source code as a string.

    Returns:
        List[str]: List of relevant tokens as strings.
    """
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
    """
    Preprocess the provided code using tree-sitter for analysis and tokenization.

    Args:
        code: Source code as a string.
        file_path: Path to the file being processed (for logging purposes).

    Returns:
        Tuple[str, List[str]]: 
            - Concatenated token string.
            - List of tokens.
    """
    print(f"[INFO] Preprocessing file: {file_path}")
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception as e:
        print(f"[ERROR] Error parsing {file_path}: {e}")
        return None, []
    tokens = collect_relevant_tokens(tree.root_node, code)

    if tokens:
        removed_token = tokens.pop(0)
    print(f"[DEBUG] Extracted {len(tokens)} tokens from {file_path}")
    return " ".join(tokens), tokens

def process_files(base_path, pattern, min_files, max_files):
    """
    Recursively process files matching a specific pattern in the base directory,
    assigning labels based on CWE directories.

    Args:
        base_path: Base directory to search for files.
        pattern: Filename pattern to match.
        min_files: Minimum number of matching files required per CWE directory.
        max_files: Maximum number of matching files to process per CWE directory.

    Returns:
        Tuple[List[dict], List[int], dict]: (data, labels, label_map)
    """
    print(f"[INFO] Searching for files in {base_path} matching pattern '{pattern}'")
    data = []
    labels = []
    label_map = {}
    label_counter = 0
    total_processed = 0

    for root, _, files in os.walk(base_path):
        # Identify CWE directories by extracting the CWE identifier from the folder name
        # Assumes CWE directories are named like 'CWE122_Heap_Based_Buffer_Overflow'
        # Adjust the extraction logic if your directory naming convention differs
        folder_name = os.path.basename(root)
        if not folder_name.startswith("CWE"):
            continue  # Skip non-CWE directories

        # Filter files that match the pattern
        matching_files = [f for f in files if f.endswith((".c", ".cpp")) and pattern in f]

        # Ensure directory meets the file count interval criteria
        if len(matching_files) < min_files or len(matching_files) > max_files:
            print(f"[INFO] Skipping directory '{root}': {len(matching_files)} matching files (required interval: {min_files}-{max_files}).")
            continue

        # Assign a label to this CWE directory
        label_map[folder_name] = label_counter
        current_label = label_counter
        label_counter += 1
        print(f"[INFO] Processing directory '{root}' with label '{current_label}'.")

        processed_in_dir = 0
        for file_name in matching_files:
            if processed_in_dir >= max_files:
                print(f"[INFO] Reached the max file limit for directory: {root}")
                break

            file_path = os.path.join(root, file_name)
            print(f"[INFO] Found matching file: {file_path}")

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                    processed_code, tokens = preprocess_code(code, file_path)
                    if processed_code:
                        data.append({"tokens": tokens})
                        labels.append(current_label)
                        print(f"[INFO] Successfully processed: {file_path}")
                        processed_in_dir += 1
                        total_processed += 1
                    else:
                        print(f"[WARN] Skipping file {file_path}: Preprocessing failed.")
            except Exception as e:
                print(f"[ERROR] Could not process {file_path}: {e}")

    if total_processed == 0:
        print("[WARN] No matching files found within the specified interval.")
    else:
        print(f"[INFO] Finished processing {total_processed} files across all directories.")

    return data, labels, label_map

def train_word2vec(corpus):
    """
    Train a Word2Vec model on the provided corpus.

    Args:
        corpus: List of token lists.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    print(f"[INFO] Training Word2Vec model on {len(corpus)} samples")
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    print("[INFO] Word2Vec model saved as 'word2vec.model'")
    return model

def convert_to_vectors(data, word2vec_model):
    """
    Convert code data to vectors using a trained Word2Vec model.

    Args:
        data: List of code data dictionaries.
        word2vec_model: Trained Word2Vec model.

    Returns:
        np.array: Array of vectors.
    """
    print("[INFO] Converting processed code to vectors")
    vectors = []
    for idx, item in enumerate(data):
        tokens = item["tokens"]
        token_vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
        if token_vectors:
            vectors.append(np.mean(token_vectors, axis=0))
        else:
            vectors.append(np.zeros(word2vec_model.vector_size))
        if (idx + 1) % 10 == 0:
            print(f"[DEBUG] Converted {idx + 1}/{len(data)} samples to vectors")
    print("[INFO] Vectorization complete")
    return np.array(vectors)

def main():
    parser = argparse.ArgumentParser(description="Process CWE datasets and generate required files.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the main directory containing CWE subdirectories.")
    parser.add_argument("--pattern", type=str, default="bad", help="Pattern to match in file names.")
    parser.add_argument("--minFiles", type=int, default=1, help="Minimum number of matching files per CWE directory.")
    parser.add_argument("--maxFiles", type=int, default=100, help="Maximum number of matching files per CWE directory.")
    args = parser.parse_args()

    if not os.path.isdir(args.base_path):
        print(f"[ERROR] The base path does not exist or is not a directory: {args.base_path}")
        return

    print(f"[INFO] Starting processing for base path: {args.base_path} with pattern: '{args.pattern}'")
    data, labels, label_map = process_files(args.base_path, args.pattern, args.minFiles, args.maxFiles)

    if data:
        # Save cwe_data.json
        print(f"[INFO] Saving processed data to 'cwe_data.json'")
        with open("cwe_data.json", "w", encoding="utf-8") as data_file:
            json.dump(data, data_file, ensure_ascii=False, indent=4)

        # Save cwe_labels.json
        print(f"[INFO] Saving labels to 'cwe_labels.json'")
        with open("cwe_labels.json", "w", encoding="utf-8") as labels_file:
            json.dump(labels, labels_file, ensure_ascii=False, indent=4)

        # Save cwe_label_map.json
        print(f"[INFO] Saving label map to 'cwe_label_map.json'")
        with open("cwe_label_map.json", "w", encoding="utf-8") as label_map_file:
            json.dump(label_map, label_map_file, ensure_ascii=False, indent=4)

        # Train Word2Vec model
        print("[INFO] Preparing corpus for Word2Vec training")
        corpus = [item["tokens"] for item in data]
        word2vec_model = train_word2vec(corpus)

        # Convert to vectors
        vectors = convert_to_vectors(data, word2vec_model)

        # Save cwe_vectors.json
        print(f"[INFO] Saving vectors to 'cwe_vectors.json'")
        with open("cwe_vectors.json", "w", encoding="utf-8") as vectors_file:
            json.dump(vectors.tolist(), vectors_file, ensure_ascii=False, indent=4)

        print("[INFO] Processing complete. All files successfully processed!")
    else:
        print("[WARN] No files matched the pattern or were processed.")

if __name__ == "__main__":
    main()
