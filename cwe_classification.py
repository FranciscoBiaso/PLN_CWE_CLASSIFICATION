import os
import json
import joblib
import argparse
import numpy as np
import re
from gensim.models import Word2Vec

# Color coding for logs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    GRAY = '\033[90m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Load the label map
def load_label_map(label_map_path):
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Label map loaded successfully!")
        return label_map
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Label map file not found: {label_map_path}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Failed to parse label map JSON: {e}")
        exit(1)

# Load Random Forest model
def load_random_forest_model(model_path="random_forest_cwe_classifier.pkl"):
    try:
        model = joblib.load(model_path)
        print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Random Forest model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Random Forest model file not found: {model_path}")
        exit(1)
    except Exception as e:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Failed to load Random Forest model: {e}")
        exit(1)

# Load Word2Vec model
def load_word2vec_model(model_path="word2vec.model"):
    try:
        model = Word2Vec.load(model_path)
        print(f"{bcolors.OKGREEN}[INFO]{bcolors.ENDC} Word2Vec model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Word2Vec model file not found: {model_path}")
        exit(1)
    except Exception as e:
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Failed to load Word2Vec model: {e}")
        exit(1)

# Extract CWE prefix using regex for robustness
def extract_cwe_prefix(label):
    match = re.match(r'(CWE\d+)', label, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Garantir que o prefixo está em maiúsculas
    else:
        # Fallback to splitting if regex does not match
        return label.split('_')[0].upper()

# Classify files using Random Forest
def classify_with_random_forest(directory, model, label_map, word2vec_model):
    results = []
    correct_predictions = 0
    total_files = 0

    print(f"{bcolors.OKCYAN}[INFO]{bcolors.ENDC} Scanning directory for files: {directory}")

    # Extract valid prefixes from the label map
    valid_prefixes = {extract_cwe_prefix(cwe) for cwe in label_map.keys()}
    print(f"{bcolors.OKCYAN}[DEBUG]{bcolors.ENDC} Valid Prefixes from label_map: {valid_prefixes}")

    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith((".c", ".cpp")):
                file_path = os.path.join(root, file_name)
                total_files += 1
                expected_cwe_prefix = extract_cwe_prefix(file_name)
                print(f"{bcolors.OKCYAN}[DEBUG]{bcolors.ENDC} Processing file: {file_path}")
                print(f"{bcolors.OKCYAN}[DEBUG]{bcolors.ENDC} Extracted Prefix: {expected_cwe_prefix}")

                # Skip files with prefixes not in the label map
                if expected_cwe_prefix not in valid_prefixes:
                    print(f"{bcolors.GRAY}[SKIP]{bcolors.ENDC} {file_path} -> Prefix '{expected_cwe_prefix}' not in label map.")
                    results.append((file_path, "Skipped", False))
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                        code = file.read()

                    tokens = code.split()
                    vectors = [
                        word2vec_model.wv[token]
                        for token in tokens
                        if token in word2vec_model.wv
                    ]

                    # Generate a feature vector or use zeros if none found
                    if vectors:
                        feature_vector = np.mean(vectors, axis=0).reshape(1, -1)
                    else:
                        feature_vector = np.zeros((1, word2vec_model.vector_size))

                    prediction = model.predict(feature_vector)[0]
                    predicted_label = next(
                        (cwe for cwe, label in label_map.items() if label == prediction), "Unknown"
                    )
                    predicted_cwe_prefix = extract_cwe_prefix(predicted_label)

                    is_correct = (predicted_cwe_prefix == expected_cwe_prefix)

                    if is_correct:
                        results.append((file_path, predicted_cwe_prefix, True))
                        correct_predictions += 1
                    else:
                        results.append((file_path, predicted_cwe_prefix, False))
                except Exception as e:
                    results.append((file_path, f"Error processing: {e}", False))

    print(f"{bcolors.HEADER}[RESULTS]{bcolors.ENDC}")
    for file_path, classification, is_correct in results:
        if classification == "Skipped":
            print(f"{bcolors.GRAY}[FILE]{bcolors.ENDC} {file_path} -> {bcolors.GRAY}Skipped{bcolors.ENDC}")
        elif classification == "Unknown":
            print(f"{bcolors.OKBLUE}[FILE]{bcolors.ENDC} {file_path} -> {bcolors.WARNING}Unknown{bcolors.ENDC}")
        elif is_correct:
            print(f"{bcolors.OKBLUE}[FILE]{bcolors.ENDC} {file_path} -> {bcolors.OKGREEN}{classification}{bcolors.ENDC}")
        else:
            expected_cwe_prefix = extract_cwe_prefix(os.path.basename(file_path))
            print(f"{bcolors.OKBLUE}[FILE]{bcolors.ENDC} {file_path} -> {bcolors.FAIL}{classification}{bcolors.ENDC} (Expected: {expected_cwe_prefix})")

    # Calculate accuracy
    # Only consider files that were not skipped
    classified_files = total_files - sum(1 for _, classification, _ in results if classification == "Skipped")
    accuracy = (correct_predictions / classified_files) * 100 if classified_files > 0 else 0
    print(f"\n{bcolors.OKCYAN}[INFO]{bcolors.ENDC} Accuracy: {accuracy:.2f}% ({correct_predictions}/{classified_files})")

# Main function
def main():
    parser = argparse.ArgumentParser(description="CWE vulnerability classifier for C++ files")
    parser.add_argument("--file", type=str, required=True, help="Directory containing C++ files for classification")
    parser.add_argument("--model", type=str, default="random_forest_cwe_classifier.pkl", help="Path to the trained model")
    parser.add_argument("--label_map", type=str, default="cwe_label_map.json", help="Path to the label mapping file")
    parser.add_argument("--word2vec", type=str, default="word2vec.model", help="Path to the Word2Vec model")
    args = parser.parse_args()

    # Verify that the provided directory exists
    if not os.path.isdir(args.file):
        print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} The specified directory does not exist: {args.file}")
        exit(1)

    label_map = load_label_map(args.label_map)
    model = load_random_forest_model(args.model)
    word2vec_model = load_word2vec_model(args.word2vec)

    classify_with_random_forest(args.file, model, label_map, word2vec_model)

if __name__ == "__main__":
    main()
