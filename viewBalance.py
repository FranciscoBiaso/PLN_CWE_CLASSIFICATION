import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # Força backend não interativo
import matplotlib.pyplot as plt

# 1. Carregar dados salvos
def load_data():
    with open("cwe_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("cwe_labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open("cwe_label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    print(f"Dados carregados: {len(data)} exemplos")
    return data, labels, label_map

# Carregar os dados
data, labels, label_map = load_data()

# Contagem das classes
class_counts = Counter(labels)

# Converter os índices numéricos para nomes das CWEs
label_to_cwe = {v: k for k, v in label_map.items()}
class_counts_named = {label_to_cwe[label]: count for label, count in class_counts.items()}

# Ordenar as CWEs pela contagem
class_counts_named = dict(sorted(class_counts_named.items(), key=lambda item: item[1], reverse=True))

# Salvar a contagem em texto
def save_class_distribution_to_text(class_counts_named, filename="cwe_class_distribution.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Distribuição das Classes CWE:\n")
        f.write("="*30 + "\n")
        for cwe, count in class_counts_named.items():
            line = f"{cwe}: {count}\n"
            print(line.strip())  # Exibe no terminal
            f.write(line)
    print(f"\nDistribuição salva em '{filename}'")

# Visualizar em gráfico e salvar como imagem
def plot_class_distribution(class_counts_named, output_filename="cwe_class_distribution.png"):
    plt.figure(figsize=(14, 8))
    plt.bar(class_counts_named.keys(), class_counts_named.values(), color='steelblue')
    plt.xticks(rotation=90)  # Rotaciona os rótulos no eixo X
    plt.title("Distribuição das Classes CWE")
    plt.xlabel("Classes CWE")
    plt.ylabel("Número de Exemplos")
    plt.tight_layout()
    plt.savefig(output_filename)  # Salva o gráfico como imagem
    print(f"Gráfico salvo como '{output_filename}'")

# Salvar em texto e gerar o gráfico
save_class_distribution_to_text(class_counts_named)
plot_class_distribution(class_counts_named)
