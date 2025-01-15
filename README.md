# Classificador de Vulnerabilidades CWE

Este projeto tem como objetivo classificar falhas de segurança em códigos C/C++ utilizando modelos de aprendizado de máquina. O projeto foi desenvolvido para identificar falhas a partir de um conjunto de falhas previamente treinadas, utilizando dois tipos de modelos: **Random Forest** e **CodeBERT**.

## Descrição dos Scripts

1. **Carregar e Processar Conjuntos de Dados (load_and_process_cwe_datasets.py):** Este script tem como objetivo carregar e processar os arquivos de código C/C++, pré-processando-os utilizando **Tree-Sitter** para gerar tokens e associar rótulos de falhas baseados no tipo de vulnerabilidade (CWE). Os dados processados são então salvos em arquivos JSON que podem ser usados posteriormente para treinar modelos de classificação. Para executar o script, use o seguinte comando:

    ```bash
    python3 load_and_process_cwe_datasets.py --base_path <diretório_dos_dados> --minFiles <número_mínimo_de_arquivos> --maxFiles <número_máximo_de_arquivos>
    ```

2. **Pipeline de Treinamento (train_model.py):** Este script é responsável por carregar e pré-processar os dados, treinando um modelo **Random Forest** com os vetores extraídos do código utilizando **Word2Vec**. Ele também gera arquivos JSON contendo os vetores, rótulos e mapeamento das falhas, necessários para a classificação posterior. Para treinar o modelo, basta rodar o script da seguinte forma:

    ```bash
    python3 train_model.py --base_path <diretório_dos_dados> --minFiles <número_mínimo_de_arquivos> --maxFiles <número_máximo_de_arquivos>
    ```

3. **Classificador (classify.py):** Este script é utilizado para classificar novos arquivos de código C++ em uma das falhas (CWE) treinadas anteriormente, utilizando os modelos **Random Forest** ou **CodeBERT**. O classificador pode ser executado em dois modos: **Random Forest** ou **CodeBERT**, dependendo da escolha do usuário. O modelo Random Forest é baseado em vetores gerados pelo **Word2Vec**, enquanto o modelo **CodeBERT** é uma rede neural treinada para a análise de código. Para utilizar o classificador, execute o script com os parâmetros necessários:

    ```bash
    python3 classify.py --file <caminho_do_arquivo> --model_type <random_forest|codebert> --model <caminho_para_o_modelo> --label_map <caminho_para_o_label_map>
    ```

## Descrição Técnica

O modelo foi treinado utilizando um conjunto de dados contendo códigos C/C++ com falhas de segurança conhecidas, categorizadas em diferentes tipos de vulnerabilidades, como buffer overflow, falhas de controle de fluxo, entre outras. Para gerar os vetores de características do código, utilizamos o **Tree-Sitter** para análise sintática e o **Word2Vec** para gerar vetores de palavras. Esses vetores foram então alimentados no modelo **Random Forest** para a classificação das falhas.

No caso do modelo **CodeBERT**, utilizamos o modelo pré-treinado da Hugging Face para tokenizar e classificar os códigos diretamente, sem a necessidade de vetorização manual. A classificação do arquivo é feita com base nos tokens extraídos e no treinamento realizado nos dados.

## Melhorias no Projeto - Pré-processamento

A introdução de dois modos distintos de pré-processamento de código no projeto representa um avanço significativo na análise de dados e na geração de embeddings mais ricos e precisos. O primeiro modo adota uma abordagem simplificada, concentrando-se exclusivamente nos identificadores, como nomes de variáveis, funções e classes.

Por outro lado, o segundo modo adota uma abordagem mais abrangente, enriquecendo os embeddings ao incluir tanto elementos semânticos (e.g., identificadores como `data`, `strncpy`, `SRC_STRING`) quanto estruturais (e.g., estruturas de controle como `if` e `for`, literais como números e strings, e diretivas do pré-processador como `#include` e `#define`). Ao integrar essas informações adicionais, o modelo é capaz de capturar não apenas os elementos que descrevem o que o código faz, mas também como ele é estruturado e executado. Isso permite que os embeddings reflitam com maior precisão as relações lógicas e contextuais presentes no código, tornando-os mais ricos e informativos.

Esse segundo modo melhora significativamente a qualidade dos embeddings, pois fornece uma visão completa do código, incorporando tanto o fluxo lógico quanto os valores e padrões específicos que afetam o comportamento do programa. A inclusão de estruturas de controle e diretivas do pré-processador ajuda a modelar o fluxo de execução e as dependências externas, enquanto os literais fornecem detalhes específicos, como valores críticos e mensagens exibidas. Como resultado, o modelo se torna mais eficiente na identificação de vulnerabilidades, padrões de codificação e erros contextuais.

Essa evolução no pré-processamento amplia a capacidade do modelo de realizar análises mais detalhadas e confiáveis, especialmente em tarefas de maior complexidade. Ao enriquecer a granularidade e o contexto dos dados de entrada, o segundo modo eleva a qualidade dos embeddings gerados, permitindo ao modelo capturar padrões mais complexos e fornecer classificações mais precisas, fortalecendo o potencial analítico do projeto.

## Objetivo do Projeto

O objetivo deste projeto é construir um sistema que, ao receber um código C/C++, seja capaz de classificar qual tipo de falha de segurança (CWE) o código possui, com base em falhas conhecidas previamente treinadas.

## Dependências

O projeto foi desenvolvido para rodar em **plataforma Ubuntu** e possui as seguintes dependências:

- **Python 3.x**
- **pip** (gerenciador de pacotes Python)

Você pode instalar as dependências do projeto utilizando o `pip`. Para isso, crie um ambiente virtual e instale as bibliotecas necessárias com os seguintes comandos:

1. Crie um ambiente virtual:

    ```bash
    python3 -m venv tree-sitter-env
    source tree-sitter-env/bin/activate
    ```

2. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

Onde o arquivo `requirements.txt` deve conter as seguintes bibliotecas (ou similares):
