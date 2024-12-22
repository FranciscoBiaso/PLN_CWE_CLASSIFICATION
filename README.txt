README: Classificador de Vulnerabilidades CWE

Este projeto tem como objetivo classificar falhas de segurança em códigos C/C++ utilizando modelos de aprendizado de máquina. Utilizamos dados da base de testes do **SAMATE CWE** (https://samate.nist.gov/SARD/test-suites/112), que contém códigos vulneráveis com diferentes falhas de segurança. O projeto foi desenvolvido para identificar falhas a partir de um conjunto de falhas previamente treinadas, utilizando dois tipos de modelos: **Random Forest** e **CodeBERT**.

Descrição dos Scripts

1. **Pipeline de Treinamento (train_model.py):** Este script é responsável por carregar e pré-processar os dados da base CWE, treinando um modelo **Random Forest** com os vetores extraídos do código utilizando **Word2Vec**. Ele também gera arquivos JSON contendo os vetores, rótulos e mapeamento das falhas, necessários para a classificação posterior. Para treinar o modelo, basta rodar o script da seguinte forma:

    python train_model.py --base_path <diretório_dos_dados> --minFiles <número_mínimo_de_arquivos> --maxFiles <número_máximo_de_arquivos>

2. **Classificador (classify.py):** Este script é utilizado para classificar novos arquivos de código C++ em uma das falhas (CWE) treinadas anteriormente, utilizando os modelos **Random Forest** ou **CodeBERT**. O classificador pode ser executado em dois modos: **Random Forest** ou **CodeBERT**, dependendo da escolha do usuário. O modelo Random Forest é baseado em vetores gerados pelo **Word2Vec**, enquanto o modelo **CodeBERT** é uma rede neural treinada para a análise de código. Para utilizar o classificador, execute o script com os parâmetros necessários:

    python classify.py --file <caminho_do_arquivo> --model_type <random_forest|codebert> --model <caminho_para_o_modelo> --label_map <caminho_para_o_label_map>

Descrição Técnica

O **dataset** utilizado para treinar o modelo é o **CWE Dataset** do SAMATE (https://samate.nist.gov/SARD/test-suites/112). Este conjunto contém códigos C/C++ com falhas conhecidas, categorizadas em diferentes tipos de vulnerabilidades, como buffer overflow, falhas de controle de fluxo, entre outras. Para gerar os vetores de características do código, usamos o **Tree-Sitter** para análise sintática e o **Word2Vec** para gerar vetores de palavras, que são alimentados no modelo **Random Forest** para classificar as falhas.

No caso do modelo **CodeBERT**, utilizamos o modelo pré-treinado da Hugging Face para tokenizar e classificar os códigos diretamente, sem a necessidade de vetorização manual. A classificação do arquivo é feita com base nos tokens extraídos e no treinamento realizado nos dados.

Objetivo do Projeto

O objetivo deste projeto é construir um sistema que, ao receber um código C/C++, seja capaz de classificar qual tipo de falha de segurança (CWE) o código possui, com base em falhas conhecidas previamente treinadas. Esse sistema pode ser útil para ferramentas de análise estática de código, ajudando desenvolvedores e equipes de segurança a identificar vulnerabilidades de maneira automática e eficiente.
