# Classificador de Vulnerabilidades CWE

## Índice

- [Objetivo do Projeto](#objetivo-do-projeto)
- [Descrição dos Scripts](#descrição-dos-scripts)
  - [Carregar e Processar Conjuntos de Dados](#carregar-e-processar-conjuntos-de-dados)
  - [Pipeline de Treinamento](#pipeline-de-treinamento)
  - [Classificador](#classificador)
- [Pipelines](#pipelines)
  - [Pipeline do Script `load_and_process_cwe_datasets.py`](#pipeline-do-script-load_and_process_cwe_datasetspy)
  - [Pipeline do Script `train_cwe_model.py`](#pipeline-do-script-train_cwe_modelpy)
- [Descrição Técnica](#descrição-técnica)
  - [Vetorização com Tree-Sitter e Word2Vec](#vetorização-com-tree-sitter-e-word2vec)
- [Melhorias no Projeto - Pré-processamento](#melhorias-no-projeto---pré-processamento)
  - [Modo 1: Foco nos Identificadores](#modo-1-foco-nos-identificadores)
  - [Modo 2: Embeddings Enriquecidos](#modo-2-embeddings-enriquecidos)
  - [Benefícios](#benefícios)
- [Apresentação e Trabalho](#apresentação-e-trabalho)
  - [Modelo Random Forest](#modelo-random-forest)
  - [Resultados Finais (métricas)](#resultados-finais-métricas)
  - [Adicionando Estratificação](#adicionando-estratificação)
- [Descrição dos Parâmetros](#descrição-dos-parâmetros)
- [Base de Dados](#base-de-dados)
- [Dependências](#dependências)
  - [Instalação das Dependências](#instalação-das-dependências)

## Objetivo do Projeto

Este projeto tem como objetivo classificar falhas de segurança em códigos C/C++ utilizando modelos de aprendizado de máquina. O sistema foi desenvolvido para identificar vulnerabilidades a partir de um conjunto de falhas previamente treinadas, utilizando dois tipos de modelo: Random Forest e Codebert.

## Descrição dos Scripts

1. **Carregar e Processar Conjuntos de Dados (`load_and_process_cwe_datasets.py`):**  
   Este script realiza o carregamento e o processamento de arquivos de código C/C++, utilizando a biblioteca Tree-Sitter para analisar a estrutura sintática e extrair tokens relevantes. Ele associa rótulos de vulnerabilidades (CWE) aos códigos com base em suas categorias e organiza os dados em um formato estruturado. Além disso, o script salva os dados processados em arquivos JSON, prontos para serem utilizados em etapas posteriores, como treinamento de modelos de classificação. O objetivo é criar uma base sólida para identificar e classificar falhas de segurança de maneira eficiente.

   **Execução:**
   ```
   python3 load_and_process_cwe_datasets.py --base_path <diretório_dos_dados> --minFiles <número_mínimo_de_arquivos> --maxFiles <número_máximo_de_arquivos>
   ```
   1.1 Exemplo de Dados Processados  

   Após o processamento de um arquivo de código C/C++, o script gera tokens que representam os elementos relevantes do código. Esses tokens incluem identificadores, literais, diretivas de pré-processador e outros componentes sintáticos importantes. Abaixo, apresentamos um exemplo do formato dos dados processados:  

   ```json
   {
      "code": [
         "#include \"std_testcase.h\"",
         "\"std_testcase.h\"",
         "#ifndef _WIN32\n#include <wchar.h>\n#endif",
         "_WIN32",
         "#include <wchar.h",
         "#define SRC_STRING \"AAAAAAAAAA\"",
         "SRC_STRING",
         "\"AAAAAAAAAA\"",
         "namespace",
         "CWE122_Heap_Based_Buffer_Overflow__cpp_CWE193_char_ncpy_54",
         "#ifndef OMITBAD",
         "badSink_d",
         "data",
         "badSink_c",
         "data",
         "badSink_d",
         "data",
         "#ifndef OMITGOOD",
         "goodG2BSink_d",
         "data",
         "goodG2BSink_c",
         "data",
         "goodG2BSink_d",
         "data"
      ]
   }
   ```

   1.2 Rótulos de vunerabilidade

   ```json
   {
      "CWE122_Heap_Based_Buffer_Overflow": 0,
      "CWE124_Buffer_Underwrite": 1,
      "CWE126_Buffer_Overread": 2,
      "CWE127_Buffer_Underread": 3,
      "CWE134_Uncontrolled_Format_String": 4,
      "CWE190_Integer_Overflow": 5,
      "CWE191_Integer_Underflow": 6,
      "CWE194_Unexpected_Sign_Extension": 7,
      "CWE195_Signed_to_Unsigned_Conversion_Error": 8,
      "CWE197_Numeric_Truncation_Error": 9,
      "CWE23_Relative_Path_Traversal": 10,
      "CWE369_Divide_by_Zero": 11,
      "CWE36_Absolute_Path_Traversal": 12,
      "CWE400_Resource_Exhaustion": 13,
      "CWE401_Memory_Leak": 14,
      "CWE415_Double_Free": 15,
      "CWE457_Use_of_Uninitialized_Variable": 16,
      "CWE590_Free_Memory_Not_on_Heap": 17,
      "CWE690_NULL_Deref_From_Return": 18,
      "CWE762_Mismatched_Memory_Management_Routines": 19,
      "CWE789_Uncontrolled_Mem_Alloc": 20,
      "CWE78_OS_Command_Injection": 21
   }
   ```

2. **Pipeline de Treinamento (`train_model.py`):**  
   Este script realiza o carregamento e o pré-processamento dos dados, utilizando modelos Word2Vec para transformar o código-fonte em vetores numéricos. Em seguida, treina um modelo de classificação Random Forest com esses vetores para identificar vulnerabilidades em códigos C/C++. Além disso, o script gera arquivos JSON contendo os vetores processados, os rótulos correspondentes e o mapeamento de falhas (CWE), que podem ser utilizados para etapas posteriores de classificação e análise.

   **Execução:**
   ```
   python3 train_model.py --base_path <diretório_dos_dados> --minFiles <número_mínimo_de_arquivos> --maxFiles <número_máximo_de_arquivos>
   ```
   ```json
   [
    [
        0.12609584629535675,
        -1.8871763944625854,
        0.859345555305481,
        0.39093509316444397,
        -0.8027384281158447,
        1.0338129997253418,
        -0.6320635676383972,
        1.360384225845337,
        -0.11257253587245941,
        0.5287671685218811,
        -0.0769924744963646,
        0.1340981423854828,
        -0.7386022806167603,
        0.22158850729465485,
        -0.0841221883893013,
        1.1517157554626465,
        -1.2060550451278687,
        ...
   ```

3. **Classificador (`classify.py`):**  
   Este script classifica arquivos de código C++ em categorias de falhas de segurança (CWEs) previamente treinadas, utilizando modelos de aprendizado de máquina como Random Forest ou CodeBERT. Ele processa o código, transforma tokens em vetores numéricos com um modelo Word2Vec e verifica o mapeamento de rótulos para identificar vulnerabilidades. Com suporte à análise de múltiplos arquivos em um diretório, o script fornece relatórios detalhados de classificação, precisão e erros, permitindo a validação eficiente de falhas no código-fonte.

   **Execução:**
   ```
   python3 cwe_classification.py --file non_dataset_code --model random_forest_cwe_classifier.pkl --label_map cwe_label_map.json --word2vec word2vec.model
   ```
## Pipelines

   ### 🚀 **Pipeline do Script `load_and_process_cwe_datasets.py`**
   📁 **Coleta de Dados** (Identificação e seleção de arquivos `.c` e `.cpp` nas subpastas especificadas)  
     ↓  
   🧹 **Pré-processamento** (Análise sintática com Tree-sitter, extração e normalização de tokens relevantes)  
     ↓  
   🤖 **Treinamento de Modelos de Embeddings** (Treinamento de modelos Word2Vec para capturar relações semânticas entre tokens)  
     ↓  
   📈 **Visualização de Embeddings** (Redução de dimensionalidade com PCA e plotagem dos embeddings para visualização)  
     ↓  
   💾 **Salvamento de Resultados** (Exportação dos dados processados e dos modelos treinados para arquivos como `cwe_data.json` e `word2vec.model`)  


### 🚀 **Pipeline do Script `train_cwe_model.py`**

📁 **Carregamento de Dados** (Importação dos arquivos JSON contendo os códigos e mapeamento de rótulos)  
  ↓  
📊 **Vetorização com Word2Vec** (Conversão dos códigos-fonte em vetores numéricos utilizando o modelo Word2Vec treinado)  
  ↓  
🔀 **Divisão Treino/Teste** (Separação dos dados vetorizados em conjuntos de treino e teste, com possibilidade de estratificação)  
  ↓  
🧠 **Treinamento do Random Forest** (Treinamento do classificador Random Forest com os dados de treino)  
  ↓  
📈 **Avaliação do Modelo** (Geração de relatório de classificação e matriz de confusão para avaliar o desempenho do modelo)  
  ↓  
💾 **Salvamento do Modelo** (Exportação do modelo Random Forest treinado para um arquivo `.pkl` para uso futuro)


## Descrição Técnica

O modelo foi treinado utilizando um conjunto de dados contendo códigos C/C++ com falhas de segurança conhecidas, categorizadas em diferentes tipos de vulnerabilidades, como buffer overflow, falhas de controle de fluxo, entre outras.

### Vetorização com Tree-Sitter e Word2Vec
- **Tree-Sitter:** Utilizado para análise sintática e geração de tokens dos códigos.
- **Word2Vec:** Gera vetores de palavras a partir dos tokens processados.
- **Random Forest:** Modelo de classificação que utiliza os vetores gerados para identificar as vulnerabilidades.

## Melhorias no Projeto - Pré-processamento

A evolução no pré-processamento de código representa um marco importante no desenvolvimento do projeto, permitindo uma análise mais detalhada e assertiva. O pré-processamento agora conta com dois modos complementares que aprimoram significativamente os embeddings gerados:

### Modo 1: Foco nos Identificadores (Cleyton)

Neste modo, o pré-processamento prioriza apenas os elementos semânticos do código, como nomes de variáveis, funções e classes. Essa abordagem simplificada é eficiente para tarefas que dependem exclusivamente da identificação dos componentes principais do código, sendo ideal para cenários com menor complexidade.

### Modo 2: Embeddings Enriquecidos (Fausto)

O segundo modo vai além dos identificadores, incorporando elementos semânticos e estruturais do código. Ele inclui:

- **Identificadores:** Variáveis, funções, classes.
- **Estruturas de Controle:** Instruções como `if`, `for`, e `while`.
- **Diretivas do Pré-processador:** Como `#include` e `#define`.
- **Literais:** Strings, números e valores específicos do código.

#### Benefícios
- **Melhor compreensão do fluxo de execução.**
- **Capacidade de identificar relações contextuais.**
- **Maior detalhamento dos padrões de codificação.**

## Apresentação e Trabalho

### Modelo Random Forest

#### Pipeline de Processamento e Vetorização de Código com Tree-Sitter e Word2Vec (DB minimizado)

Foi utilizado o seguinte comando para processar os conjuntos de dados:
```
python3 load_and_process_cwe_datasets.py --base_path /home/fbiaso/dados/Documentos/PLN/C/testcases --maxFiles 16 --minFiles 10
```

#### Distribuição das Classes CWE (DB minimizado)

A execução do script gerou uma base de dados cuja distribuição das classes pode ser visualizada na imagem abaixo:

<img src="imgs/cwe_class_distribution.png" alt="Distribuição das Classes CW" width="100%">

---
#### Exemplo: Treinamento do Modelo // Teste 1 // [test_size=0.3, n_estimators=100, vector_size=100]

O modelo foi treinado utilizando o seguinte comando:
```
python3 ./train_cwe_model.py --data_dir ./ --word2vec_model_path word2vec.model
```

---

#### Exemplo: Treinamento do Modelo // Teste 2 // [test_size=0.3, n_estimators=200, vector_size=100]

O modelo foi treinado utilizando o seguinte comando:
```
 python3 ./train_cwe_model.py --data_dir ./ --word2vec_model_path word2vec.model --test_size 0.3 --random_state 44 --n_estimators 200 --vector_size 100
```

---

#### Resultados Finais (métricas)

Este documento apresenta um relatório com os parâmetros de pré-processamento, modelo utilizado, acurácia geral, e outros detalhes relacionados à classificação.

| Parâmetros                                        | Tipo de Pré-processamento | Modelo        | Acurácia Geral | Quantidade Total de Dados | Quantidade de Classes | Método de Representação  |
|---------------------------------------------------|---------------------------|---------------|----------------|---------------------------|-----------------------|----------------------|
| test_size=0.3, ***n_estimators=50***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 80%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, ***n_estimators=100***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 87%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, ***n_estimators=200***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 87%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, ***n_estimators=1000***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 87%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, n_estimators=100, ***vector_size=20***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, n_estimators=100, ***vector_size=40***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, n_estimators=100, ***vector_size=80***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 123                       | 7                     | Word2Vec                 |
| test_size=0.3, n_estimators=100, ***vector_size=160***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 123                       | 7                     | Word2Vec                 |
| ***test_size=0.1***, n_estimators=100, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 90%            | 123                       | 7                     | Word2Vec                 |
| ***test_size=0.15***, n_estimators=100, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 93%            | 123                       | 7                     | Word2Vec                 |
| ***test_size=0.20***, n_estimators=100, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 90%            | 123                       | 7                     | Word2Vec                 |
| ***test_size=0.16***, n_estimators=100, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 100%            | 123                       | 7                     | Word2Vec                 |

---
### Adicionando Estratificação

#### Pipeline de Processamento e Vetorização de Código com Tree-Sitter e Word2Vec (DB ~ 1000 arquivos/categoria)

Foi utilizado o seguinte comando para processar os conjuntos de dados:
```
python3 load_and_process_cwe_datasets.py --base_path /home/fbiaso/dados/Documentos/PLN/C/testcases --minFiles 1000 --maxFiles 1200
```

#### Distribuição das Classes CWE (DB ~ 1000 arquivos/categoria)

---
#### Exemplo: Treinamento do Modelo // Teste 1 // [test_size=0.3, n_estimators=100, vector_size=100]

O modelo foi treinado utilizando o seguinte comando:
```
python3 ./train_cwe_model.py --data_dir ./ --word2vec_model_path word2vec.model
```

---

#### Exemplo: Treinamento do Modelo // Teste 2 // [test_size=0.3, n_estimators=200, vector_size=100]

O modelo foi treinado utilizando o seguinte comando:
```
 python3 ./train_cwe_model.py --data_dir ./ --word2vec_model_path word2vec.model --test_size 0.3 --random_state 44 --n_estimators 200 --vector_size 100 -stratification yes
```
---

#### Resultados Finais com estratificação (métricas)

Este documento apresenta um relatório com os parâmetros de pré-processamento, modelo utilizado, acurácia geral, e outros detalhes relacionados à classificação.

| Parâmetros                                        | Tipo de Pré-processamento | Modelo        | Acurácia Geral | Quantidade Total de Dados | Quantidade de Classes | Método de Representação | Percentual de Acerto| F1-Score |
|---------------------------------------------------|---------------------------|---------------|----------------|---------------------------|-----------------------|--------------------------|----------------------|----------------------|
|test_size=0.3, ***n_estimators=50***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 89%           | 26283                       | 21                     | Word2Vec                 | 13,64%                |89%|
|test_size=0.3, ***n_estimators=100***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 89%            | 26283                       | 21                     | Word2Vec                 | 18,18%                  |89%|
| test_size=0.3, ***n_estimators=200***, vector_size=100  | Tree-Sitter (ABS)         | Random Forest | 89%            | 26283                       | 21                     | Word2Vec                 | 13,64%                  |89%|
|test_size=0.3, n_estimators=100, ***vector_size=20***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 26283                       | 21                     | Word2Vec                 |  18,18%                   |89%|
|test_size=0.3, n_estimators=100, ***vector_size=40***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 26283                       | 21                     | Word2Vec                 | 18,18%                 |89%|
|test_size=0.3, n_estimators=100, ***vector_size=80***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 26283                       | 21                     | Word2Vec                 | 18,18%                   |89%|
| test_size=0.3, n_estimators=100, ***vector_size=160***  | Tree-Sitter (ABS)         | Random Forest | 83%            | 26283                       | 21                     | Word2Vec                 | 18,18%                  |89%|
| test_size=0.3, ***n_estimators=500***, ***vector_size=300***  | Tree-Sitter (ABS)         | Random Forest | 89%            | 26283                       | 21                     | Word2Vec                 | 9%                  |89%|

#### Descrição dos Parâmetros

- **Parâmetros**: Configurações utilizadas no modelo, como tamanho do teste (`test_size`), número de estimadores (`n_estimators`), e tamanho do vetor (`vector_size`).
- **Tipo de Pré-processamento**: Técnica utilizada para pré-processamento dos dados.
- **Modelo**: O modelo de aprendizado de máquina utilizado.
- **Acurácia Geral**: Percentual de acerto alcançado pelo modelo.
- **Quantidade Total de Dados**: Número de amostras utilizadas no modelo.
- **Quantidade de Classes**: Número de classes para a classificação.
- **Método de Representação**: Método usado para representar as palavras.
- **Percentual de Acerto**: Acurácia expressa como percentual de acerto.
- **F1-Score**: Média harmônica entre precisão e recall.

## Base de dados
- Juliet C/C++ 1.3 Test suite #112
- https://samate.nist.gov/SARD/test-suites/112

## Dependências

O projeto foi desenvolvido para rodar em **plataforma Ubuntu** e possui as seguintes dependências:

- **Python 3.x**
- **pip** (gerenciador de pacotes Python)

### Instalação das Dependências

1. **Crie um ambiente virtual:**
    ```
    python3 -m venv tree-sitter-env
    source tree-sitter-env/bin/activate
    ```

2. **Instale as dependências:**
    ```
    pip install -r requirements.txt
    ```

**Conteúdo do `requirements.txt`:**
```
tree-sitter
gensim
scikit-learn
transformers
torch
numpy
pandas
```
