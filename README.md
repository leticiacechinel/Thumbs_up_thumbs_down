# Thumbs_up_thumbs_down

## **Instrução de uso 1 (montando container docker):**
baixar o diretório completo
dentro do diretório no terminal, rodar **sudo docker build -t certiapp .**
após montar, **sudo docker run -p 8501:8501 certiapp:latest**
Usar o network URL que aparecerá para abrir o aplicativo no seu browser



## **Instrução de uso 2 (ambiente conda):**
baixar o diretório completo

criar ambiente conda utilizando o arquivo certi.yml, através do comando

conda env create -f certi.yml

dentro do diretório no terminal, rodar o comando streamlit run certi_app.py

selecionar as imagens entre as pastas, a interface já retorna o resultado do modelo

é possível adicionar nas pastas outras imagens de "thumbs up" e "thumbs down" e testar a performance do modelo.

## **Concepção do modelo**:
arquivo: model_certi.py
* Data augmentation: aumentar e generalizar a base. Nessa etapa foram geradas novas imagens a partir de uma base bastante reduzida (menos de 40 imagens) geradas por mim para treinar o modelo. Foram criadas novas imagens utilizando zoom, pequenos deslocamentos, espelhamento e rotação, expondo a rede a maiores variaçoes.
* Redução de dimensionalidade (escala de cinza), transformação para ponto flutuante, normalização.
* Foi criado um modelo supervisionado utilizando redes neurais convolucionais, com auxílio das bibliotecas tensorflow e keras.

![alt text](https://github.com/leticiacechinel/Thumbs_up_thumbs_down/blob/master/CERTI_TESTE/certi.png)

**A curva de imagens para teste acompanhou razoavelmente a curva de treino, os resultados para os dados de teste terminaram em uma precisão de 90%**

**Observações**:
Também foram testadas outras possibilidades, como adaptar um modelo pronto, formatando as imagens e alterando apenas a camada de saída. Também foi levantada a possibilidade de se utilizar o n-shot learning, não supervisionado e ideal para comparações e bases pequenas. Apesar disso seguiu-se com as CNNs.
Como o resultado foi satisfatório, não se verificou a utilização de outros filtros de pré processamento.

Summary final para o modelo de CNN proposto após ajuste de parâmetros

![alt text](https://github.com/leticiacechinel/Thumbs_up_thumbs_down/blob/master/CERTI_TESTE/summary_final.png)


