# 🌿 PlantVision AI

Sistema web local para diagnóstico de doenças em plantas utilizando Inteligência Artificial e visão computacional.

A aplicação permite o envio de imagens de plantas através do navegador, realizando a classificação automática da condição da planta com base em um modelo de Deep Learning, além de fornecer nível de confiança e uma visualização interpretável com Grad-CAM.

---

## 🚀 Tecnologias utilizadas

* Python
* FastAPI
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## 🧠 Funcionalidades

* Upload de imagens de plantas via interface web
* Classificação automática de doenças (modelo de IA)
* Exibição de nível de confiança (confidence score)
* Top 3 previsões com probabilidades
* Visualização com Grad-CAM (explicabilidade do modelo)
* Execução 100% local (sem dependência de APIs externas)

---

## 📸 Demonstração

<img width="1896" height="873" alt="image" src="https://github.com/user-attachments/assets/55415df8-f85a-4530-a46e-2f8667e44921" />


---

## ⚙️ Instalação

```bash
pip install -r requirements.txt
```

---

## ▶️ Executando a aplicação

Inicie o servidor FastAPI:

```bash
uvicorn app:app --reload --port 8000
```

ou

```bash
python -m uvicorn app:app --reload --port 8000
```

Acesse no navegador:

```
http://localhost:8000
```

---

## 🧪 Como usar

1. Clique em "Choose File" e selecione uma imagem de planta (JPG, PNG, etc.)
2. Clique em "Analyze"
3. Aguarde alguns segundos para o processamento
4. Visualize:

   * Diagnóstico
   * Nível de confiança
   * Top 3 previsões
   * Grad-CAM (mapa de calor)

---

## 🧪 Imagem de teste

Uma imagem de exemplo está disponível em:

```
codigo/teste.jpg
```

---

## 📂 Estrutura do projeto

* `app.py` — Backend com FastAPI
* `index.html` — Interface web
* `static/` — Arquivos estáticos (JS, imagens e outputs)
* `src/diagnosis/` — Serviço de diagnóstico reutilizável
* `codigo/` — Modelo treinado e scripts originais

---

## 📌 Observações

* A aplicação roda completamente local
* Não requer conexão com internet após instalação
* Modelo utilizado: `codigo/plant_model.h5`
* Saídas do Grad-CAM são salvas em: `static/outputs/`

---

## 💡 Possíveis melhorias futuras

* Deploy em nuvem (AWS, GCP, Azure)
* API pública para integração
* Suporte a mais tipos de plantas e doenças
* Interface responsiva/mobile
* Otimização do modelo para performance

---
