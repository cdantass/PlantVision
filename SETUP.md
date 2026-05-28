## PlantVision - Sistema de Diagnóstico de Doenças em Plantas

### 🚀 Instalação e Execução

#### 1. **Instalar Dependências**
```bash
pip install -r requirements.txt
```

#### 2. **Criar o Modelo Inicial (PRIMEIRO PASSO OBRIGATÓRIO)**
Execute este comando **uma única vez** para criar o arquivo `plant_model.h5`:
```bash
cd codigo
python create_initial_model.py
cd ..
```

Isso criará um modelo MobileNetV2 pré-treinado (~90MB) que será usado pela aplicação.

#### 3. **Executar a Aplicação**
```bash
python app.py
```

A aplicação estará disponível em: `http://localhost:8000`

---

### 📋 Funcionalidades

- **Análise de Imagens**: Upload de fotos de plantas para diagnóstico
- **Diagnóstico Automático**: Classificação de 15 tipos de doenças em plantas
- **GradCAM**: Visualização em heatmap das áreas identificadas como doentes
- **Confiança**: Métrica de confiabilidade da previsão
- **Interface Web**: Interface intuitiva em português

---

### 📁 Estrutura do Projeto

```
PlantVision/
├── app.py                    # FastAPI backend
├── index.html                # Frontend HTML/CSS/JS
├── requirements.txt          # Dependências Python
├── codigo/
│   ├── plant_model.py        # Funções de classificação
│   ├── plant_gradcam.py      # GradCAM visualization
│   ├── create_initial_model.py  # Script para criar modelo inicial
│   └── dataset.yaml          # Configurações do dataset
├── src/
│   └── diagnosis/
│       ├── service.py        # Serviço de análise
│       └── types.py          # Tipos de dados
└── static/
    ├── img/
    │   └── logo_plant_vision.png
    ├── js/
    │   └── app.js            # Frontend JavaScript
    └── outputs/              # GradCAM outputs salvos aqui
```

---

### 🔧 Correções Implementadas

- ✅ **app.py**: Reordenadas definições de `CLASS_TRANSLATIONS` e `generate_plant_feedback()` para evitar NameError
- ✅ **index.html**: Adicionado elemento spinner faltante e corrigidas tags HTML
- ✅ **service.py**: Verificadas importações e lógica de análise
- ✅ **plant_gradcam.py**: Corrigida importação para usar caminho absoluto
- ✅ **Modelo Inicial**: Script criado para gerar modelo pré-treinado MobileNetV2

---

### 🐛 Troubleshooting

**Erro: "FileNotFoundError: plant_model.h5 not found"**
- Solução: Execute `python codigo/create_initial_model.py` primeiro

**Erro: "CUDA not available"**
- Solução: TensorFlow usará CPU automaticamente (mais lento)

**Imagens não sendo exibidas**
- Verifique se a pasta `static/` existe e tem as subpastas `img/`, `js/` e `outputs/`

---

### 📝 Notas

- O modelo inicial é genérico. Para melhor desempenho, treine o modelo com seus dados
- As imagens de entrada devem ser plantas (224x224 pixels)
- GradCAM mostra onde o modelo identificou a doença
- Confiança < 60% é considerada não confiável

---

### 👨‍💻 Desenvolvedor

Desenvolvido com FastAPI, TensorFlow e JavaScript vanilla.
