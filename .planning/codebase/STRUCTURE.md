# Codebase Structure

**Analysis Date:** 2026-04-16

## Directory Layout

```text
analiseplant/
├── codigo/                    # Main Python source, model artifact, sample assets, and working dataset copy
│   ├── dataset/               # Training and validation directories used by the scripts
│   ├── __pycache__/           # Generated Python bytecode for local execution
│   ├── plant_analysis.py      # Main orchestration script
│   ├── plant_model.py         # Model construction and inference helpers
│   ├── plant_gradcam.py       # GradCAM generation pipeline
│   ├── plant_model.h5         # Trained TensorFlow/Keras model artifact
│   ├── teste.jpg              # Sample input image
│   └── teste_diagnosis.jpg    # Generated GradCAM output image
├── PlantVillage/              # Larger source dataset tree checked into the repository
│   ├── PlantVillage/          # Nested dataset copy with many disease folders
│   └── <class folders>/       # Additional dataset folders at the top level
├── .planning/                 # Generated planning and codebase mapping documents
└── requirements.txt           # Python dependency manifest
```

## Directory Purposes

**codigo/**
- Purpose: All executable project code and the local working assets it expects
- Contains: `*.py` modules, `plant_model.h5`, sample images, `dataset/`, and generated `__pycache__/`
- Key files: `codigo/plant_analysis.py`, `codigo/plant_model.py`, `codigo/plant_gradcam.py`
- Subdirectories: `dataset/train`, `dataset/val`, and `__pycache__/`

**codigo/dataset/**
- Purpose: Dataset layout actually consumed by the training script
- Contains: `train/` and `val/` splits with class-named subdirectories
- Key files: image JPG files organized by class
- Subdirectories: `Pepper__bell___Bacterial_spot`, `Pepper__bell___healthy`, `Tomato__Tomato_mosaic_virus`, `Tomato__Tomato_YellowLeaf__Curl_Virus`

**PlantVillage/**
- Purpose: Raw or duplicated source dataset checked into the repo outside the working `codigo/dataset` tree
- Contains: class folders and a nested `PlantVillage/PlantVillage/...` hierarchy
- Key files: large image sets only
- Subdirectories: many disease-class folders for pepper, potato, and tomato

**.planning/**
- Purpose: GSD planning artifacts created after codebase mapping
- Contains: codebase documentation in `.planning/codebase/`
- Key files: this codebase map set
- Subdirectories: `codebase/`

## Key File Locations

**Entry Points:**
- `codigo/plant_analysis.py` - main script for training-related helpers and demo inference execution

**Configuration:**
- `requirements.txt` - dependency versions
- No `.gitignore`, `pyproject.toml`, or environment files detected

**Core Logic:**
- `codigo/plant_model.py` - classifier construction, fine-tuning, and structured diagnosis output
- `codigo/plant_gradcam.py` - explainability heatmap generation and output export
- `codigo/plant_analysis.py` - dataset loading and orchestration logic

**Testing:**
- No dedicated test directory or test files detected

**Documentation:**
- `.planning/codebase/*.md` - generated codebase map

## Naming Conventions

**Files:**
- `snake_case.py` for Python modules such as `plant_analysis.py` and `plant_gradcam.py`
- Dataset classes use verbose folder names with double underscores, e.g. `Tomato__Tomato_mosaic_virus`
- Generated cache and output files use Python/runtime defaults, e.g. `__pycache__/` and `teste_diagnosis.jpg`

**Directories:**
- Lowercase `codigo/` for source and working assets
- Dataset directories mirror label names from PlantVillage rather than normalized code-oriented names

**Special Patterns:**
- Relative-path file access assumes execution from `codigo/`
- `__pycache__/` is generated locally and currently tracked in git status

## Where to Add New Code

**New model or preprocessing logic:**
- Primary code: `codigo/plant_model.py` or `codigo/plant_analysis.py`
- Tests: no existing home; adding `codigo/tests/` or root `tests/` would establish the first test structure
- Config if needed: root-level config files do not exist yet and would need to be introduced

**New explainability or output module:**
- Implementation: `codigo/` alongside `plant_gradcam.py`
- Shared helpers: `codigo/` as standalone modules

**New dataset or assets:**
- Working split for runtime code: `codigo/dataset/`
- Source/raw copy: `PlantVillage/` if the team keeps full datasets in repo

## Special Directories

**codigo/__pycache__/**
- Purpose: Generated Python bytecode cache
- Source: Local script execution
- Committed: Currently tracked in git status, though it would normally be ignored

**PlantVillage/**
- Purpose: Large data asset storage rather than executable source
- Source: Plant disease image dataset
- Committed: Yes, which makes repository size and cloning heavier

---

*Structure analysis: 2026-04-16*
*Update when directory structure changes*
