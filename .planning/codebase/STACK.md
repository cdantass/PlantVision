# Technology Stack

**Analysis Date:** 2026-04-16

## Languages

**Primary:**
- Python 3.12 inferred from `codigo/__pycache__/plant_analysis.cpython-312.pyc` and companion bytecode files
- Markdown in `.planning/codebase/*.md` for generated project documentation

**Secondary:**
- No secondary programming language detected in the repository
- JPG image assets and TensorFlow `.h5` model files are checked in as runtime data, not source code

## Runtime

**Environment:**
- CPython 3.12 is the most likely local runtime based on compiled cache filenames
- TensorFlow/Keras runtime required for both training and inference in `codigo/plant_analysis.py` and `codigo/plant_model.py`
- OpenCV native bindings required for image loading, resizing, and GradCAM overlay generation in `codigo/plant_analysis.py` and `codigo/plant_gradcam.py`

**Package Manager:**
- `pip` or equivalent Python package installer is implied by `requirements.txt`
- Lockfile: none present

## Frameworks

**Core:**
- TensorFlow / Keras (`tensorflow>=2.12`) for transfer learning, training, model loading, and inference
- MobileNetV2 from `tensorflow.keras.applications` as the pretrained image backbone in `codigo/plant_model.py`
- OpenCV (`opencv-python>=4.8`) for image I/O and heatmap composition

**Testing:**
- No automated test framework detected

**Build/Dev:**
- No build system, task runner, formatter, or lint configuration detected
- Execution appears to be direct script invocation from the `codigo/` directory

## Key Dependencies

**Critical:**
- `tensorflow>=2.12` - model definition, training, inference, and GradCAM tensor operations
- `opencv-python>=4.8` - image ingestion, resizing, color conversion, and overlay export
- `numpy>=1.24` - tensor preparation, probability sorting, and array manipulation
- `pillow>=9.0` - installed dependency for image handling, though not directly imported in the current Python files
- `matplotlib>=3.7` - installed dependency, likely intended for visualization support, but not directly imported in the current Python files
- `scipy>=1.10` - installed dependency, not directly imported in the current Python files

**Infrastructure:**
- Local filesystem paths such as `codigo/dataset/train`, `codigo/dataset/val`, `codigo/plant_model.h5`, and `codigo/teste.jpg`
- Keras `ImageDataGenerator` for directory-based dataset loading in `codigo/plant_analysis.py`

## Configuration

**Environment:**
- No `.env`, `.env.example`, or environment-variable based configuration detected
- Key runtime values are hardcoded as module constants in `codigo/plant_analysis.py`, including `IMG_SIZE`, `BATCH_SIZE`, epoch counts, and confidence threshold

**Build:**
- No project config files such as `pyproject.toml`, `setup.py`, `tox.ini`, or CI configs detected
- Dependency declaration is limited to the root `requirements.txt`

## Platform Requirements

**Development:**
- Local machine with Python, TensorFlow-compatible native libraries, and filesystem access to the image datasets
- Enough disk space for duplicated dataset trees under `PlantVillage/` and `codigo/dataset/`

**Production:**
- No production deployment target detected
- Current shape is a local research or demo workflow executed directly from the repository

---

*Stack analysis: 2026-04-16*
*Update after major dependency changes*
