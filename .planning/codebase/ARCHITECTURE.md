# Architecture

**Analysis Date:** 2026-04-16

## Pattern Overview

**Overall:** Monolithic local ML workflow with helper modules for model definition and GradCAM visualization

**Key Characteristics:**
- Script-first design centered on `codigo/plant_analysis.py`
- All state is file-based: datasets, trained model, input image, and generated output image
- No service boundary, no web API, and no persistence layer beyond local files

## Layers

**Orchestration Layer:**
- Purpose: Owns the top-level workflow for dataset loading, training, model loading, single-image diagnosis, and demo execution
- Contains: `load_and_preprocess`, `create_datasets`, `train_model`, `test_single_image`, and the `__main__` block in `codigo/plant_analysis.py`
- Depends on: TensorFlow/Keras, OpenCV, dataset directories, model helpers, and GradCAM helpers
- Used by: Manual script execution

**Model Layer:**
- Purpose: Defines the classification model and returns structured diagnosis output
- Contains: `build_plant_classifier`, `enable_fine_tuning`, and `diagnose_plant` in `codigo/plant_model.py`
- Depends on: TensorFlow/Keras and NumPy
- Used by: `codigo/plant_analysis.py` and `codigo/plant_gradcam.py`

**Explanation Layer:**
- Purpose: Generates GradCAM heatmaps and writes the combined diagnosis visualization image
- Contains: `compute_gradcam`, `overlay_heatmap`, and `run_full_analysis` in `codigo/plant_gradcam.py`
- Depends on: TensorFlow, OpenCV, NumPy, and `diagnose_plant` from `codigo/plant_model.py`
- Used by: `codigo/plant_analysis.py`

**Data Asset Layer:**
- Purpose: Supplies training, validation, sample input, and model artifact files
- Contains: `codigo/dataset/`, `PlantVillage/`, `codigo/teste.jpg`, and `codigo/plant_model.h5`
- Depends on: Filesystem layout and class folder naming
- Used by: Both training and local inference flows

## Data Flow

**Training Flow:**

1. User runs `codigo/plant_analysis.py` from a directory where relative paths resolve to `dataset/train` and `dataset/val`
2. `create_datasets()` constructs augmented and validation generators via `ImageDataGenerator.flow_from_directory(...)`
3. `build_plant_classifier(...)` creates a MobileNetV2-based classifier with a custom dense head
4. `train_model()` fits the frozen base model, enables fine-tuning, fits again, and saves `plant_model.h5`
5. Saved model becomes the local artifact reused for inference

**Inference + Explainability Flow:**

1. `__main__` loads `plant_model.h5` via `tf.keras.models.load_model(...)`
2. `run_full_analysis(...)` loads and preprocesses `teste.jpg`
3. `diagnose_plant(...)` returns predicted class, confidence, reliability flag, and top-3 probabilities
4. `compute_gradcam(...)` uses the predicted class and layer `Conv_1` to create a heatmap
5. `overlay_heatmap(...)` blends the heatmap with the original image and writes `<stem>_diagnosis.jpg`

**State Management:**
- Entirely file-based and process-local
- No persistent runtime state outside artifacts on disk

## Key Abstractions

**Classifier Builder:**
- Purpose: Encapsulates transfer learning model construction
- Examples: `build_plant_classifier(...)`, `enable_fine_tuning(...)`
- Pattern: Functional helper module

**Diagnosis Result:**
- Purpose: Standardizes inference output for reuse by CLI/demo and GradCAM
- Examples: `{"diagnosis", "confidence", "is_reliable", "top3"}`
- Pattern: Plain Python dictionary rather than typed object

**Full Analysis Pipeline:**
- Purpose: Bundles image preprocessing, inference, GradCAM, and output export into one call
- Examples: `run_full_analysis(...)` in `codigo/plant_gradcam.py`
- Pattern: Procedural pipeline function

## Entry Points

**Demo / Inference Entry:**
- Location: `codigo/plant_analysis.py`
- Triggers: Manual `python plant_analysis.py`
- Responsibilities: Load trained model, define class names, run analysis on `teste.jpg`, print result

**Training Entry:**
- Location: `codigo/plant_analysis.py`
- Triggers: Manual invocation of `train_model()`, currently not called from the `__main__` block
- Responsibilities: Build datasets, train classifier, fine-tune, save `plant_model.h5`

## Error Handling

**Strategy:** Fail fast with uncaught exceptions for missing files or unreadable images

**Patterns:**
- `load_and_preprocess(...)` in `codigo/plant_analysis.py` raises `FileNotFoundError` and `ValueError`
- `load_and_preprocess(...)` in `codigo/plant_gradcam.py` raises `ValueError` on unreadable input
- No top-level recovery or user-friendly CLI error formatting is present

## Cross-Cutting Concerns

**Logging:**
- `print(...)` statements in `codigo/plant_analysis.py` for dataset classes and diagnosis summary

**Validation:**
- Minimal input validation limited to image existence/loadability
- No validation that model class ordering matches hardcoded `CLASS_NAMES` during inference

**File Path Resolution:**
- Relative paths such as `"dataset/train"`, `"plant_model.h5"`, and `"teste.jpg"` are used throughout
- Correct execution depends on current working directory matching the `codigo/` folder layout

---

*Architecture analysis: 2026-04-16*
*Update when major patterns change*
