# Coding Conventions

**Analysis Date:** 2026-04-16

## Naming Patterns

**Files:**
- `snake_case.py` for Python modules: `plant_analysis.py`, `plant_model.py`, `plant_gradcam.py`
- Generated image output appends `_diagnosis` to the input stem, e.g. `teste_diagnosis.jpg`
- No formal test-file pattern exists because no tests are present

**Functions:**
- `snake_case` for all functions: `build_plant_classifier`, `enable_fine_tuning`, `run_full_analysis`
- Verb-first names for workflow functions: `load_and_preprocess`, `create_datasets`, `test_single_image`
- No class-based API surface detected

**Variables:**
- `snake_case` for local variables and parameters such as `img_tensor`, `base_model`, and `confidence_threshold`
- `UPPER_SNAKE_CASE` for module constants such as `IMG_SIZE`, `BATCH_SIZE`, and `CLASS_NAMES`
- Dictionary keys use lowercase strings such as `"diagnosis"` and `"confidence"`

**Types:**
- Light type hints on some function signatures, but not consistently across all modules
- Return types are plain `dict` or implicit tuples rather than custom typed objects

## Code Style

**Formatting:**
- No formatter config detected
- Source uses 4-space indentation and conventional Python line wrapping
- Strings are predominantly single-quoted in `codigo/plant_model.py` and mixed elsewhere
- Inline section-banner comments are used heavily for visual separation

**Linting:**
- No `ruff`, `flake8`, `pylint`, or equivalent config detected
- No automated style enforcement scripts detected

## Import Organization

**Order:**
1. Standard library imports such as `from pathlib import Path`
2. Third-party imports such as `numpy`, `cv2`, `tensorflow`
3. Local module imports such as `from plant_model import ...`

**Grouping:**
- Imports are grouped with blank lines in some files, but not strictly normalized
- Relative imports are not used; local modules are imported by filename from the working directory

**Path Aliases:**
- None detected

## Error Handling

**Patterns:**
- Raise built-in exceptions close to the failure site for invalid or missing input images
- Let exceptions propagate to the top-level script without a dedicated boundary handler
- No custom exception classes detected

**Error Types:**
- `FileNotFoundError` for missing images in `codigo/plant_analysis.py`
- `ValueError` for unreadable image content in `codigo/plant_analysis.py` and `codigo/plant_gradcam.py`
- Most training and model-loading operations rely on TensorFlow/OpenCV native exceptions

## Logging

**Framework:**
- Plain `print(...)` statements only

**Patterns:**
- Use console output for class discovery and final diagnosis summaries
- No structured logging, levels, or persistent log sink

## Comments

**When to Comment:**
- File headers and block separators explain broad purpose rather than line-level mechanics
- Comments are largely in Portuguese and document processing stages
- Comments are used to separate conceptual phases in long scripts

**Docstrings:**
- Present on some functions such as `build_plant_classifier(...)`, `diagnose_plant(...)`, and `run_full_analysis(...)`
- Not consistently applied across every function

**TODO Comments:**
- No TODO/FIXME comments detected in the current Python files

## Function Design

**Size:**
- Functions are moderate in size and tend to encapsulate one processing stage
- Modules stay small and focused, with fewer than 160 lines each

**Parameters:**
- Functions use direct positional parameters rather than configuration objects
- Shared defaults are pushed into module constants where practical

**Return Values:**
- Tuple returns for preprocessing helpers, e.g. original image plus tensor
- Dictionary returns for diagnosis payloads passed between modules
- Early raises used for invalid filesystem or image-loading conditions

## Module Design

**Exports:**
- Modules expose top-level functions only
- Consumers import the exact helpers they need rather than wildcard imports

**Separation of concerns:**
- `codigo/plant_model.py` keeps model concerns separate from visualization
- `codigo/plant_gradcam.py` depends on `diagnose_plant(...)` rather than duplicating prediction logic
- `codigo/plant_analysis.py` acts as the orchestration layer and currently owns the most coupling

---

*Convention analysis: 2026-04-16*
*Update when patterns change*
