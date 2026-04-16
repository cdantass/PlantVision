# Testing Patterns

**Analysis Date:** 2026-04-16

## Test Framework

**Runner:**
- No automated test runner detected
- No config files for `pytest`, `unittest`, `nose`, or other Python test frameworks

**Assertion Library:**
- None detected

**Run Commands:**
```bash
python plant_analysis.py    # Manual script execution from `codigo/`
```

## Test File Organization

**Location:**
- No `tests/` directory detected
- No `test_*.py` or `*_test.py` files detected in the repository

**Naming:**
- Not established

**Structure:**
```text
No automated test tree exists yet.
```

## Test Structure

**Observed verification approach:**
- Manual execution of `codigo/plant_analysis.py`
- Manual inspection of printed diagnosis output
- Manual inspection of generated GradCAM overlay `codigo/teste_diagnosis.jpg`

**Patterns:**
- Demo-oriented validation rather than repeatable assertions
- Inference path has a single baked-in sample image workflow
- Training path appears to be invoked manually by calling `train_model()` rather than through a dedicated command

## Mocking

**Framework:**
- None detected

**Patterns:**
- No mocking utilities or fixtures detected

**What would need mocking first:**
- Filesystem reads for image loading
- TensorFlow model loading and prediction for fast unit tests
- OpenCV image I/O for deterministic visualization tests

## Fixtures and Factories

**Test Data:**
- Existing sample image: `codigo/teste.jpg`
- Generated output artifact: `codigo/teste_diagnosis.jpg`
- Dataset folders under `codigo/dataset/` act as training/validation fixtures, but they are production-sized assets rather than test-scoped fixtures

**Location:**
- No dedicated fixture directory exists

## Coverage

**Requirements:**
- No coverage target detected
- No enforcement in git or CI detected

**Configuration:**
- None detected

## Test Types

**Unit Tests:**
- Not present

**Integration Tests:**
- Not present

**E2E / Workflow Tests:**
- Manual end-to-end style validation only: load model, analyze image, inspect output

## Common Gaps

**Inference behavior:**
- No regression checks for class ordering, confidence threshold behavior, or top-3 output formatting

**Training behavior:**
- No smoke test that datasets load correctly from `codigo/dataset/train` and `codigo/dataset/val`

**Explainability behavior:**
- No automated check that `compute_gradcam(...)` still works if the model architecture changes

## Recommended First Test Layer

- Add `pytest` at the repo root or `codigo/tests/`
- Start with smoke tests for `diagnose_plant(...)`, preprocessing helpers, and path resolution behavior
- Add one deterministic GradCAM test using a mocked model or frozen fixture output

---

*Testing analysis: 2026-04-16*
*Update when test patterns change*
