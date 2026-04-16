# Codebase Concerns

**Analysis Date:** 2026-04-16

## Tech Debt

**Path handling in `codigo/plant_analysis.py` and `codigo/plant_gradcam.py`:**
- Issue: Dataset, model, and image paths are hardcoded as relative strings such as `"dataset/train"`, `"plant_model.h5"`, and `"teste.jpg"`
- Why: The code is optimized for quick local execution from one expected working directory
- Impact: Running from the repo root with `python codigo/plant_analysis.py` will likely fail to find required files
- Fix approach: Resolve all paths relative to `Path(__file__).parent` or a central config object

**Duplicated dataset storage in `PlantVillage/` and `codigo/dataset/`:**
- Issue: The repo carries at least two dataset trees with overlapping plant-disease classes
- Why: One copy appears to be a raw source dataset while `codigo/dataset/` is the working training/validation split
- Impact: Repository size is inflated, data updates are harder to manage, and provenance becomes unclear
- Fix approach: Keep one source-of-truth dataset location and document how train/val splits are produced

**Large binary artifacts committed to source control:**
- Issue: `codigo/plant_model.h5`, image datasets, and generated files are stored directly in git
- Why: Simple local workflow with no external artifact store
- Impact: Clones are heavy, diffs are opaque, and model/version management is brittle
- Fix approach: Move large assets to release storage, Git LFS, or a documented data/artifact pipeline

## Known Bugs

**Script execution depends on current working directory:**
- Symptoms: File-not-found failures for model, dataset, or sample image when launched from an unexpected directory
- Trigger: Running from `G:\analiseplant` instead of `G:\analiseplant\codigo`
- Workaround: Run from inside `codigo/`
- Root cause: Relative paths are not anchored to the script location

**Potential class mismatch between training output and inference constant:**
- Symptoms: Inference can mislabel outputs if `CLASS_NAMES` diverges from dataset folder ordering
- Trigger: Changing dataset classes or training on a different class set without updating the hardcoded list in `codigo/plant_analysis.py`
- Workaround: Keep hardcoded list manually synchronized
- Root cause: `train_model()` derives classes from the dataset, but the `__main__` path redefines `CLASS_NAMES` manually

## Security Considerations

**Untrusted model and image files:**
- Risk: Loading arbitrary `.h5` or image files can crash the process or expose the runtime to library-level parsing issues
- Current mitigation: None beyond basic OpenCV load checks
- Recommendations: Validate accepted file locations, isolate model loading, and avoid loading untrusted artifacts directly

**No repository hygiene for generated artifacts:**
- Risk: `codigo/__pycache__/` and output images can be committed accidentally, increasing noise and leaking local execution state
- Current mitigation: None, because no `.gitignore` exists
- Recommendations: Add `.gitignore` for `__pycache__/`, generated images, and transient local artifacts

## Performance Bottlenecks

**Training cost on local hardware:**
- Problem: TensorFlow training with MobileNetV2 fine-tuning can be slow or memory-heavy on CPU-only machines
- Measurement: No benchmark committed
- Cause: Transfer learning plus image augmentation on local datasets
- Improvement path: Document GPU expectations, add smaller smoke datasets, and expose tunable training parameters

**Repository size and I/O load:**
- Problem: Large checked-in datasets and model binaries make clone, checkout, and backup operations heavier than the source code requires
- Measurement: Not quantified in-repo, but visible from the dataset-heavy file tree and 28MB `codigo/plant_model.h5`
- Cause: Data and artifacts live beside source without filtering
- Improvement path: Separate code from datasets/artifacts and introduce download/build steps

## Fragile Areas

**GradCAM layer selection in `codigo/plant_gradcam.py`:**
- Why fragile: `compute_gradcam(...)` assumes the last convolution layer is named `Conv_1`
- Common failures: Changing model architecture or layer naming will break explainability generation
- Safe modification: Discover the last convolution layer programmatically or make it configurable
- Test coverage: None

**Main script coupling in `codigo/plant_analysis.py`:**
- Why fragile: One module owns preprocessing, dataset creation, training, testing helper logic, and the demo entry point
- Common failures: Small changes to one workflow can affect another because constants and imports are shared
- Safe modification: Split CLI/demo entry from reusable pipeline code before larger feature work
- Test coverage: None

## Scaling Limits

**Local-only execution model:**
- Current capacity: Single-user, local workstation workflow
- Limit: No batching service, no remote inference API, no job orchestration
- Symptoms at limit: Difficult to share, automate, or run concurrently across environments
- Scaling path: Introduce a service boundary or a reproducible CLI with explicit config and artifact handling

## Dependencies at Risk

**TensorFlow / Keras + `.h5` serialization path:**
- Risk: Serialization compatibility can drift across TensorFlow versions
- Impact: Model loading in `codigo/plant_analysis.py` may break after dependency upgrades
- Migration plan: Pin versions more tightly or migrate to a more current Keras save format with documented compatibility

## Missing Critical Features

**No reproducible training/inference CLI separation:**
- Problem: Training helpers exist, but the main executable path only loads a preexisting model and runs one baked-in sample
- Current workaround: Edit source or call functions manually
- Blocks: Easy retraining, batch inference, and consistent automation
- Implementation complexity: Medium

## Test Coverage Gaps

**All executable logic is effectively untested:**
- What's not tested: preprocessing, dataset loading, diagnosis output, GradCAM generation, and path handling
- Risk: Refactors can silently break the only workflows the repo depends on
- Priority: High
- Difficulty to test: Moderate, because TensorFlow and image assets should be isolated behind smaller fixtures or mocks

---

*Concerns audit: 2026-04-16*
*Update as issues are fixed or new ones discovered*
