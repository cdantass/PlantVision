# External Integrations

**Analysis Date:** 2026-04-16

## APIs & External Services

**External APIs:**
- None detected in source files under `codigo/*.py`
  - Integration method: not applicable
  - Auth: not applicable
  - Rate limits: not applicable

**Email/SMS:**
- None detected

**Payment Processing:**
- None detected

## Data Storage

**Databases:**
- None detected
  - Connection: not applicable
  - Client: not applicable
  - Migrations: not applicable

**File Storage:**
- Local repository storage only
  - Source datasets: `PlantVillage/` and `codigo/dataset/`
  - Model artifact: `codigo/plant_model.h5`
  - Sample input/output images: `codigo/teste.jpg`, `codigo/teste_diagnosis.jpg`

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- None detected

**OAuth Integrations:**
- None detected

## Monitoring & Observability

**Error Tracking:**
- None detected

**Analytics:**
- None detected

**Logs:**
- Console output only via `print(...)` statements in `codigo/plant_analysis.py`

## CI/CD & Deployment

**Hosting:**
- None detected
- The repository appears to be intended for local execution rather than hosted inference or training services

**CI Pipeline:**
- None detected under `.github/`, GitHub Actions, or equivalent tooling

## Environment Configuration

**Development:**
- Required env vars: none detected
- Secrets location: none detected
- Mock/stub services: not applicable because there are no external services

**Staging:**
- No staging environment structure detected

**Production:**
- No production environment structure detected

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Local Data Dependencies

**Training data:**
- `codigo/dataset/train` is expected by `create_datasets()` in `codigo/plant_analysis.py`
- `codigo/dataset/val` is expected by `create_datasets()` in `codigo/plant_analysis.py`

**Inference artifacts:**
- `codigo/plant_model.h5` is loaded in the `__main__` block of `codigo/plant_analysis.py`
- `codigo/teste.jpg` is the default image analyzed by `run_full_analysis(...)`

---

*Integration audit: 2026-04-16*
*Update when adding/removing external services*
