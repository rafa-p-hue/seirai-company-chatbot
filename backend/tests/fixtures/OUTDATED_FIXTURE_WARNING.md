# Outdated fixture warning

Do **not** use repository synthetic corpora as the live Hikari City evaluation set.

## Live corpus (session uploads)

Filenames:

- `01_resident_registration_guide.pdf`
- `02_garbage_recycling_guide.html`
- `03_national_health_insurance_faq.docx`
- `04_childcare_support_programs.md`
- `05_city_fees_schedule.csv` — **current** residence certificate fees **¥350 / ¥250** (effective April 1, 2026)
- `06_disaster_preparedness.pptx` — Sakuragi Community Center pet-friendly area
- `07_city_fees_2024_archived.csv` — archived fees **¥300 / ¥200** (superseded)

## Conflicting test fixtures

Several backend tests still embed synthetic docs with **current** fees of ¥300/¥200
(or markdown substitutes for pets/sofa). Those values are **archived-only** in the
live corpus and must not be reported as current-fee retrieval passes.

Examples:

- `tests/test_service_domain_rag.py` (PDF CORPUS certificates)
- `tests/test_followup_entity_procedure_eval.py` (markdown CORPUS)
- `tests/test_seven_file_recall_regression.py` (markdown CORPUS)
- `tests/test_compound_procedure_continuity.py`

Treat those as **legacy synthetic fixtures**, separate from live session diagnosis.
