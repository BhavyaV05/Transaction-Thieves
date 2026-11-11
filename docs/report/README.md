# docs/report/

- **The final report** is the LaTeX source and PDF in [`final/`](final/): read [`final/report.pdf`](final/report.pdf).
- [`00_outline.md`](00_outline.md) is the report plan from 25 Oct 2026. It is kept for one reason:
  `scripts/final_analysis.py` reads the placeholder ids (`{{TBD:<id>}}`) of its table "Numbers and figures the
  report needs" (`ftm.eval.final_tables.outline_ids`), and `--fill` fills them in copies of any Markdown files in a folder.
  Do not rename the ids. `tests/test_report_docs.py` checks the table.

The Markdown chapter drafts (`00_abstract.md`, `01_introduction.md` to `11_compute_reproducibility.md`, `references.md`)
were replaced by the LaTeX report and removed. They are in the git history.
