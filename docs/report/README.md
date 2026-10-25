# Report folder

The project report ("Follow the Money Without Moving the Data"), written as one Markdown file per section. Owner of the report: T4 (Shreyas). The full draft is due on 6 Nov 2026 and the submission is on 10 Nov (`docs/schedule_nov10.md`). `00_outline.md` is the plan for the whole report: sections, owners, sources, target lengths and the table of every number and figure the report needs.

## Files

| File | What | Status on 25 Oct |
|---|---|---|
| `00_outline.md` | Section list with owner, source docs and scripts, target length; the numbers table | done |
| `01_introduction.md` | Section 1 | draft 1 |
| `02_related_work.md` | Section 2 | draft 1 |
| `references.md` | Bibliography, [1]-[41] numbered as in the proposal, then [42]-[77] | done, with open checks |
| `03_problem_approach.md` ... `11_compute.md` | Sections 3 to 11 | to be written (names below) |

Names for the remaining files, so that the build command below works unchanged: `03_problem_approach.md`, `04_data_setup.md`, `05_methods.md`, `06_experiments.md`, `07_results.md`, `08_safety.md`, `09_failure_analysis.md`, `10_limitations.md`, `11_compute.md`, and `00_abstract.md` for the abstract.

## Who writes what

| Section | Owner |
|---|---|
| Abstract, 1 Introduction, 2 Related work, 3 Problem and approach, 6 Experiments and baselines, 8 Safety, 10 Limitations, 11 Compute and reproducibility | T4 Shreyas (gate, fusion, decision, eval, safety, Group C ports, report) |
| 4 Data and setup; 5.1 one-bank model, upper limits, intake, retriever; 7.1 detection results | T1 Abhishek (data, detection models, retrieval) |
| 5.2 banks, DSL, privacy, back-testing, harness; 7.2 question and planner results | T2 Bhavya |
| 5.3 planner, fine-tuning, library, fast model, LLM frameworks; 7.3 results; case studies in 9 | T3 Akshat |
| 5.4 gate, fusion, decision, Group C ports; 7.4 headline results, hypothesis table; 9 error analysis | T4 Shreyas |

Each section's sources and target length are in `00_outline.md`. Write in plain, specific prose: say what was done and what was measured, no claims without a number or a source, and no result in text that is not in a table of section 7.

## Placeholder convention

A number or figure that comes from a script is written in the text as `{{TBD:<id>}}`, where `<id>` is lower case letters, digits and underscores. Every id must be a row of the table "Numbers and figures the report needs" in `00_outline.md`, which names the script or experiment that produces it, who runs it, and its status. When the number exists, replace the placeholder in the text with the number (and in the table set the status to "filled"). Never type a result that did not come from a script.

| Prefix | Meaning | Example |
|---|---|---|
| `gap_` | gap closed, with its CI | `{{TBD:gap_closed_gfp}}` |
| `h<n>_` | the number behind hypothesis H`n` | `{{TBD:h9_difference}}` |
| `tab_` | a whole table | `{{TBD:tab_5a}}` |
| `fig_` | a figure | `{{TBD:fig_pareto}}` |
| `n_` | counts (alerts, positives) | `{{TBD:n_alerts}}` |
| other | a named result or section element | `{{TBD:e18_sweep}}` |

To add a placeholder: add its row to the numbers table first, then use it in the text. `tests/test_report_docs.py` fails if a placeholder in any file is not in the table.

Markers for things a person must check use square brackets around the word CHECK, followed by a colon and the question, for example a missing author list in `references.md`. Do not leave a marker in a section handed in for the final draft; resolve it or delete the claim.

## Build one document

Needs `pandoc` (not installed by this repo; do not install it on the shared machines). From `docs/report/`, with all section files present:

```bash
# 1. fail if anything is unresolved
! grep -n '{{TBD' 0*.md 1*.md references.md
! grep -n '\[CHECK' 0*.md 1*.md references.md

# 2. one Markdown file, the outline left out
cat 00_abstract.md 01_*.md 02_*.md 03_*.md 04_*.md 05_*.md 06_*.md 07_*.md \
    08_*.md 09_*.md 10_*.md 11_*.md references.md > report_full.md

# 3. a single document (HTML needs no LaTeX; PDF needs a LaTeX engine such as xelatex)
pandoc report_full.md --from markdown --standalone --toc --toc-depth=2 \
       --metadata title="Follow the Money Without Moving the Data" -o report.html
pandoc report_full.md --from markdown --standalone --toc --pdf-engine=xelatex -o report.pdf
pandoc report_full.md --from markdown --toc -o report.docx
```

Do not use `--number-sections`: the headings are already numbered. Remove the italic "Owner ... Draft" line at the top of each section file before the final build. The generated `report_full.*` files are not committed.

## Tests

```bash
python -m pytest -q tests/test_report_docs.py
```

checks that (a) every `[n]` citation in sections 1 and 2 has an entry in `references.md`, (b) every `{{TBD:<id>}}` placeholder in these files is a row of the numbers table, and (c) every open-check marker in these files has a row in the table below, one row per marker. Keep the table up to date when a marker is added or resolved.

## Open checks

Each row is one marker in the file named in the first column. Resolve it from the source named in the row (or from the paper itself) and delete both the marker and the row. The references are copied from `docs/proposal_v5.pdf` and `plan.md` section 18, which are often missing authors' initials, titles or identifiers; those need the paper in hand.

| File | Where | Open question |
|---|---|---|
| 02_related_work.md | section 2.1 | the same document could not confirm whether ExSTraQt's decision threshold was tuned on test or what the FraudGT split ratios are; verify both in the paper before quoting these numbers |
| 02_related_work.md | section 2.2 | describe what Aurora actually evaluated, from the report itself; the repo only characterises it, in Table 1 of the proposal, among the "federated AML" work in which banks train jointly or compute on pooled data |
| 02_related_work.md | section 2.2 | add references for federated averaging, DP-SGD, Flower and Opacus; none of them is in the proposal or in plan.md section 18 |
| 02_related_work.md | section 2.3 | the repo states this point only through the proposal's one-line summary of [15 |
| 02_related_work.md | section 2.7 | describe what Spotlighting and StruQ do from the papers; the repo names them only as comparisons |
| 02_related_work.md | section 2.8 | verify the summary of 31 CFR 1010.540 against the regulation text before it is quoted in the report |
| references.md | [5] | the proposal lists this entry as "FIA Framework" without authors; the title prefix and the author names are taken from docs/external_code_check.md, which says the arXiv title is "FAA Framework"; confirm the full title and the author initials on arXiv |
| references.md | [9] | docs/t1_week0.md records that the arXiv page is now titled "Federated Layer-Wise Exchange for Cross-Client Money Laundering Pattern Detection"; decide which title to print, and add the initials |
| references.md | [11] | authors missing in the proposal |
| references.md | [15] | authors missing in the proposal |
| references.md | [18] | confirm the notice designation and the year; the proposal gives "Notice FSM-N02", while the red-flag corpus (configs/compliance/red_flags.yaml) uses the Guidelines to MAS Notice 626 (Banks), listed here as [77 |
| references.md | [19] | no report titles, years or URLs in the proposal; list the reports actually used; configs/compliance/red_flags.yaml says the FATF PDFs could not be fetched (HTTP 403), so the v0 corpus does not contain FATF red flags yet |
| references.md | [22] | no authors or URLs in the proposal; docs/external_code_check.md gives the AEGIS code at github.com/Kunsh162007/AEGIS (a June 2026 submission, no licence); the federated-analytics submission has no further details anywhere in the repo |
| references.md | [30] | no authors or URL in the proposal |
| references.md | [31] | no authors or URL in the proposal |
| references.md | [39] | authors missing in the proposal |
| references.md | [42] | initials and arXiv identifier missing in plan.md |
| references.md | [43] | initials and arXiv identifier missing in plan.md |
| references.md | [44] | plan.md gives this title; docs/t1_week0.md gives the arXiv title as "Multigraph Message Passing with Bi-Directional Multi-Edge Aggregations" (Bilgi, Chen, Atasu); confirm which to print, and the year |
| references.md | [45] | authors and year missing in plan.md; docs/external_code_check.md attributes the paper to PayPal |
| references.md | [46] | authors, full title and year missing in plan.md |
| references.md | [47] | authors missing in plan.md; docs/t1_week0.md names Bashir and Mohan |
| references.md | [48] | authors and year missing in plan.md; the repo's documents do not describe this paper, so it is listed but not discussed in the text |
| references.md | [49] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [50] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [51] | initials, full title, venue and arXiv identifier missing in plan.md |
| references.md | [52] | initials and arXiv identifier missing in plan.md |
| references.md | [53] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [54] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [55] | initials and arXiv identifier missing in plan.md |
| references.md | [56] | initials, full title, venue and arXiv identifier missing in plan.md |
| references.md | [57] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [58] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [59] | initials, full title and arXiv identifier missing in plan.md |
| references.md | [60] | initials, venue and arXiv identifier missing in plan.md |
| references.md | [61] | initials and full title missing in plan.md |
| references.md | [62] | initials, full title, venue and arXiv identifier missing in plan.md |
| references.md | [63] | initials, full title, venue and arXiv identifier missing in plan.md |
| references.md | [64] | initials and full title missing in plan.md |
| references.md | [65] | initials and full title missing in plan.md |
| references.md | [66] | initials, full title, venue and arXiv identifier missing in plan.md |
| references.md | [67] | initials and full title missing in plan.md |
| references.md | [68] | initials missing in plan.md |
| references.md | [72] | initials missing in plan.md |
| references.md | [74] | plan.md notes the source paper was still to be traced; docs/t1_week0.md says the page redirects to sota2.com and that the figure comes from [75 |
| references.md | [76] | year of the manual edition and the live URL; the repo cites the archive snapshot |
| references.md | [77] | re-check against the current guidelines |
