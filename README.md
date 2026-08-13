# Conversational AI increases political knowledge as effectively as self-directed internet search

Analysis code and data for:

> Lennart Luettgau, Hannah Rose Kirk, Kobi Hackenburg, Jessica Bergs, Henry Davidson,
> Henry Ogden, Divya Siddarth, Saffron Huang, and Christopher Summerfield.
> *Conversational AI increases political knowledge as effectively as self-directed internet
> search.* arXiv:2509.05219.
> [arxiv.org/abs/2509.05219](https://arxiv.org/abs/2509.05219)

The paper reports two studies conducted on UK samples around the 2024 general election:

- A **nationally representative survey** (N = 2,499) measuring how widely conversational AI
  was used to find politically relevant information.
- **Randomised controlled trials** (N = 2,858) comparing task-directed conversation with an
  AI chatbot against self-directed Google search, with outcomes measured pre- and
  post-interaction across four domains: misinformation susceptibility, trust, privately held
  beliefs, and attitude extremity. Effects were assessed across topics, viewpoints, prompting
  strategies, and model families.

This repository contains the data and code behind those analyses. See
[Reproducing the results](#reproducing-the-results) for what does and does not regenerate
from the published data.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Analysis pipeline, notebooks, and the de-identification script |
| `data/` | Pseudonimised participant data and survey codebooks |
| `parameter_estimates/` | **Reported** GLM parameter estimates and WAIC comparisons |
| `parameter_estimates_compliance_cov/` | Same models retaining the post-treatment compliance covariate (not reported; see below) |
| `figures/` | Figures as published, in PDF |

## Setup

```bash
bash setup.sh                     # creates analysis_env/ and registers a Jupyter kernel
source analysis_env/bin/activate
```

`requirements.txt` records the constraints the analysis was developed against, not a
lockfile — see the note at the top of that file. The MCMC runs are GPU-accelerated via
JAX; they will run on CPU but slowly.

## Reproducing the results

```bash
cd code
bash run_pipeline.sh              # all four outcomes x both prompting conditions
```

or a single combination:

```bash
python data_analysis_pipeline_no_compliance.py --dataset trust --prompting 0
```

`data_analysis_pipeline_no_compliance.py` is the **canonical** pipeline. Random seeds are
fixed (`jax.random.PRNGKey(1)`), so runs are reproducible given the same package versions.

> **Both** pipeline scripts write to `parameter_estimates/`. Running
> `data_analysis_pipeline.py` will overwrite the reported results in place. Back that
> directory up first if you want to keep both sets. The `_compliance_cov` outputs were
> separated by hand after running that variant.

### Why there are two pipelines

`data_analysis_pipeline.py` includes the subject-level compliance score
(`sub_compliance`) as a GLM covariate. That score is measured *post-treatment*, so
conditioning on it can bias the treatment estimates. It was therefore dropped from the
reported analyses. Both sets of estimates are published so the difference can be
inspected.

## Data

Row-level participant data in this repository has been de-identified. The following were
removed in full:

- **All demographic variables** — from `survey_data/analysis.csv`: age, gender, region,
  housing tenure, education, income, ethnicity, children, social grade, work status; and
  from `Ztable_indiv_combined.csv` and the `*_exclusions.csv` files: age, gender,
  ethnicity, region, income, religion, education, vote, Brexit vote, disability, mental
  health, chatbot use.
- **The fieldwork panel's respondent identifier**, replaced with a randomly assigned
  pseudonym. Row order was shuffled before assignment, so the original ordering is not
  recoverable.
- **The survey weight** (`W8`) and the four free-text "other" response columns.

Respondent-level data cannot be re-linked to the original panel records from anything in
this repository. The transform is documented in full in the project's de-identification
record, which is held with the restricted source data rather than published here.

| File | Contents |
|---|---|
| `data/survey_data/analysis.csv` | 2,499 respondents x 83 attitudinal items, pseudonymous IDs |
| `data/survey_data/survey_codebook.csv` | Variable → question text (tab-separated) |
| `data/survey_data/options_codebook.csv` | Response value → label (tab-separated) |
| `data/Ztable_{misinfo,trust,private,extreme}_combined.csv` | Pre/post outcome measures by subject, topic and question |
| `data/Ztable_dur_combined.csv` | Task durations |
| `data/Ztable_indiv_combined.csv` | Subject-level compliance and search condition |
| `data/*_exclusions.csv` | Inclusion flags by arm, per model |
| `data/*_model_reliable.csv` | Manipulation-check items |
| `data/demographics_aggregated.csv` | Sample composition (marginal frequencies only) |
| `data/statement_sourcing.csv` | Provenance of the political statements used as stimuli |

Row order in `Ztable_indiv_combined.csv` is significant: subject IDs are derived
positionally (`groupby('model').cumcount() + 1`) and joined against the outcome tables.
Do not re-sort it.

### What does not reproduce from public data

Removing demographics makes six artefacts unreproducible. Their outputs are committed
here, but cannot be regenerated from the data in this repository:

| Artefact | Requires |
|---|---|
| `demographics_analysis.ipynb` → `data/demographics_aggregated.csv` | all 12 demographics |
| Attrition GLM cells in `attrition_duration_analysis.ipynb` | 9 demographics |
| `parameter_estimates/summary_progressive_parsimonious.csv` | `vote` |
| `parameter_estimates/summary_progressive_with_vote.csv` | `vote` |
| `parameter_estimates/summary_progressive_without_vote.csv` | `vote` |
| `parameter_estimates/waic_comparison_progressive_3models.csv` | `vote` |

Everything else reproduces: the remaining **33 of 37** files in `parameter_estimates/`
come from `data_analysis_pipeline_no_compliance.py`, which uses no demographic variables
at all, so the headline results are unaffected. The duration analysis and the exclusion
count tables also run as published.

The two affected notebooks carry a header cell stating which of their cells will not run.

### Controlled access

The underlying data cannot be published because it contains UK GDPR Article 9
special-category data (ethnicity, religion, disability, mental health, political opinion)
at a level of detail that makes most respondents unique. Researchers wanting access for
replication should contact the corresponding author of the paper.

## Citation

Please cite the paper rather than this repository. A machine-readable version is in
[`CITATION.cff`](CITATION.cff).

```bibtex
@misc{luettgau2026conversationalaiincreasespolitical,
      title={Conversational AI increases political knowledge as effectively as self-directed internet search},
      author={Lennart Luettgau and Hannah Rose Kirk and Kobi Hackenburg and Jessica Bergs and Henry Davidson and Henry Ogden and Divya Siddarth and Saffron Huang and Christopher Summerfield},
      year={2026},
      eprint={2509.05219},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2509.05219},
}
```

## Licence

This repository is dual-licensed.

| Contents | Licence | File |
|---|---|---|
| `code/` | MIT | [`LICENSE`](LICENSE) |
| `data/`, `figures/`, `parameter_estimates/`, `parameter_estimates_compliance_cov/` | Open Government Licence v3.0 | [`LICENSE-DATA`](LICENSE-DATA) |

Copyright © Crown copyright 2026, UK AI Security Institute.

When reusing the data, figures, or parameter estimates, include the attribution
statement:

> Contains public sector information licensed under the Open Government Licence v3.0.

Please do not attempt to re-identify individual respondents.
