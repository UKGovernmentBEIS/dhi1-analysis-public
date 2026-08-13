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

## About

Analysis code and pseudonymised data supporting the paper above, published by the
**UK AI Security Institute**.

- **What this is** — the data and analysis pipeline behind the two studies described above.
- **What you can do with it** — re-run the reported Bayesian ordinal GLMs end to end and
  regenerate 33 of the 37 published parameter-estimate files. See
  [What does not reproduce from public data](#what-does-not-reproduce-from-public-data)
  for the four exceptions and why they exist.
- **Data protection** — participant data is pseudonymised. All demographic variables, the
  fieldwork panel's respondent identifier, exact age etc. have been
  removed.
- **Licence** — MIT for code; Open Government Licence v3.0 for data, figures and parameter
  estimates. See [Licence](#licence).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Analysis pipeline and notebooks |
| `data/` | Pseudonymised participant data and survey codebooks |
| `parameter_estimates/` | **Reported** GLM parameter estimates and WAIC comparisons |
| `parameter_estimates_compliance_cov/` | Same models retaining the post-treatment compliance covariate (not reported; see below) |
| `figures/` | Figures as published, in PDF |

## Setup

```bash
bash setup.sh                     # creates analysis_env/ and registers a Jupyter kernel
source analysis_env/bin/activate
```

`requirements.txt` lists the pinned package versions for the analysis. MCMC estimates
vary slightly across `jax`/`numpyro` versions, so re-runs may introduce sampling variance (which do not influence the inferences drawn based on the results).

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

The two pipelines write to separate directories by default, so neither can overwrite the
other's results: the canonical script writes to `parameter_estimates/` and `plots/`, and
`data_analysis_pipeline.py` writes to `parameter_estimates_compliance_cov/` and
`plots_compliance_cov/`. Override either with `--output-dir` / `--plots-dir`.

The MCMC runs are GPU-accelerated via JAX; they will run on CPU but slowly.

### Why there are two pipelines

`data_analysis_pipeline.py` includes the subject-level compliance score
(`sub_compliance`) as a GLM covariate. That score is measured *post-treatment*, so
conditioning on it can bias the treatment estimates. It was therefore dropped from the
reported analyses. Both sets of estimates are published so the difference can be
inspected.

## Data

Row-level participant data in this repository is pseudonymised.

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
at a level of detail that makes most respondents unique.

## Citation

If you use the findings, **cite the paper**. If you use the data or code specifically,
please cite **both** the paper and this repository.

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff), so GitHub's
*Cite this repository* button resolves to the paper.

### The paper

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

### This repository

```bibtex
@software{luettgau2026dhi1analysis,
      title={Conversational AI increases political knowledge as effectively as self-directed internet search: analysis code and data},
      author={Lennart Luettgau and Hannah Rose Kirk and Kobi Hackenburg and Jessica Bergs and Henry Davidson and Henry Ogden and Divya Siddarth and Saffron Huang and Christopher Summerfield},
      year={2026},
      publisher={UK AI Security Institute},
      url={https://github.com/lenluettgau/dhi1-analysis-public},
}
```

When reusing the data, figures or parameter estimates, also include the Open Government
Licence attribution statement given under [Licence](#licence).

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
