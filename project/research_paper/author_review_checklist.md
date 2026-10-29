# Author Review Checklist: Pre-Submission Verification Items

This checklist identifies specific items in the generated manuscript that require confirmation, unblinding, or institutional approval by the primary human researchers prior to formal journal or conference submission.

---

### 1. Authorship, Affiliations, and Funding
- [ ] **Author Byline:** Insert confirmed author names, order, email addresses, and ORCID identifiers in place of the generic placeholder `[Author Names Blinded for Peer Review]`.
- [ ] **Institutional Affiliations:** Specify institutional departments, universities, and hospitals where the study was approved and conducted.
- [ ] **Funding & Grants:** Disclose research grants, computational facility support, or fellowship funding in the Acknowledgments section.
- [ ] **Conflict of Interest:** Verify whether any co-author holds commercial ties or patents related to digital health devices, psychiatric pharmaceuticals, or wearable software.

### 2. Dataset Attribution and Privacy Protocol
- [ ] **Cohort Name & Ethical Approval:** Confirm the formal institutional review board (IRB) protocol number, ethics approval committee, and data-use agreement for the 4,948-participant wearable cohort.
- [ ] **Hardware Sensors:** Confirm exact wearable accelerometer specifications (e.g., tri-axial wrist actigraphy, sampling frequency, device make/model) if permitted by data governance.
- [ ] **Open-Access / Repository Release:** Decide whether the processed feature parquet files and model checkpoints will be released in an open Zenodo / Synapse repository upon acceptance.

### 3. Methodological and Experimental Confirmatory Checks
- [ ] **Inferential Hypothesis Testing:** The current manuscript reports descriptive performance differences without claimed statistical significance. If target reviewers demand inferential p-values, run a post-hoc Wilcoxon signed-rank test or McNemar's test on the paired predictions in `stage4_progression_results.csv`.
- [ ] **Sub-Monthly Timestamp Alignment:** Confirm that survey administration occurred at the conclusion of monthly tracking periods, as designated by `phq9_score_end`. The paper transparently documents the limitation of monthly aggregate resolution (Case C).
- [ ] **Observation Window Selection:** Ensure strict adherence to $W=1$ as the prospectively selected configuration. The paper explicitly explains that while $W=\text{Full}$ had a higher test Macro-F1 (0.3309 vs. 0.3204), $W=1$ was selected on validation Macro-F1 (0.3359) to prevent test-set data snooping.

### 4. Regulatory and Intended Use Disclaimer
- [ ] **Software Classification:** Maintain the explicit **"RESEARCH USE ONLY"** status. Verify that no claims of automated diagnosis, autonomous drug prescription, or emergency crisis dispatch are made.
- [ ] **Clinical Triage Threshold:** Confirm the operational utility of $T_4^* = 0.2100$ in a two-tier screening context where false positive alerts trigger non-invasive low-cost follow-ups (e.g., nurse telephone check-ins) rather than invasive medication changes.

### 5. Target Publication Venue Selection
- [ ] **Target Journal Selection:** Confirm the formatting requirements of the selected journal:
  - *IEEE Journal of Biomedical and Health Informatics (J-BHI)* [IEEE numbered style, ~8–10 pages].
  - *npj Digital Medicine* [Nature Springer style, ~5,000 words].
  - *Journal of Medical Internet Research (JMIR Mental Health)* [Vancouver style, ~5,500 words].
  - *ACM Transactions on Computing for Healthcare (HEALTH)* [ACM numbered style].
