# Submission Checklist

## Scientific Audit

- [x] Every task-versus-generic comparison uses the same K.
- [x] Processed-v2 is called method-selection evidence, not sealed external evidence.
- [x] SmolVLM appears only at K4 in completed-result plots.
- [x] The IIIT5K negative transfer result is in the abstract or main text.
- [x] Confidence intervals that cross/touch zero are not described as confirmed wins.
- [x] Parameter, latency, and checkpoint-size claims are distinguished.
- [x] Qwen final-route latency is described as missing rather than inferred.
- [ ] One independent ML researcher has checked aggregation, splits, claims, and novelty.

## Manuscript Package

- [ ] Author names, affiliations, email, ORCID, acknowledgements, and funding are supplied or explicitly confirmed
  as not applicable. The repository does not invent missing author metadata.
- [x] Title and abstract match the final PDF and `paper/arxiv-metadata.md`.
- [x] All figure text is readable at one-column or two-column width.
- [x] Tables use `booktabs`; captions and surrounding text identify the relevant split, metric, and uncertainty.
- [x] Appendix includes routes, seeds, hashes, and search configuration.
- [x] Repository URL and candidate tag are included in the manuscript.
- [x] The `arxiv-v1-candidate` tag is created after the manuscript assets are frozen.
- [ ] A software license and data-license notes are selected by the author.

## Human And AI Review

- [ ] The human author has read every sentence and can explain every equation, design choice, result,
  limitation, and conclusion without relying on an AI transcript.
- [ ] Every cited paper has been opened at its primary source, and its title, authors, venue, year,
  identifier, and claimed relevance have been checked manually.
- [ ] Every number in the abstract, figures, tables, and conclusion has been traced to committed
  evidence or a named analysis artifact.
- [ ] The target conference or journal's current generative-AI policy has been checked separately;
  arXiv is a repository, not a substitute for the venue's authorship and disclosure rules.
- [x] No AI system is listed as an author.
- [x] Substantive AI assistance is disclosed in the acknowledgements. The target venue's separate policy must
  still be checked before venue submission.

> AI-assisted tools were used for code assistance, experiment orchestration, figure generation,
> manuscript organization, and language editing. No AI system is an author; responsibility for the
> submitted work remains with the human author.

- [ ] The disclosure above is included only after its final sentence is true. Human review cannot be
  replaced by adding a disclosure.
- [ ] No confidential credentials, restricted images, private review material, or personal data were
  included in prompts or submission files.

## arXiv

- [x] Sign in to a registered arXiv author account and resolve the required `cs.CV` endorsement for v1.
- [x] Freeze author metadata, title, abstract, comments, and categories for announced v1.
- [x] Use `cs.CV` as the v1 primary category. No cross-list was added.
- [x] Choose the v1 distribution license deliberately. The choice is irrevocable.
- [x] Prepare a source ZIP containing `main.tex`, `references.bib` or matching `main.bbl`, all eight
  PDF figures, and `tables/generated-main-results.tex` with their relative directories intact.
- [x] Run `make arxiv-package` if the repository version is final, or use Overleaf's arXiv source
  export if Overleaf contains newer edits. Do not upload an archive without knowing which version won.
- [x] Remove compiled PDFs, auxiliary files, caches, model weights, raw restricted images,
  credentials, and unrelated repository files from the upload ZIP.
- [x] Upload the v1 ZIP, select **pdfLaTeX**, and confirm `main.tex` as the top-level file.
- [x] Retain every referenced figure, table, and bibliography file during the arXiv file review.
- [x] Complete arXiv's generated-PDF preview, metadata, license, agreement, and submission flow for v1.
- [ ] If an error is found before announcement, use **Unsubmit**, correct the existing submission, and
  resubmit. Do not create a second paper entry.
- [x] After announcement, add [arXiv:2607.17052](https://arxiv.org/abs/2607.17052) to the website,
  README, `CITATION.cff`, and submission record. The `arxiv-v1-candidate` release tag preserves the
  submitted source candidate.

Official references:

- Submission process: https://info.arxiv.org/help/submit/index.html
- TeX and bibliography requirements: https://info.arxiv.org/help/submit_tex.html
- Endorsement: https://info.arxiv.org/help/endorsement.html
- Irrevocable license choice: https://info.arxiv.org/help/license/index.html

As checked on 2026-07-18, the official arXiv author guide does not state a separate generative-AI
disclosure field. That does not remove the human author's responsibility or any stricter policy from a
conference, journal, employer, or funder.

## Website/Wix

- [x] Run `make submission` and inspect `docs/index.html` locally.
- [x] Replace the paper placeholder with [arXiv:2607.17052](https://arxiv.org/abs/2607.17052).
- [ ] Add author/affiliation information after approval.
- [ ] Publish `docs/` with GitHub Pages, or recreate its sections in Wix using the supplied copy.
- [ ] Verify mobile layout, figure alt text, repository link, and caveat text.
