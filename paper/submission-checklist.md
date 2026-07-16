# Submission Checklist

## Scientific Audit

- [ ] Every task-versus-generic comparison uses the same K.
- [ ] Processed-v2 is called method-selection evidence, not sealed external evidence.
- [ ] SmolVLM appears only at K4 in completed-result plots.
- [ ] The IIIT5K negative transfer result is in the abstract or main text.
- [ ] Confidence intervals that cross/touch zero are not described as confirmed wins.
- [ ] Parameter, latency, and checkpoint-size claims are distinguished.
- [ ] Qwen final-route latency is described as missing rather than inferred.
- [ ] One independent ML researcher has checked aggregation, splits, claims, and novelty.

## Manuscript Package

- [ ] Author names, affiliations, email, ORCID, acknowledgements, and funding are supplied.
- [ ] Title and abstract match the final PDF and arXiv metadata.
- [ ] All figure text is readable at one-column or two-column width.
- [ ] Tables use `booktabs`; captions state examples, split, metric, and uncertainty.
- [ ] Appendix includes routes, seeds, hashes, and search configuration.
- [ ] Repository URL and commit/tag are included in the manuscript.
- [ ] A release tag is created after the manuscript assets are frozen.
- [ ] A software license and data-license notes are selected by the author.

## arXiv

- [ ] Confirm account access and any cs.CV endorsement requirement early.
- [ ] Select `cs.CV` primary; consider `cs.LG` or `cs.CL` cross-list only if justified.
- [ ] Choose arXiv distribution license deliberately.
- [ ] Upload LaTeX source, `.bbl` or bibliography inputs, and all figures; no absolute paths.
- [ ] Compile using arXiv's TeX environment and inspect every page of the generated PDF.
- [ ] Check references, hyperlinks, metadata, page count, and anonymous-review language.
- [ ] Keep the submitted source bundle free of model weights, raw restricted images, caches, and
  credentials.

Official process: https://info.arxiv.org/help/submit/index.html

## Website/Wix

- [ ] Run `make submission` and inspect `site/index.html` locally.
- [ ] Replace the paper placeholder with the final arXiv URL.
- [ ] Add author/affiliation information after approval.
- [ ] Upload `site/assets/` alongside the page, or recreate sections in Wix using the supplied copy.
- [ ] Verify mobile layout, figure alt text, repository link, and caveat text.
