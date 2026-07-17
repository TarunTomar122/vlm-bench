# Redesign Pages Site As A Research Paper

## Intended Commit Subject

`Redesign Pages site as a research paper`

## Problem Or Decision

The first website design used a large promotional hero, gradients, colored cards, dark sections, buttons, shadows, and animation. That presentation looked like a product launch and did not match the requested tone of a simple academic research page.

## Files And Behavior Changed

- Rebuilt `docs/index.html` as a compact paper-style document with a title block, abstract, numbered sections, findings, figures, captions, limitations, and reproducibility links.
- Replaced `docs/styles.css` with a monochrome light theme using a narrow reading column, serif body typography, simple rules, grayscale figures, and responsive mobile spacing.
- Removed the previous hero card, gradients, color bands, CTA styling, dark mode sections, shadows, and animation.
- Verified that `README.md` and all website files contain no Unicode en dash or em dash characters.

## Alternatives Considered

- Recolor the existing marketing layout in grayscale. This was rejected because its hierarchy and components would still feel promotional.
- Mimic a literal two-column PDF paper. This was rejected because narrow columns are less readable on phones and long web pages.
- Hide most experimental detail behind cards or accordions. This was rejected because a research page should expose claims and caveats directly.

## Verification Evidence

- `make PYTHON=/tmp/vlm-paper-venv/bin/python submission`
- A Unicode punctuation scan of `README.md` and `docs/` returned no en dash or em dash matches.
- `git --git-dir=.git-data --work-tree=. diff --check`
- Rendered and visually inspected the page with Chromium at 1440 x 1000 and 390 x 844.

## Known Limitations And Unsupported Claims

- Publication figures are displayed in grayscale by CSS, while the downloadable generated files retain their original colors.
- The site remains a static research summary and does not contain the full manuscript.
- This presentation change does not alter experimental results or claims.

## Next Action Enabled

Publish the updated `docs/` directory through GitHub Pages and review the live rendering after deployment.
