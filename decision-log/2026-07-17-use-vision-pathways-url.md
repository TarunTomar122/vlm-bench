# Use Canonical `vision-pathways` URL

## Intended Commit Subject

`Use canonical vision-pathways links`

## Problem Or Decision

GitHub reported during the Pages-layout push that the repository had moved from `vlm-bench` to `vision-pathways`. The static website still linked to the old repository name and would therefore depend on GitHub redirects.

## Files And Behavior Changed

- Updated the repository, paper-kit, and evidence links in `docs/index.html` to use `TarunTomar122/vision-pathways` directly.
- Updated the local `origin` remote separately to the canonical SSH URL; that local Git configuration is not tracked in this commit.

## Alternatives Considered

- Keep the old links and rely on GitHub redirects. This was rejected because redirects are unnecessary and can stop working if the old name is reused.
- Parameterize the repository URL during generation. This was rejected because the site is currently a small static page with only three repository links.

## Verification Evidence

- Searched the live site and publication files for remaining `TarunTomar122/vlm-bench` references.
- Ran `scripts/verify_submission.py` through the submission target.
- Ran `git --git-dir=.git-data --work-tree=. diff --cached --check` before committing.

## Known Limitations And Unsupported Claims

- This change only updates repository links; it does not enable GitHub Pages in repository settings.
- It does not configure a custom domain or change experimental outputs.

## Next Action Enabled

Enable GitHub Pages from `main` and `/docs` at the canonical `vision-pathways` repository.
