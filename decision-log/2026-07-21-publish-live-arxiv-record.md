# Publish Live arXiv Record

## Intended Commit Subject

`Publish live arXiv record`

## Problem Or Decision

The paper was announced as arXiv:2607.17052 on 19 July 2026, but repository-facing copy still
described a pending submission and lacked a machine-readable citation for the published paper. The
decision is to make the official arXiv abstract and PDF the canonical public paper links while
leaving the v1 manuscript source unchanged.

## Files And Behavior Changed

- `README.md` links readers to the official arXiv record and PDF.
- `CITATION.cff` adds the arXiv DOI and a preferred article citation.
- `docs/index.html` replaces the pending-submission status with the announced arXiv record and
  exposes official abstract and PDF links.
- `docs/README.md` records the live paper URL for Pages and Wix reuse.
- `paper/README.md` and `paper/arxiv-metadata.md` distinguish the announced v1 archive from a
  future-version workflow.
- `paper/submission-checklist.md` records the completed v1 submission and announcement actions
  without asserting unfinished human-review items.
- The `arxiv-v1-candidate` GitHub release title and notes now point to the live arXiv paper.

## Alternatives Considered

- Keeping GitHub's locally compiled PDF as the primary paper link was rejected because arXiv is the
  archival, announced record readers should cite and share.
- Editing `paper/main.tex` after announcement was rejected because it would create an unversioned
  mismatch with the archived v1 source.
- Releasing a second source archive was rejected because the existing release asset exactly matches
  the submitted v1 package.

## Verification Evidence

- Checked the official arXiv record: title, author, `cs.CV` category, 14 pages, eight figures,
  v1 date, identifier, and DOI.
- Rebuilt and verified the repository Pages assets with `make submission`.
- Parsed `CITATION.cff` and checked the preferred citation fields.
- Checked Markdown and HTML links, `git diff --check`, and the GitHub release metadata.

## Known Limitations And Unsupported Claims

- The arXiv DOI is issued through arXiv/DataCite and is not a journal DOI.
- This announcement does not imply peer review, acceptance, or a venue publication.
- The source archive remains v1; substantive manuscript changes require a new arXiv version.

## Next Action Enabled

Readers can cite the announced paper from GitHub and follow the official record, PDF, and archived
source package without encountering stale submission-pending language.
