# Fix README Math Rendering

## Intended Commit Subject

`Fix GitHub README math rendering`

## Problem Or Decision

GitHub displayed the four equations in the evolutionary-search explanation as raw LaTeX because the
README used `\[` and `\]` delimiters. GitHub Markdown supports display equations with paired `$$`
delimiters, so the notation was correct but the surrounding Markdown syntax was not.

## Exact Changes

- Replaced the four `\[` and `\]` display-math pairs in `README.md` with `$$` pairs.
- Preserved every equation, symbol, coefficient, and surrounding explanation without changing the
  documented search method.

## Alternatives Considered

- Rendering the equations as images was rejected because images are harder to maintain, search, copy,
  and read with assistive technology.
- Rewriting every equation as inline code was rejected because it would remain visually noisy and
  would not provide mathematical typesetting.
- Removing the equations was rejected because they precisely define the experiment's objectives.

## Verification Evidence

- Confirmed that no `\[` or `\]` delimiters remain in `README.md`.
- Confirmed that the README contains eight standalone `$$` lines for four balanced display equations.
- Ran `git diff --check`.
- Audited `README.md` and `docs/` for prohibited en dash and em dash characters.

## Known Limitations And Unsupported Claims

- This change only fixes GitHub's display-math delimiters; it does not alter or revalidate the search
  formulas themselves.
- Other Markdown viewers without math support may still show the LaTeX source between `$$` markers.

## Next Action Enabled

Readers can inspect the genetic-search objective directly in the GitHub README without raw delimiter
syntax obscuring the equations.
