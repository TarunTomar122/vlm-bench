# Restore Figure Colors

## Intended Commit Subject

`Restore publication figure colors`

## Problem Or Decision

The website applied a global grayscale filter to every generated figure. Several charts encode route families through color, so grayscale made distinct legend entries appear duplicated even though the underlying publication assets were correct.

## Files And Behavior Changed

- Removed the grayscale image filter from `docs/styles.css`.
- All seven generated website figures now display their original publication colors.
- The surrounding website remains a monochrome light document design.

## Alternatives Considered

- Redesign every chart with additional marker shapes and line patterns. This was not required because the generated assets already have distinct colors and line styles.
- Keep grayscale and add text labels directly to each series. This was rejected because it would crowd the charts and discard the existing color encoding.

## Verification Evidence

- Confirmed that the Qwen budget chart contains distinct blue, gray, green, and orange legend entries in the generated source asset.
- Searched `docs/` for remaining grayscale or image-filter rules.
- Ran `make PYTHON=/tmp/vlm-paper-venv/bin/python submission`.
- Ran `git --git-dir=.git-data --work-tree=. diff --check`.

## Known Limitations And Unsupported Claims

- The charts depend partly on color, although several also use solid and dashed line styles.
- This presentation fix does not change plot data, experimental results, or research claims.

## Next Action Enabled

Deploy the corrected CSS through GitHub Pages so all publication figures render in color.
