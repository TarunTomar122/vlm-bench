# GitHub Pages Research Website

`index.html`, `styles.css`, and `assets/` form a static, dependency-free research page. Running
`make paper-assets` refreshes the figure images in `assets/` from frozen experiment JSON.

In the GitHub repository settings, set Pages to deploy from the `main` branch and `/docs` folder.
The site uses only relative asset paths, so it works whether the repository keeps its current name
or is renamed later.

For Wix, use this folder as the visual/content source: recreate the sections with Wix components,
upload the generated PNG assets, and preserve the evidence caveats verbatim. If the Wix plan allows
custom embeds, the GitHub Pages site can be embedded or linked. The published paper is
[arXiv:2607.17052](https://arxiv.org/abs/2607.17052).
