// MathJax configuration for mkdocs-material + pymdownx.arithmatex (generic mode).
//
// IMPORTANT: do NOT add polyfill.io here.  That domain was sold to a
// malicious operator in 2024 and injects fake sign-in modals.  MathJax 3
// runs in every modern browser without it.

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
