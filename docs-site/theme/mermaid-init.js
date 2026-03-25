// Mermaid diagram rendering for mdBook
// Loads mermaid from CDN and renders ```mermaid code blocks
(function () {
    var defined = false;

    function renderMermaid() {
        if (defined) return;
        defined = true;

        mermaid.initialize({
            startOnLoad: false,
            theme: document.querySelector('html').classList.contains('light')
                ? 'default'
                : 'dark',
        });

        document.querySelectorAll('pre code.language-mermaid').forEach(function (el) {
            var pre = el.parentElement;
            var container = document.createElement('div');
            container.className = 'mermaid';
            container.textContent = el.textContent;
            pre.parentElement.replaceChild(container, pre);
        });

        mermaid.run();
    }

    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js';
    script.onload = renderMermaid;
    document.head.appendChild(script);
})();
