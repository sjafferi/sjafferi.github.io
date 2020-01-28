<script>
  import { Remarkable } from "remarkable";
  import hljs from "highlight.js";
  // import "highlight.js/styles/agate.css";

  import { toSlug } from "../util.js";

  export let content;

  const md = new Remarkable({
    highlight: function(str, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(lang, str).value;
        } catch (err) {}
      }

      try {
        return hljs.highlightAuto(str).value;
      } catch (err) {}

      return ""; // use external default escaping
    }
  });

  function anchors(md) {
    md.renderer.rules.heading_open = function(tokens, idx /*, options, env */) {
      return `<h${tokens[idx].hLevel} id='${toSlug(tokens[idx + 1].content)}'>`;
    };

    md.renderer.rules.heading_close = function(
      tokens,
      idx /*, options, env */
    ) {
      return `</h${tokens[idx].hLevel}>\n`;
    };
  }

  md.use(anchors);
</script>

<style>
  :global(.markdown h1, .page h1) {
    font-feature-settings: "smcp";
    font-size: 1.75em;
    line-height: 1.25;
    letter-spacing: -0.75px;
  }

  :global(.page h1, .markdown h1, .markdown h2, .markdown h3, .markdown
      h4, .markdown h5, .markdown h6) {
    margin: 1.25em 0 0.5em -0.75rem;
    font-weight: bold;
    position: relative;
  }

  :global(.markdown h2) {
    text-transform: uppercase;
    font-size: 1.25em;
    padding: 0 0.5em 0 0;
    line-height: 1.25;
  }

  :global(.markdown p) {
    line-height: 1.55;
  }

  :global(.markdown ul li) {
    margin: 10px 5px;
  }
</style>

<div class="markdown">
  {@html md.render(content)}
</div>
