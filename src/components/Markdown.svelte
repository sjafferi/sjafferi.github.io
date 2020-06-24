<script>
  import { Remarkable } from "remarkable";
  import hljs from "highlight.js";
  import "highlight.js/scss/agate.scss";

  import { toSlug } from "util/index.js";

  export let content;

  const md = new Remarkable({
    highlight: function (str, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(lang, str).value;
        } catch (err) {}
      }

      try {
        return hljs.highlightAuto(str).value;
      } catch (err) {}

      return ""; // use external default escaping
    },
  });

  function plugin(md) {
    md.renderer.rules.heading_open = function (tokens, idx) {
      return `<h${tokens[idx].hLevel} id='${toSlug(tokens[idx + 1].content)}'>`;
    };

    md.renderer.rules.heading_close = function (tokens, idx) {
      return `</h${tokens[idx].hLevel}>\n`;
    };

    md.renderer.rules.image = function (tokens, idx, options /*, env */) {
      const src = ` src="${tokens[idx].src}"`;
      const title = tokens[idx].title ? ` title="${tokens[idx].title}"` : "";
      const alt = ' alt="' + (tokens[idx].alt ? tokens[idx].alt : "") + '"';
      const suffix = options.xhtmlOut ? " /" : "";
      return `<p class="image-container"> <img ${src} ${alt} ${title} ${suffix}> </p>`;
    };
  }

  md.use(plugin);
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
    margin: 1.25em 0 0.5em 0;
    font-weight: bold;
    position: relative;
  }

  :global(.markdown h2) {
    text-transform: uppercase;
    font-size: 1.3em;
    padding: 0 0.5em 0 0;
    line-height: 1.25;
  }

  :global(.markdown h5) {
    font-size: 1em;
  }

  :global(.markdown p, .markdown li) {
    line-height: 1.55;
    font-size: 1.25rem;
  }

  :global(.markdown ul li) {
    margin: 10px 5px;
  }

  :global(.markdown table th, .markdown table td) {
    padding: 5px 10px;
  }

  :global(.markdown p.image-container) {
    display: flex;
    justify-content: center;
  }

  :global(.markdown p.image-container img) {
    max-width: 50vw;
  }

  :global(.markdown p.image-container + em) {
    width: 100%;
    text-align: center;
  }

  :global(.markdown code) {
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    margin: 0;
    padding: 0.2em 0.4em;
  }

  :global(.markdown pre) {
    word-wrap: normal;
  }

  :global(.markdown pre > code) {
    background: transparent;
    border: 0;
    margin: 0;
    padding: 0;
    white-space: pre;
    word-break: normal;
  }

  :global(.markdown .highlight) {
    margin-bottom: 16px;
  }

  :global(.markdown .highlight pre) {
    margin-bottom: 0;
    word-break: normal;
  }

  :global(.markdown .highlight pre),
  :global(.markdown pre) {
    background-color: #2b2b2b;
    border-radius: 3px;
    line-height: 1.45;
    overflow: auto;
    padding: 16px;
  }

  :global(.markdown pre code) {
    color: #b8b8b8;
    background-color: transparent;
    border: 0;
    display: inline;
    line-height: inherit;
    margin: 0;
    max-width: auto;
    overflow: visible;
    padding: 0;
    word-wrap: normal;
  }

  :global(.markdown ol > li) {
    margin: 10px 0;
  }

  @media screen and (max-width: 550px) {
    :global(.markdown p.image-container img) {
      max-width: 85vw;
    }
    :global(.markdown p, .markdown li) {
      font-size: 1rem;
    }
  }
</style>

<div class="markdown">
  {@html md.render(content)}
</div>
