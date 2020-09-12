<script>
  import { Remarkable } from "remarkable";
  import hljs from "highlight.js";

  import { toSlug } from "util/index.js";

  export let content;

  const md = new Remarkable({
    langPrefix: "hljs language-",
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

<style lang="scss">
  @mixin headings {
    :global(.page h1, .markdown h1, .markdown h2, .markdown h3, .markdown
        h4, .markdown h5, .markdown h6) {
      @content;
    }
  }
  @mixin text {
    :global(.markdown p, .markdown ul, .markdown ol, .markdown a) {
      @content;
    }
  }
  @mixin mobile {
    @media screen and (max-width: 850px) {
      @content;
    }
  }
  @mixin lists {
    :global(.markdown ul, .markdown ol, .markdown li) {
      @content;
    }
  }

  :global(.markdown) {
    @include text {
      overflow: hidden;
      font-size: 1.2rem;
      line-height: 1.25;
      word-break: break-word;
      hyphens: auto;
    }
    ol {
      line-height: 23px;
    }
  }

  @include headings {
    margin: 1.5em 0 0.75em 0;
    font-weight: bold;
    position: relative;
    padding: 0 0 5px 0;
    line-height: 1.25;
    font-size: 1.25em;
    box-shadow: 0 -1px 0px 0 #848484 inset, 0 -1px 0 0 #000 inset;
    overflow: hidden;
    text-transform: uppercase;
    padding: 0 0.5em 0 0;
  }

  :global(.markdown h1, .page h1) {
    font-feature-settings: "smcp";
    font-size: 2em;
    line-height: 1.25;
    box-shadow: 0 -1px 0px 0 #848484 inset, 0 -2px 0 0 #888 inset;
  }

  @for $index from 1 through 4 {
    :global(.markdown h#{$index}) {
      font-size: 2em - 0.25em * $index;
    }
  }

  :global(.markdown ul li, .markdown ol li) {
    margin: 10px 5px;
  }

  :global(.markdown a) {
    color: #a00000 !important;
  }

  :global(.markdown table th, .markdown table td) {
    padding: 5px 10px;
  }

  :global(.markdown p.image-container) {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 1em 0;
  }

  :global(.markdown p.image-container img) {
    max-width: 99%;
    box-shadow: 3px 4px 8px 0 rgba(0, 0, 0, 0.2);
    margin: 9px 0;
  }

  :global(.markdown p.image-container + em) {
    width: 100%;
    text-align: center;
  }

  :global(.markdown pre > code) {
    border: 0;
    margin: 0;
    padding: 0;
    word-break: break-all;
    white-space: pre-wrap;
    font-size: 1rem;
  }

  :global(.markdown .highlight) {
    margin-bottom: 16px;
  }

  :global(.markdown pre) {
    margin: 16px 0;
    line-height: 1.45;
    border-radius: 3px;
    word-break: normal;
    overflow: hidden !important;
    background: #f6f8fa;
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

  @include mobile {
    .markdown {
      max-width: 95vw;
    }
    @include headings {
      margin: 0.75em 0 0.65em 0;
    }
    :global(.markdown pre > code) {
      font-size: 12px;
    }
  }

  :global(html.dark) {
    :global(.markdown a) {
      color: #9693ff !important;
    }
    :global(.markdown pre) {
      background: transparent !important;
    }
  }
</style>

<div class="markdown">
  {@html md.render(content)}
</div>
