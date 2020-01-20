<script>
  import { onMount } from "svelte";
  import { Remarkable } from "remarkable";
  import hljs from "highlight.js";
  // import "highlight.js/styles/agate.css";
  export let title, subtitle, date, tags, content, slug;

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

  onMount(() => {
    fetch(`/get-post`, {
      method: "post",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ slug })
    })
      .then(resp => resp.json())
      .then(({ post }) => {
        content = post;
      });
  });
</script>

<style>
  .page {
    width: 70vw;
    max-width: 85ch;
  }

  header h1 {
    margin: 0.75em 0;
    margin-top: 0;
    text-align: center;
    text-transform: none;
    font-variant: small-caps;
    font-size: 2.5em;
    line-height: 1.15;
    font-weight: 600;
    letter-spacing: -1px;
  }

  :global(.page h1) {
    font-feature-settings: "smcp";
    font-size: 1.75em;
    line-height: 1.25;
    letter-spacing: -0.75px;
  }

  :global(.page h1, .page h2, .page h3, .page h4, .page h5, .page h6) {
    margin: 1.25em 0 0.5em -0.75rem;
    font-weight: bold;
    position: relative;
  }

  :global(.page h2) {
    text-transform: uppercase;
    font-size: 1.25em;
    padding: 0 0.5em 0 0;
    line-height: 1.25;
  }

  :global(.page p) {
    line-height: 1.55;
  }
</style>

<div class="page">

  <header>
    <h1>{title}</h1>
  </header>

  <article>
    <div class="page-metadata" />
    <div class="table-of-contents" />
    <div class="markdown">
      {@html md.render(content)}
    </div>
  </article>
</div>
