<script>
  import { onMount } from "svelte";
  import moment from "moment";
  import Markdown from "../Markdown.svelte";
  import TOC from "./TOC.svelte";

  export let title, subtitle, date, tags, content, slug;

  onMount(() => {
    fetch(`/get-post`, {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ slug }),
    })
      .then((resp) => resp.json())
      .then(({ post }) => {
        content = post;
      });
  });
</script>

<style>
  .markdown {
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

  .page h1 {
    font-feature-settings: "smcp";
    font-size: 1.75em;
    line-height: 1.25;
    letter-spacing: -0.75px;
    margin: 1.25em 0 0.5em -0.75rem;
    font-weight: bold;
    position: relative;
  }

  .page-metadata {
    display: flex;
    flex-direction: column;
    margin: 20px 0;
  }
  .page-metadata .subtitle {
    text-align: center;
    font-style: italic;
  }
  .page-metadata .date {
    font-style: italic;
  }
  .page-metadata .date-container {
    margin-top: 18px;
  }

  @media (max-width: 800px) {
    .markdown {
      width: auto;
    }
  }
</style>

<div class="page">
  <header>
    <h1>{title}</h1>
  </header>

  <article>
    <div class="page-metadata">
      {#if subtitle}
      <span class="subtitle">{subtitle}</span>
      {/if} {#if date}
      <span class="date-container">
        Created:
        <span class="date">
          {moment(date, 'MM/DD/YYYY').format('MMM Do YYYY')}
        </span>
      </span>
      {/if}
    </div>
    <div class="table-of-contents">
      <TOC {content} />
    </div>
    <div class="markdown">
      <Markdown {content} />
    </div>
  </article>
</div>
