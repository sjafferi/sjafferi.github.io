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

<style lang="scss">
  .post-page {
    width: 100%;
    display: flex;
    flex-flow: column;
    align-items: center;
    box-shadow: 0 4px 16px 0 rgba(33, 33, 33, 0.2);
    background: white;
    border: grey;
  }

  :global(html.dark) {
    .post-page {
      box-shadow: none;
      box-shadow: 0 2px 12px 0 rgba(255, 255, 255, 0.2);
      background: #3c3c3c;
    }
  }

  .markdown {
    width: 70vw;
    max-width: 85ch;
  }

  header h1 {
    // text-align: center;
    text-transform: none;
    // font-variant: small-caps;
    font-size: 2.5em;
    line-height: 1.15;
    font-weight: 600;
    letter-spacing: -1px;
  }

  .post-page h1 {
    font-size: 1.75em;
    line-height: 1.25;
    letter-spacing: 0.5px;
    margin: 1.25em 0 0.5em 0rem;
    font-weight: bold;
    position: relative;
  }

  .page-metadata {
    display: flex;
    flex-direction: column;
    margin: 5px 0 25px 0;
  }
  .page-metadata .subtitle {
    text-align: center;
    // font-style: italic;
    font-size: 21px;
  }
  .page-metadata .date {
    font-style: italic;
  }
  .page-metadata .date-container {
    margin-top: 18px;
  }

  @media (max-width: 850px) {
    .markdown {
      width: auto;
    }
    .post-page {
      box-shadow: none !important;
      background: none !important;
    }
  }
</style>

<div class="post-page">
  <header>
    <h1>{title}</h1>
  </header>

  <article>
    <div class="page-metadata">
      <!-- {#if subtitle}
        <span class="subtitle">{subtitle}</span>
      {/if}  -->
      {#if date}
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
