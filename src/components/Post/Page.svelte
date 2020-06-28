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
  }

  .content {
    // display: flex;
  }

  .markdown {
    max-width: 55rem;
  }

  header h1 {
    text-align: center;
    text-transform: none;
    font-size: 2.25em;
    line-height: 1.25;
    font-weight: 600;
    letter-spacing: 0.5px;
    font-family: Raleway, sans-serif;
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
      float: right;
    }
    .post-page {
      box-shadow: none !important;
      background: none !important;

      header h1 {
        font-size: 1.5em;
      }
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
    <div class="content">
      <div class="table-of-contents">
        <TOC {content} />
      </div>
      <div class="markdown">
        <Markdown {content} />
      </div>
    </div>
  </article>
</div>
