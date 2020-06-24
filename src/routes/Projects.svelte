<script>
  import { Images } from "svelte-images";
  import { fade } from 'svelte/transition';
  import Tag from "components/Tag.svelte";
  import Projects from "metadata/projects.js";
</script>

<style lang="scss">
  $text-color: var(--text-color);

  .projects {
    display: flex;
    flex-flow: column;
  }

  .tile {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #e1e4e8;
    margin: 30px 0;
  }

  .content,
  .links {
    display: flex;
    flex-flow: column;
  }

  .links {
    justify-content: center;
  }
  .link-btn {
    display: inline-block;
    padding: 0.5em 1em;
    margin: 0.1em 0.65em 1em 0;
    white-space: nowrap;
    color: var(--text-color);
    background-color: #eff3f6;
    background-image: linear-gradient(-180deg, #fafbfc, #eff3f6 90%);
    border: 1px solid rgba(27, 31, 35, 0.2);
    text-align: center;
    border-radius: 3px;
    text-decoration: none;
  }
  .link-btn:hover {
    background-color: #e6ebf1;
    background-image: linear-gradient(-180deg, #f0f3f6, #e6ebf1 90%);
    background-position: -0.5em;
    border-color: rgba(27, 31, 35, 0.35);
  }

  :global(html.dark .link-btn) {
    background-image: linear-gradient(-180deg, #505050, #202020 90%);
    &:hover {
      background-image: linear-gradient(-180deg, #505050, #686868 100%);
      border-color: rgba(27, 31, 35, 0.35);
    }
  }

  .header-link {
    color: var(--text-color);
    text-decoration: none;
    font-size: 1.75em;
    font-weight: 600;
  }
  .header-link:hover {
    opacity: 0.75;
  }

  .description {
    color: var(--text-color);
    font-size: 1.25em;
    line-height: 1.5;
  }

  .tags {
    display: flex;
  }

  .images {
    padding-bottom: 20px;
  }

  :global(.images .nav button) {
    background: #b0afafc9 !important;
  }

  @media (max-width: 550px) {
    .projects {
      margin: 0;
      margin-top: 2rem;
      padding: 0;
      width: 100%;
    }
    .tile {
      flex-flow: column;
      padding: 20px;
    }
    .tags {
      flex-flow: wrap;
      justify-content: center;
    }
    a, p {
      text-align: center;
    }
    :global(.svelte-images-gallery) {
      justify-content: center;
    }
    .links {
      margin-top: 1rem;
    }
  }
</style>

<svelte:head>
  <title>Projects | Sibtain Jafferi</title>
</svelte:head>
<div class="projects">
  {#each Projects as { title, titleLink, description, images, tags, links }}
    <div class="tile">
      <div class="content">
        <a class="header-link" href={titleLink} target="_blank">{title}</a>
        <p class="description">{description}</p>
        {#if images && images.length > 0}
          <div class="images">
            <Images numCols={3} {images} />
          </div>
        {/if}
        <div class="tags">
          {#each tags as tag}
            <Tag>{tag}</Tag>
          {/each}
        </div>
      </div>
      <div class="links">
        {#each links as { link, text }}
          <a in:fade class="link-btn" target="_blank" href={link}>{text}</a>
        {/each}
      </div>
    </div>
  {/each}
</div>
