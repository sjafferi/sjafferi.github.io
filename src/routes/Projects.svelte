<script>
  import { Images } from "svelte-images";
  import Projects from "metadata/projects.js";
</script>

<style>
  .projects {
    display: flex;
    flex-flow: column;
    margin-left: 4vw;
    margin-top: 4vw;
    width: 110%;
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
    color: #24292e;
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

  .header-link {
    color: #333;
    text-decoration: none;
    font-size: 1.5em;
    font-weight: 600;
  }
  .header-link:hover {
    color: #888;
  }

  .description {
    color: #586069;
    font-size: 1em;
    line-height: 1.5;
  }

  .tags {
    display: flex;
  }
  .tag {
    display: inline-block;
    padding: 0.3em 0.9em;
    margin: 0 0.5em 0.5em 0;
    white-space: nowrap;
    background-color: #f1f8ff;
    border-radius: 3px;
  }

  .images {
    padding-bottom: 20px;
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
            <div class="tag">{tag}</div>
          {/each}
        </div>
      </div>
      <div class="links">
        {#each links as { link, text }}
          <a class="link-btn" target="_blank" href={link}>{text}</a>
        {/each}
      </div>
    </div>
  {/each}
</div>
