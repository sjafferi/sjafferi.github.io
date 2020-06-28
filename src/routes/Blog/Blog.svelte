<script>
  import moment from "moment";
  import { Router, Link, Route } from "svelte-routing";
  import { groupBy } from "util/index.js";
  import Post from "components/Post/Page.svelte";
  import List from "components/List.svelte";
  import Tag from "components/Tag.svelte";
  import Posts from "metadata/posts.js";

  const posts = Posts.sort((a, b) => moment(a.date, "MM/DD/YYYY").isBefore(moment(b.date, "MM/DD/YYY")) ? 1 : -1); // descending order of date
</script>

<style lang="scss">
  .blog-container {
    width: 100%;
    font-family: "Roboto", sans-serif;
  }

  .posts {
    list-style-type: none;
    margin: 0;
    padding: 0;
  }

  .posts li {
    border-bottom: 1px solid #a8a8a84a;
    padding-bottom: 10px;
    margin: 0;
    margin-bottom: 30px;
  }

  .posts li > * {
    margin: 15px 0;
  }

  :global(.posts .title a) {
    // font-size: 24px;
    font-size: 1.5rem;
    font-weight: 600;
    text-decoration: none !important;
    font-variant: small-caps;
    color: var(--text-color);
    line-height: 30px;
  }

  :global(.posts .title a:hover) {
    opacity: 0.75;
  }

  .posts p {
    line-height: 2rem;
    font-size: 1.3rem;
    margin: 20px 0;
  }

  .posts .date {
    // font-size: 12px;
    font-size: 1rem;
  }

  .posts .subtitle {
    // font-size: 16px;
    font-size: 1.25rem;
    margin: 20px 0;
  }

  .posts .tags {
    display: flex;
  }

  @media (max-width: 1500px) and (min-width: 850px) {
    .blog-container ul {
      margin-left: 10%;
    }
  }

  @media (max-width: 850px) {
    .blog-container {
      margin: 0;
    }
    :global(.posts .title a) {
      font-size: 24px !important;
      font-variant: small-caps;
    }
    .posts .date {
      font-size: 14px;
    }
    .posts .subtitle {
      font-size: 16px;
      line-height: 25px;
    }

    .posts {
      margin-top: 50px;
    }
  }
</style>

<svelte:head>
  <title>Writings | Sibtain Jafferi</title>
</svelte:head>

<div class="blog-container">

  <Router>
    <Route path="/">
      <ul class="posts">
        {#each posts as { title, slug, subtitle, date, tags }}
          <li>
            <div class="title"><Link to={slug}>{title}</Link></div>
            <p class="date">Created: {moment(date, 'MM/DD/YYYY').format('MMM Do YYYY')}</p>
            <p class="subtitle">{subtitle}</p>
            <div class="tags">
              {#each tags as tag}
                <Tag>{tag}</Tag>
              {/each}
            </div>
          </li>
        {/each}
      </ul>
    </Route>

    {#each Posts as post}
      <Route path={post.slug}>
        <Post {...post} />
      </Route>
    {/each}
  </Router>

</div>
