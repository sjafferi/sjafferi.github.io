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

<style>
  .container {
    width: 100%;
    padding: 0 20px;
  }
  .posts {
    list-style-type: none;
    margin-top: 85px;
  }

  .posts li {
    margin: 30px 0;
  }

  .posts li > * {
    margin: 15px 0;
  }

  :global(.posts .title a) {
    font-size: 24px;
    font-weight: 600;
    text-decoration: none !important;
    font-variant: small-caps;
    color: #333;
    line-height: 30px;
  }

  :global(.posts .title a:hover) {
    color: #888;
  }

  .posts p {
    line-height: 4px;
  }

  .posts .date {
    font-size: 12px;
  }

  .posts .subtitle {
    font-size: 16px;
    margin: 20px 0;
  }

  .posts .tags {
    display: flex;
  }

  /* @media (max-width: 550px) {
    .container {
      margin: 0;
    }
    .posts {
      margin: 0;
      padding: 0;
    }
    .posts li {
      margin: 45px 0;
    }
    .posts .subtitle {
      line-height: 1.5;
    }
  } */
</style>

<svelte:head>
  <title>Writings | Sibtain Jafferi</title>
</svelte:head>

<div class="container">

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
