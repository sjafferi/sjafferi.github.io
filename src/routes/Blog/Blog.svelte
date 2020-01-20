<script>
  import { Router, Link, Route } from "svelte-routing";
  import { groupBy } from "../../util.js";
  import Post from "../../components/Post/Page.svelte";
  import Posts from "./posts.js";

  const algorithms = groupBy("parent")(
    Posts.filter(({ tags }) => tags.includes("algorithms"))
  );
</script>

<style>
  .list h3 {
    font-size: 1.5em;
    font-weight: 700;
    font-variant: small-caps;
    margin-top: 4px;
    line-height: 1.125;
  }
  .list li {
    line-height: 1.55;
  }

  .container {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    padding: 0 20px;
    margin-left: 30px;
  }

  :global(.list a) {
    color: #333;
  }
  :global(.list a:hover) {
    color: #888;
  }
</style>

<div class="container">

  <Router>
    <Route path="/">
      <div class="list">
        <h3>Algorithms</h3>
        <ul>
          {#each Object.entries(algorithms) as [group, posts]}
            <h4>{group}</h4>
            <ul>
              {#each posts as post}
                <li>
                  <Link to={post.slug}>{post.title}</Link>
                </li>
              {/each}
            </ul>
          {/each}
        </ul>
      </div>

      <div class="list">
        <h3>Practical</h3>
        <ul>
          {#each Posts as post}
            {#if post.tags.includes('practical')}
              <li>
                <Link to={post.slug}>{post.title}</Link>
              </li>
            {/if}
          {/each}
        </ul>
      </div>

    </Route>
    {#each Posts as post}
      <Route path={post.slug}>
        <Post {...post} />
      </Route>
    {/each}
  </Router>

</div>
