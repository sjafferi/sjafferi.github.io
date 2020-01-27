<script>
  import { Router, Link, Route } from "svelte-routing";
  import { groupBy } from "../../util.js";
  import Post from "../../components/Post/Page.svelte";
  import List from "../../components/List.svelte";
  import Posts from "./posts.js";

  const algorithms = groupBy("parent")(
    Posts.filter(({ tags }) => tags.includes("algorithms"))
  );
</script>

<style>
  .container {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    padding: 0 20px;
    margin-left: 30px;
  }
</style>

<svelte:head>
  <title>Writings | Sibtain Jafferi</title>
</svelte:head>

<div class="container">

  <Router>
    <Route path="/">
      <!-- <List title="Algorithms">
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
      </List> -->

      <List title="Technical">
        {#each Posts as post}
          {#if post.tags.includes('technical')}
            <li>
              <Link to={post.slug}>{post.title}</Link>
            </li>
          {/if}
        {/each}
      </List>

      <List title="Practical">
        {#each Posts as post}
          {#if post.tags.includes('practical')}
            <li>
              <Link to={post.slug}>{post.title}</Link>
            </li>
          {/if}
        {/each}
      </List>

    </Route>

    {#each Posts as post}
      <Route path={post.slug}>
        <Post {...post} />
      </Route>
    {/each}
  </Router>

</div>
