<script>
  import { Router, Route } from "svelte-routing";
  import Nav from "./routes/Nav.svelte";
  import Projects from "./routes/Projects.svelte";
  import Blog from "./routes/Blog/Blog.svelte";
  import Me from "./routes/Me.svelte";

  // Used for SSR. A falsy value is ignored by the Router.
  export let url = "";
</script>

<style>
  .container {
    max-width: 110ch;
    margin: auto;
    padding: 0 20px;
    display: flex;
  }

  .page {
    width: 100%;
  }

  @media (max-width: 550px) {
    .container {
      overflow-x: hidden;
      flex-flow: column;
      padding: 0;
    }
  }
</style>

<svelte:head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>

<div class="container">
  <Router {url}>
    <Nav />
    <div class="page">
      <Route path="projects" component={Projects} />
      <Route path="/" component={Me} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
