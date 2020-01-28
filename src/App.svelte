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
    overflow-x: hidden;
  }
</style>

<div class="container">
  <Router {url}>
    <Nav />
    <div>
      <Route path="projects" component={Projects} />
      <Route path="/" component={Me} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
