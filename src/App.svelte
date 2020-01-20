<script>
  import { Router, Route } from "svelte-routing";
  import Nav from "./routes/Nav.svelte";
  import Home from "./routes/Home.svelte";
  import Projects from "./routes/Projects.svelte";
  import Blog from "./routes/Blog/Blog.svelte";

  // Used for SSR. A falsy value is ignored by the Router.
  export let url = "";
</script>

<style>
  .container {
    max-width: 110ch;
    margin: auto;
    margin-top: 5vw;
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
      <Route path="/*" component={Blog} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
