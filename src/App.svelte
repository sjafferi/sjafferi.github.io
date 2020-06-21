<script>
  import { onMount } from 'svelte';
  import { Router, Route } from "svelte-routing";
  import { themeManager } from 'stores';
  import Sun from "components/Sun.svelte";
  import Moon from "components/Moon.svelte";
  import Nav from "./routes/Nav.svelte";
  import Projects from "./routes/Projects.svelte";
  import Blog from "./routes/Blog/Blog.svelte";
  import Me from "./routes/Me.svelte";
  import "./main.scss";
  // Used for SSR. A falsy value is ignored by the Router.
  export let url = "";

  
  onMount(() => {
    themeManager.toggle();
    if (location.pathname === '/') {
      location.href = '/about'
    }
  });
</script>

<style>
  :global(body) {
    /* background: linear-gradient(
      rgba(74, 88, 103, 1) 18%,
      rgba(36, 41, 47, 1) 57%,
      rgb(0, 0, 0) 100%
    ); */
  }

  :global(html.dark) {
    background: rgb(44, 62, 80);
  }

  img {
    position: absolute;
    right: 0;
    bottom: 0;
    opacity: 0.2;
  }
  
  .container {
    max-width: 110ch;
    margin: auto;
    padding: 0 20px;
    display: flex;
    flex-direction: column;
  }

  .section {
    width: 100%;
    z-index: 2;
  }

  @media (max-width: 550px) {
    .container {
      overflow-x: hidden;
      flex-flow: column;
      padding: 0;
    }

    img {
      max-width: 100vw;
    }
  }
</style>

<svelte:head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>


<div class="container">
  <Router {url}>
    <Sun />
    <Moon />
    <Nav />
    <div class="section">
      <Route path="projects" component={Projects} />
      <Route path="about" component={Me} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
