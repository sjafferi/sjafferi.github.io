<script>
  import { Router, Route } from "svelte-routing";
  import Sun from "components/Sun.svelte";
  import Nav from "./routes/Nav.svelte";
  import Projects from "./routes/Projects.svelte";
  import Blog from "./routes/Blog/Blog.svelte";
  import Me from "./routes/Me.svelte";

  // Used for SSR. A falsy value is ignored by the Router.
  export let url = "";
</script>

<style>
  :global(body) {
    /* background: linear-gradient(
      rgba(74, 88, 103, 1) 18%,
      rgba(36, 41, 47, 1) 57%,
      rgb(0, 0, 0) 100%
    ); */
  }

  .overlay-img-container {
    width: 73%;
    height: 100%;
    position: fixed;
    pointer-events: none;
    max-width: 170ch;
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

  .page {
    width: 100%;
    z-index: 2;
  }

  @media (max-width: 550px) {
    .container {
      overflow-x: hidden;
      flex-flow: column;
      padding: 0;
    }
    .overlay-img-container {
      width: 100vw;
      height: 100vh;
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
  <!-- <div class="overlay-img-container">
    <img src="images/light-background.png" />
  </div> -->
  <Router {url}>
    <Sun />
    <Nav />
    <div class="page">
      <Route path="projects" component={Projects} />
      <Route path="/" component={Me} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
