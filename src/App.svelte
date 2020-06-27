<script>
  import { onMount, onDestroy } from 'svelte';
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
  let theme, unsubscribe;
  let currentPage;
  
  onMount(() => {
    themeManager.initialize();
    themeManager.toggle();
    if (location.pathname === '/') {
      location.href = '/about'
    }
    if (!unsubscribe) {
      unsubscribe = themeManager.theme.subscribe(value => theme = value);
    }
  });

  onDestroy(() => {
    if (unsubscribe) {
      unsubscribe();
      themeManager.destroy();
    }
  })

  $: about = !currentPage || currentPage === 'about';
</script>

<style>
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
    z-index: 300;
    margin-top: 5rem;
  }

  .about {
    --theme-changer-top: 45%;
    --theme-changer-left: 60%;
    --sun-size: 125px;
    --moon-size: 100px;
  }

  @media (max-width: 800px) {
    :global(.about > *) {
      --theme-changer-top: 18%;
      --theme-changer-left: calc(50% - 33px);
    }
    .container {
      overflow-x: hidden;
      flex-flow: column;
      padding: 0;
    }

    .section {
      margin-top: 2rem;
      padding: 0.6rem;
    }

    img {
      max-width: 100vw;
    }
  }
</style>

<svelte:head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>

<div class="container" class:about>
  <Router {url}>
    {#if theme == "light"}
      <Sun on:click={themeManager.toggle} {about} />
    {:else}
      <Moon on:click={themeManager.toggle} {about} />
    {/if}
    <Nav bind:currentPage={currentPage} />
    <div class="section">
      <Route path="projects" component={Projects} />
      <Route path="about" component={Me} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
