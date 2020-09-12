<script>
  import { onMount, onDestroy } from "svelte";
  import { Router, Route } from "svelte-routing";
  import { themeManager, router } from "stores";
  import Sun from "components/Sun.svelte";
  import Moon from "components/Moon.svelte";
  import Nav from "./routes/Nav.svelte";
  import Projects from "./routes/Projects.svelte";
  import ThemeSwitcher from "components/ThemeSwitcher.svelte";
  import Blog from "./routes/Blog/Blog.svelte";
  import Me from "./routes/Me.svelte";
  import "./main.scss";
  // Used for SSR. A falsy value is ignored by the Router.
  export let url = "";
  let theme,
    unsubscribe,
    isMounted = false;
  let currentPage;

  let onThemeChange = () => {};

  onMount(() => {
    router.initialize();
    themeManager.initialize();
    themeManager.toggle();
    onThemeChange = themeManager.toggle;
    if (location.pathname === "/") {
      location.href = "/about";
    }
    if (!unsubscribe) {
      unsubscribe = [
        themeManager.theme.subscribe((value) => (theme = value)),
        router.currentPage.subscribe((value) => (currentPage = value)),
      ];
    }

    isMounted = true;
  });

  onDestroy(() => {
    if (unsubscribe) {
      unsubscribe.forEach((unsub) => unsub());
      themeManager.destroy();
    }
  });

  $: about = !currentPage || currentPage === "about";
  $: inBlogPost = currentPage && currentPage.split("/").length > 1;
</script>

<style lang="scss">
  .container {
    width: calc(100vw - 40px);
    min-height: 100vh;
    position: relative;
    display: flex;
    flex-direction: column;

    &.about {
      overflow: hidden;
    }
  }

  .theme-switcher-container {
    position: absolute;
    top: calc(3vw + 15px);
    right: 2vw;
    transition: all 500ms linear;
  }

  .section {
    width: 100%;
    z-index: 300;
    max-width: 110ch;
    margin: auto;
    padding: 0 20px;
    margin-top: 5rem;
  }

  .about {
    --sun-size: 120px;
    --moon-size: 80px;
  }

  :not(.about) {
    :global(.moon-container),
    :global(.sun-container) {
      opacity: 0 !important;
    }
  }

  @media (max-width: 850px) {
    .theme-switcher-container {
      top: calc(4vw + 2px);
      left: 0.5rem;
    }
    :not(.about) {
      :global(.moon-container),
      :global(.sun-container) {
        --sun-size: 13vw !important;
        --moon-size: 12vw !important;
      }
    }
    .container {
      overflow-x: hidden;
      flex-flow: column;
      padding: 0;
      width: 100%;
    }

    .section {
      margin-top: 0;
      padding: 0 0.6rem;
    }

    img {
      max-width: 100vw;
    }

    :global(html.light) {
      .container.inBlogPost {
        background: white !important;
      }
    }
  }
</style>

<svelte:head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>

<div class="container" class:about class:inBlogPost>
  <div class="theme-switcher-container">
    <ThemeSwitcher {theme} on:click={onThemeChange} />
  </div>

  <Router {url}>
    {#if theme == 'light'}
      <Sun on:click={themeManager.toggle} {about} />
    {:else}
      <Moon on:click={themeManager.toggle} {about} />
    {/if}

    <Nav {currentPage} />
    <div class="section">
      <Route path="projects" component={Projects} />
      <Route path="about" component={Me} />
      <Route path="writings/*" component={Blog} />
    </div>
  </Router>
</div>
