<script>
  import { onMount } from "svelte";
  import { slide, fade } from "svelte/transition";
  import { link } from "svelte-routing";
  import { router } from "stores";
  import MediaQuery from "svelte-media-query";
  import NavLink from "components/NavLink.svelte";
  import Menu from "components/Icons/Menu.svelte";
  import Close from "components/Icons/Close.svelte";

  export let currentPage;

  let closed = true;

  const options = [
    { title: "writings", link: "/writings" },
    { title: "projects", link: "/projects" },
    { title: "about", link: "/about" },
  ];

  const close_menu = () => (closed = !closed);
  const select_menu_option = (link) => {
    router.go(link.slice(1));
    closed = true;
  };
  const is_active_link = (link) => {
    return (currentPage || location.pathname).includes(link.slice(1));
  };
</script>

<style type="text/scss">
  .navbar {
    display: flex;
    justify-content: center;
    margin-top: 4vw;
    z-index: 200;
    width: fit-content;
    align-self: center;
  }

  :global(.navbar .link) {
    min-width: 75px;
    display: flex;
    justify-content: center;
    margin: 0 25px !important;
    transition: all 25ms ease-in;
    @media (hover: hover) and (pointer: fine) {
      &:hover {
        > a {
          background: #333;
          color: white !important;
        }
      }
    }
  }

  :global(.navbar a) {
    font-size: 1.3rem !important;
    text-align: center;
    font-weight: 300 !important;
    text-transform: uppercase;
    font-family: "Montserrat", sans-serif;
    letter-spacing: 6px;
    padding: 6px 20px;
    color: var(--text-color) !important;
    transition: all 150ms ease-in-out;
  }

  :global(.navbar .link) {
    a.active {
      border-bottom: 2px solid #939393;
    }
  }

  :global(.navbar div + div) {
    border-left: none;
  }

  :global(html.dark) {
    :global(.navbar .link) {
      @media (hover: hover) and (pointer: fine) {
        &:hover {
          > a {
            background: #fff !important;
            color: black !important;
          }
        }
      }
    }

    :global(.navbar-cta svg) {
      fill: var(--text-color);
    }
  }

  @media (max-width: 850px) {
    .navbar {
      flex-flow: column;
      margin: 0;
      margin-top: 13px;
      justify-content: space-evenly;
      min-width: 175px;
    }
    :global(.navbar div + div) {
      border: 1px solid #939393;
    }
    :global(.navbar .link) {
      position: relative;
      border: 1px solid #939393 !important;
      margin: 0 !important;
      a {
        font-size: 1rem !important;
      }
      a.active {
        border-bottom: none;
      }
    }
  }

  .navbar-cta {
    position: absolute;
    right: 30px;
    top: 20px;
    width: 20px;
    height: 20px;
    z-index: 250;
  }
</style>

<MediaQuery query="(max-width: 850px)" let:matches>
  {#if matches}
    <div class="navbar" class:closed>
      {#each options as { title, link }}
        {#if !closed || is_active_link(link)}
          <div transition:slide|local>
            <NavLink to={link} on:click={() => select_menu_option(link)}>
              {title}
            </NavLink>
          </div>
        {/if}
      {/each}
    </div>
    <div class="navbar-cta">
      {#if closed}
        <div in:fade>
          <Menu on:click={close_menu} />
        </div>
      {:else}
        <div in:fade>
          <Close on:click={close_menu} />
        </div>
      {/if}
    </div>
  {:else}
    <div class="navbar" class:closed>
      {#each options as { title, link }}
        <NavLink to={link} on:click={() => select_menu_option(link)}>
          {title}
        </NavLink>
      {/each}
    </div>
  {/if}
</MediaQuery>
