<script>
  import { onMount } from 'svelte';
  import { slide, fade } from 'svelte/transition';
  import MediaQuery from "svelte-media-query";
  import NavLink from "components/NavLink.svelte";
  import Menu from "components/Icons/Menu.svelte";
  import Close from "components/Icons/Close.svelte";

  const options = [
    {title: 'writings', link: '/writings'},
    {title: 'projects', link: '/projects'},
    {title: 'about', link: '/about'}
  ]
  let closed = true;
  let current_page;
  const close_menu = () => closed = !closed;
  const select_menu_option = (link) => {
    current_page = link;
    closed = true;
  }

  onMount(() => {
    current_page = location.pathname;
  });

</script>

<style type="text/scss">
  .navbar {
    display: flex;
    justify-content: center;
    margin-top: 4vw;
    z-index: 2;
    width: fit-content;
    align-self: center;
  }

  :global(.navbar .link) {
    min-width: 75px;
    border: 1px solid #939393;
    display: flex;
    justify-content: center;
    margin: 0 !important;
    transition: all 100ms ease-in;
    @media(hover: hover) and (pointer: fine) {
      &:hover {
        > a {
          background: #333;
          color: white !important;
        }
      }
    }
  }

  :global(.navbar a) {
    font-size: 1rem !important;
    text-align: center;
    font-weight: 300 !important;
    text-transform: uppercase;
    font-family: "Montserrat", sans-serif;
    letter-spacing: 6px;
    padding: 6px 20px;
    color: var(--text-color) !important;
  }

  :global(.navbar div + div) {
    border-left: none;
  }

  :global(html.dark) {
    :global(.navbar .link) {
      @media(hover: hover) and (pointer: fine) {
        &:hover {
          > a {
            background: #fff !important;
            color: black !important;
          }
        }
      }
    }

    :global(.navbar-cta svg) {
      fill: white;
    }
  }

  @media (max-width: 550px) {
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
    }
    :global(.navbar .link) {
      &:active {

      }
    }
  }

  .navbar-cta {
    position: absolute;
    right: 40px;
    top: 15px;
    width: 20px;
    height: 20px;
    z-index: 20;
  }  
</style>

<MediaQuery query="(max-width: 550px)" let:matches>
  {#if matches}
    <div class="navbar" class:closed>
      {#each options as { title, link }}
        {#if !closed || current_page.startsWith(link)}
        <div transition:slide|local>
            <NavLink
            to={link} 
            on:click={() => select_menu_option(link)}
            >
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
          <NavLink to={link}>{title}</NavLink>
      {/each}
    </div>
  {/if}
</MediaQuery>
