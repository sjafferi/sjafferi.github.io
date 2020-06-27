<script>
  import { Router, Link, Route } from "svelte-routing";
  import List from "components/List.svelte";
  import Github from 'components/Icons/Github.svelte';
  import LinkedIn from 'components/Icons/LinkedIn.svelte';
  import Medium from 'components/Icons/Medium.svelte';

  const links = [
    { Logo: Github, link: 'https://github.com/sjafferi' },
    { Logo: Medium, link: 'https://medium.com/@sjafferi' },
    { Logo: LinkedIn, link: 'https://www.linkedin.com/in/mohammad-sibtain-jafferi-a6ab79a2/' }
  ]
</script>

<style lang="scss">
  .about-container {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    font-family: "Montserrat", sans-serif;
    max-width: 48rem;
    left: calc(var(--theme-changer-left) - 525px);
    top: calc(var(--theme-changer-top) - 525px);
    position: absolute;
    pointer-events: none;
    min-height: 700px;
  }

  @media (max-height: 1575px) {
    :global(.about > *) {
      --theme-changer-top: 60%;
    }

    @media (max-height: 1090px) {
      :global(.about > *) {
        --theme-changer-top: 65%;
      }
    }

    @media (max-width: 800px) {
      :global(.about > *) {
        --theme-changer-top: 18%;
        --theme-changer-left: calc(50% - 33px);
      }
    }
  }

  .description {
    .header {
      display: flex;
      align-items: flex-end;
      .phonetic-spelling {
        font-size: 2.5rem;
        font-weight: bold;
        font-style: italic;
        line-height: 1.5rem;
        margin-left: 25px;
        color: var(--text-color);
        font-family: 'Amiri', serif;
      }
    }
    h1,
    .text {
      line-height: 2;
      margin: 0;
      .inner-text-1, .inner-text-2, .inner-text-3 {
        color: black;
        // font-weight: bold;
      }
    }
    .text {
      font-size: 2rem;
      font-weight: 400;
      color: #797979;
    }
    h1 {
      font-size: 3.5rem !important;
      font-weight: bold;
      color: var(--text-color);
    }
  }

  .links {
    display: flex;
    margin-top: 4rem;
    pointer-events: all;
    width: fit-content;
    > .link {
      margin: 0 2rem;
    }
    > .link:first-child {
      margin-left: 0;
    }
    :global(.link svg) {
      width: 3rem;
      height: 3rem;
      fill: var(--text-color);
      @media(hover: hover) and (pointer: fine) {
        &:hover {
          fill: #6c6c6c;
        }
      }
    } 
  }

  :global(html.dark) {
    .links {
      :global(.link svg) {
        @media(hover: hover) and (pointer: fine) {
          &:hover {
            fill: white;
          }
        }
      }
    }
    .text {
      color: #cdcdcdb5;
    }
    .inner-text-1, .inner-text-2, .inner-text-3 {
      color: white !important;
      // font-weight: normal !important;
    }
  }

  @media (max-width: 800px) {
    .about-container {
      margin: 0;
      padding: 3rem 0.5rem;
      top: calc(50% - 15ch);
      left: calc(50% - 17ch);
      max-width: 35ch;
    }
    .description {
      h1 {
        font-size: 2.25rem !important;
        font-weight: 600;
      }
      .phonetic-spelling {
        font-size: 2rem !important;
        line-height: 10px !important;
      }
      .text {
        font-size: 1.25rem;
        color: var(--text-color) !important;
        .inner-text-1, .inner-text-2, .inner-text-3 {
          color: var(--text-color) !important;
          font-weight: normal !important;
        }
      }
    }
    .links {
      > .link {
        margin: 0 1rem;
      }
      :global(svg) {
        width: 2rem !important;
        height: 2rem !important;
      }
    }
  }
</style>

<svelte:head>
  <title>Me | Sibtain Jafferi</title>
</svelte:head>

<div class="about-container">
  <div class="description">
    <span class="header">
      <h1>I'm Sibtain</h1>
      <p class="phonetic-spelling">سبطین</p>
    </span>
    <span class="text">
      Currently at Sumo Logic in Redwood City, CA. 
      What drives me: 
      <span class="inner-text-1">curiosity</span> for <span class="inner-text-1">all things web</span>,
      creating <span class="inner-text-2">stellar user experiences</span>,
      and <span class="inner-text-3">self-development</span>.
    </span>
  </div>

  <div class="links">
    {#each links as { Logo, link }}
      <a class="link" href={link} target="_blank">
        <Logo />
      </a>
    {/each}
  </div>
</div>
