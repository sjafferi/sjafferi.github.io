<script>
  import { Router, Link, Route } from "svelte-routing";
  import List from "components/List.svelte";
  import Github from 'components/Icons/Github.svelte';
  import LinkedIn from 'components/Icons/LinkedIn.svelte';
  import Medium from 'components/Icons/Medium.svelte';

  const links = [
    { Logo: Github, link: 'https://github.com/sjafferi' },
    { Logo: Medium, link: 'https://medium.com/@sjafferi' },
    { Logo: LinkedIn, link: 'https://github.com/sjafferi' }
  ]
</script>

<style lang="scss">
  .about-container {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    font-family: "Montserrat", sans-serif;
    max-width: 70ch;
    margin-left: 15%;
    margin-top: 25%;
  }

  .description {
    h1 {
      font-size: 4rem;
    }
    p {
      font-size: 2rem;
    }
    h1,
    p {
      color: #4c4c4c;
      font-weight: 400;
      line-height: 2;
      margin: 0;

      a {
        color: #000;
        font-weight: 600;
        text-decoration: none;
      }
    }
    span {
      display: flex;
      align-items: flex-end;
      .phonetic-spelling {
        font-size: 1.5rem;
        font-weight: bold;
        font-style: italic;
        line-height: 6.5rem;
        margin-left: 25px;
        color: black;
      }
    }
  }

  .links { 
    display: flex;
    margin-top: 4rem;
    .link {
      margin: 0 2rem;
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
  }

  @media (max-width: 550px) {
    .about-container {
      margin: 0;
      padding: 3rem 1.5rem;
    }
    .description {
      h1 {
        font-size: 2rem;
        font-weight: 600;
      }
      .phonetic-spelling {
        font-size: 12px !important;
        line-height: 54px !important;
      }
      .text {
        font-size: 1.25rem;
      }
    }
    .links {
      .link {
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
    <span>
      <h1>I'm Sibtain</h1>
      <p class="phonetic-spelling">سبطین</p>
    </span>
    <p class="text">
      Multi-disciplinary engineer working at Sumologic in Redwood City, CA.
      Right now I'm most interested in developing open source svelte components,
      <a href="/writings">writing</a>
      and a few notable
      <a href="/projects">projects.</a>
    </p>
  </div>

  <div class="links">
    {#each links as { Logo, link }}
      <a class="link" href={link} target="_blank">
        <Logo />
      </a>
    {/each}
  </div>
</div>
