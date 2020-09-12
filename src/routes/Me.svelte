<script>
  import { Router, Link, Route } from "svelte-routing";
  import List from "components/List.svelte";
  import Github from "components/Icons/Github.svelte";
  import LinkedIn from "components/Icons/LinkedIn.svelte";
  import Medium from "components/Icons/Medium.svelte";

  const links = [
    { Logo: Github, link: "https://github.com/sjafferi" },
    { Logo: Medium, link: "https://medium.com/@sjafferi" },
    { Logo: LinkedIn, link: "https://www.linkedin.com/in/sibtain-jafferi" },
  ];

  const credits = [
    {
      feature: "Sun & Moon",
      name: "@DeMichieli",
      link: "https://codepen.io/DeMichieli",
    },
    {
      feature: "Animated Radio Button",
      name: "@kiibriia",
      link: "https://codepen.io/kiibriia",
    },
    {
      feature: "favicon",
      name: "@freepik",
      link: "https://www.flaticon.com/authors/freepik",
    },
  ];
</script>

<style lang="scss">
  .about-container {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    font-family: "Montserrat", sans-serif;
    max-width: 48rem;
    top: calc(
      var(--theme-changer-top) - (var(--sun-size) * 2.5) * 2
    ) !important;
    left: calc(
      var(--theme-changer-left) - (var(--sun-size) * 2.5) * 2
    ) !important;
    position: absolute;
    pointer-events: none;
    min-height: 700px;
    overflow: hidden;
  }

  @mixin variable-change {
    :global(.moon-container),
    :global(.sun-container),
    :global(.about > *),
    .about-container {
      @content;
      &.about-container {
        @content;
      }
    }
  }

  @media (min-width: 1650px) {
    @include variable-change {
      --theme-changer-top: 60%;
      --theme-changer-left: 55%;
    }
  }

  @media (max-height: 1650px) {
    @media (max-height: 1250px) {
      // height <= 1250
      @include variable-change {
        --theme-changer-top: 72.5vh !important;
        --theme-changer-left: 65% !important;
        --sun-size: 13vh !important;
        --moon-size: 8.5vh !important;
      }
      &.about-container {
        top: calc(
          var(--theme-changer-top) - (var(--sun-size) * 1.8) * 2
        ) !important;
        left: calc(
          var(--theme-changer-left) - (var(--sun-size) * 1.8) * 2
        ) !important;
        .description {
          max-width: 57ch;
          .header h1 {
            font-size: 3rem !important;
          }
          .text {
            font-size: 1.75rem;
          }
        }
      }
    }

    @media (max-height: 950px) {
      // height <= 950
      @include variable-change {
        --theme-changer-top: 70vh !important;
        --theme-changer-left: 60% !important;
        &.about-container {
          top: calc(
            var(--theme-changer-top) - (var(--sun-size) * 2) * 2
          ) !important;
          left: calc(
            var(--theme-changer-left) - (var(--sun-size) * 2) * 2
          ) !important;
        }
      }
    }

    @media (max-width: 1700px) {
      // height <= 1650 and width <= 1700
      @media (max-height: 1350px) {
        // height <= 1350 and width <= 1700
        @include variable-change {
          --theme-changer-top: 65vh !important;
          --theme-changer-left: 60% !important;
          &.about-container {
            top: calc(
              var(--theme-changer-top) - (var(--sun-size) * 2.25) * 2
            ) !important;
            left: calc(
              var(--theme-changer-left) - (var(--sun-size) * 2.25) * 2
            ) !important;
          }
        }
      }
      @media (max-height: 1550px) {
        // height <= 1550 and width <= 1700
        @include variable-change {
          --theme-changer-top: 65vh !important;
          --theme-changer-left: 70% !important;
          --sun-size: 10vh !important;
          --moon-size: 8vh !important;
          &.about-container {
            top: calc(
              var(--theme-changer-top) - (var(--sun-size) * 2.5) * 2
            ) !important;
            left: calc(
              var(--theme-changer-left) - (var(--sun-size) * 3) * 2
            ) !important;
          }
        }
      }

      @media (min-height: 1000px) {
        // 1000 <= height <= 1650 and width <= 1700
        @include variable-change {
          --theme-changer-top: 65vh !important;
          --theme-changer-left: 65% !important;
          &.about-container {
            top: calc(
              var(--theme-changer-top) - (var(--sun-size) * 2) * 2
            ) !important;
            left: calc(
              var(--theme-changer-left) - (var(--sun-size) * 2) * 2
            ) !important;
          }
        }
      }
    }

    @media (min-width: 1650px) {
      // height <= 1650 and width <= 1650
      @include variable-change {
        --theme-changer-left: 60%;
      }
    }

    @media (max-width: 850px) {
      // height <= 1650 and width <= 850
      @include variable-change {
        --theme-changer-top: 18% !important;
        --theme-changer-left: calc(50% - 33px) !important;
        --sun-size: 8vh !important;
        --moon-size: 5vh !important;
      }
    }

    @media (max-width: 650px) and (max-height: 850px) {
      @include variable-change {
        --theme-changer-top: 18% !important;
        --theme-changer-left: calc(50% - 33px) !important;
        --sun-size: 8vh !important;
        --moon-size: 7vh !important;
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
        font-family: "Amiri", serif;
      }
    }
    h1,
    .text {
      line-height: 2;
      margin: 0;
      .inner-text-1,
      .inner-text-2,
      .inner-text-3 {
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
      @media (hover: hover) and (pointer: fine) {
        &:hover {
          fill: #6c6c6c;
        }
      }
    }
  }

  :global(html.dark) {
    .description {
      .text {
        color: #a4a4a4;
      }
    }
    .links {
      :global(.link svg) {
        @media (hover: hover) and (pointer: fine) {
          &:hover {
            fill: white;
          }
        }
      }
    }
    .text {
      color: #cdcdcdb5;
    }
    .inner-text-1,
    .inner-text-2,
    .inner-text-3 {
      color: white !important;
      // font-weight: normal !important;
    }
  }

  @media screen and (max-width: 850px) {
    #about-container {
      &.about-container {
        margin: 0;
        left: 50% !important;
        top: calc(var(--theme-changer-top) + 5vh + 75px) !important;
        transform: translate(-50%, 0);
        max-width: 93vw;
        width: 34ch;
      }

      @media (min-height: 320px) {
        .text {
          font-size: 1rem;
        }
      }
      div.description {
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
          .inner-text-1,
          .inner-text-2,
          .inner-text-3 {
            color: var(--text-color) !important;
            font-weight: normal !important;
          }
        }
        @media (max-width: 320px) {
          .text {
            font-size: 1.15rem;
          }
        }
      }
      .links {
        margin-top: 2rem;
        > .link {
          margin: 0 1rem;
        }
        :global(svg) {
          width: 2rem !important;
          height: 2rem !important;
        }
      }
    }

    .credits {
      display: none;
    }
  }

  .credits {
    display: flex;
    position: absolute;
    bottom: 30px;
    left: 50px;
    cursor: pointer;
    background: transparent;
    border: none !important;
    outline: none !important;
    margin: 0;
    padding: 0;
    h2 {
      color: #94949496;
      font-family: "Montserrat", sans-serif;
      font-size: 1.5rem;
    }
    .pop-up {
      position: absolute;
      padding: 10px;
      width: 300px;
      top: -35px;
      left: 105px;
      text-align: left;
      display: none;
    }
    &:focus {
      .pop-up {
        display: block;
      }
    }
  }

  :global(html.dark) {
    .credits h2 {
      color: #adadad96;
    }
  }
</style>

<svelte:head>
  <title>Me | Sibtain Jafferi</title>
</svelte:head>

<div class="about-container" id="about-container">
  <div class="description">
    <span class="header">
      <h1>I'm Sibtain</h1>
      <p class="phonetic-spelling">سبطین</p>
    </span>
    <span class="text">
      Currently at Sumo Logic in Redwood City, CA. What drives me: <span
        class="inner-text-1">curiosity</span> for <span class="inner-text-1">all
        things web</span>, creating <span class="inner-text-2">stellar user
        experiences</span>, and <span
        class="inner-text-3">self-development</span>.
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

<button class="credits">
  <h2>Credits</h2>
  <div class="pop-up">
    {#each credits as { feature, name, link }}
      <span class="credit">
        <h3>{feature} -- <a href={link}>{name}</a></h3>
      </span>
    {/each}
  </div>
</button>
