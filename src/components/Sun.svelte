<script>
  import { slide, fade } from "svelte/transition";
  export let about = true;
  let hovering = false;
</script>

<style type="text/scss">
  :root {
    --sun-size: 150px;
    --mobile-sun-size: 50px;
  }

  .sun-container {
    top: var(--theme-changer-top);
    left: var(--theme-changer-left);
    position: absolute;
    transition: all 2s cubic-bezier(0.215, 0.61, 0.355, 1);
    z-index: 30000 !important;
  }

  :global(html.dark .sun-container) {
    display: none;
  }

  .highlight-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    /* background: rgba(0, 0, 0, 0.31); */
    background: #fc0101;
    background: -webkit-linear-gradient(bottom right, #fc0101, #ffd6d6);
    background: -moz-linear-gradient(bottom right, #fc0101, #ffd6d6);
    background: linear-gradient(to top left, #fc0101, #ffd6d6);
    z-index: 0;
  }

  :global(::-webkit-full-page-media, _:future, :root) #sun {
    &::after {
      background: radial-gradient(
          yellow,
          orange 27%,
          transparent calc(27% + 3px) 100%
        ),
        repeating-radial-gradient(
          gold 34.87%,
          rgba(255, 0, 0, 0) 67.65%,
          rgba(251, 198, 170, 0) 81.93%
        ),
        repeating-conic-gradient(
          from 0deg,
          gold 0deg 5deg,
          transparent 5deg 7deg
        );
    }
  }

  #sun {
    position: absolute;
    left: 1%;
    top: 1%;
    height: var(--sun-size);
    width: var(--sun-size);
    transform: scale(2);
    border-radius: 50%;
    background: #ffe789;
    background: radial-gradient(
      circle farthest-corner at center center,
      #ffe789 69%,
      #ffb70f 100%
    );
    // clip-path: ellipse(229% 209% at 109% 119%);
    transition: background 1s cubic-bezier(0.74, 0, 0.455, 1);
    /* transition-delay: 0.4s; */
    opacity: 0;
    z-index: 1;

    &::after,
    &::before {
      content: "";
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      opacity: 0;
      transition: opacity 2s ease-in-out;
      transform-origin: center;
      border-radius: 50%;
      mask-image: radial-gradient(rgba(0, 0, 0, 1) 40%, transparent 65%);
    }

    &::before {
      background: repeating-conic-gradient(
        from 0deg,
        yellow 0deg 20deg,
        transparent 20deg 40deg
      );
      animation: rotate 720s linear infinite, scale 2s linear;
    }

    &::after {
      background: yellow;
      background: radial-gradient(
          yellow,
          orange 27%,
          transparent calc(27% + 3px) 100%
        ),
        repeating-radial-gradient(
          gold 34.87%,
          rgba(255, 0, 0, 0) 67.65%,
          rgb(255, 215, 19) 81.93%
        ),
        repeating-conic-gradient(from 0deg, gold 0deg 5deg, #ff0c0c5c 5deg 7deg);
      transform: rotate(15deg);
      animation: rotate 360s linear, scale 2s linear;
      width: 200%;
      height: 200%;
      top: -50%;
      left: -50%;
    }

    &.animate {
      animation: rays 2s;
      opacity: 0.6;
      &::before {
        opacity: 0.5;
      }
      &::after {
        opacity: 0.55;
      }
      &:hover {
        opacity: 0.75;
        animation: rays 1.5s infinite;
        cursor: pointer;

        .outreaching-rays {
          &::before {
            content: "";
            position: absolute;
            top: -25%;
            left: -25%;
            width: 150%;
            height: 150%;
            border-radius: 15%;
            background: repeating-conic-gradient(
              from 0deg,
              white 0deg 3deg,
              transparent 5deg 10deg
            );
          }
        }
      }
    }

    .outreaching-rays {
      position: absolute;
      top: -70%;
      left: -70%;
      width: 250%;
      height: 250%;
      border-radius: 15%;
      background: repeating-conic-gradient(
        from 0deg,
        gold 0deg 5deg,
        transparent 5deg 10deg
      );
      z-index: -2;
      opacity: 0.35;
      animation: scale 2s linear;
      overflow: hidden;
      /* &::before {
            content: "";
            position: absolute;
            top: -25%;
            left: -25%;
            width: 150%;
            height: 150%;
            border-radius: 15%;
            background: repeating-conic-gradient(from 0deg, white 0deg 4deg, transparent 5deg 10deg);
          } */
    }
  }

  .overlay {
    top: 0;
    left: 0;
    position: fixed;
    width: 100vw;
    height: 100vh;
    background: #ffeee3;
    opacity: 0.25;
    pointer-events: none;
    z-index: 30001;
  }

  /* ANIMATIONS */

  @keyframes rays {
    0% {
      box-shadow: 0 0 0 0px rgba(255, 26, 0, 0.2);
    }
    100% {
      box-shadow: 0 0 0 125px rgba(0, 0, 0, 0);
    }
  }

  @keyframes rotate {
    100% {
      transform: rotate(360deg);
    }
  }

  @keyframes scale {
    0% {
      transform: scale(0.5);
    }
    50% {
      transform: scale(0.8);
    }
    100% {
      transform: scale(1);
    }
  }

  @supports (-moz-appearance: none) {
    .sun-container {
      display: none;
    }
  }

  @media (max-width: 850px) {
    .about {
      &.sun-container {
        z-index: 40001 !important;
      }
    }
  }
</style>

<!-- {#if hovering}
  <div class="highlight-overlay" />
{/if} -->

<div class="overlay" />
<div class="sun-container" on:click class:about transition:fade>
  <div
    id="sun"
    class="animate"
    on:mouseover={() => {
      hovering = true;
    }}
    on:mouseout={() => {
      hovering = false;
    }}>
    <div class="outreaching-rays" />
  </div>
</div>
