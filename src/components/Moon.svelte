<script>
  import { onMount } from "svelte";
  import { slide, fade } from "svelte/transition";
  const ANIMATION_DELAY = 1000;
  const MESSAGE_DELAY = 5000;
  let animate = true,
    showMessage = true;

  export let about;

  const toggleAnimation = () => {
    animate = true;
    setTimeout(() => (animate = false), ANIMATION_DELAY);
  };

  onMount(() => {
    toggleAnimation();
    // setTimeout(() => (showMessage = false), MESSAGE_DELAY);
  });
</script>

<style lang="scss">
  .moon-container {
    top: var(--theme-changer-top);
    left: var(--theme-changer-left);
    position: absolute;
    display: none;
    --moon-size: 125px;
    transition: all 1s ease-in-out;
    z-index: 30000;
    // transition: all 2s cubic-bezier(0.215, 0.610, 0.355, 1.000);
    // transition: all 1s cubic-bezier(0.74, 0, 0.455, 1);
  }

  :global(html.dark .moon-container) {
    display: block !important;
  }

  /* Moon is down in its initial position */
  #moon {
    position: absolute;
    width: var(--moon-size);
    height: var(--moon-size);
    background: silver;
    border-radius: 50%;
    left: 40%;
    top: 130%;
    transform: scale(2.5);
    -webkit-box-shadow: inset -40px 30px 10px -20px rgba(0, 0, 0, 0.48);
    -moz-box-shadow: inset -40px 30px 10px -20px rgba(0, 0, 0, 0.48);
    box-shadow: inset -40px 30px 10px -20px rgba(0, 0, 0, 0.48),
      10px 1px 20px 0px rgba(0, 0, 0, 0.3);
    transition: all 2s cubic-bezier(0.215, 0.61, 0.355, 1);
    transition-delay: 0.4s;

    &:hover {
      cursor: pointer;
    }
  }

  /* this serves for the animation and it's behind the shadow   */
  #moon-shadow {
    position: absolute;
    width: var(--moon-size);
    height: var(--moon-size);
    background: silver;
    border-radius: 50%;
    left: 40%;
    top: 130%;
    transform: scale(2.5);
    z-index: -1;
    animation: glowing 5s infinite;
    transition: all 1s cubic-bezier(0.74, -0.6, 0.455, 1.65), width, height, top,
      left, bottom, right 2s cubic-bezier(0.74, -0.6, 0.455, 1.65);
    transition-delay: 0.4s;
    z-index: -4;
  }

  #star {
    position: absolute;
    left: 130%;
    top: 7%;
    height: 90px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 9%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(13deg) scale(0.35);
    transition: all 2s cubic-bezier(0.215, 0.61, 0.355, 1);
  }

  #star:before {
    position: absolute;
    content: "";
    height: 90px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(90deg);
  }

  #star:after {
    position: absolute;
    content: "";
    height: 90px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(-45deg);
  }

  #star1 {
    position: absolute;
    left: 115%;
    top: 72.5%;
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(55deg) scale(0.35);
    transition: all 2s cubic-bezier(0.215, 0.61, 0.355, 1);
  }

  #star1:before {
    position: absolute;
    content: "";
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(90deg);
  }

  #star1:after {
    position: absolute;
    content: "";
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(-45deg);
    z-index: -1;
  }

  #star2 {
    position: absolute;
    left: 50%;
    top: 95%;
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(31deg) scale(0.35);
  }

  #star2:before {
    position: absolute;
    content: "";
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(90deg);
  }

  #star2:after {
    position: absolute;
    content: "";
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(-45deg);
    z-index: -1;
  }

  #star,
  #star1,
  #star2 {
    transition: transform 1000ms ease-in-out;
    transition-delay: 0 !important;
    transition: all 2s cubic-bezier(0.215, 0.61, 0.355, 1);
  }

  @media (hover: hover) and (pointer: fine) {
    .animate {
      &.animate {
        #star {
          transform: scale(0.25) !important;
        }
        #star1 {
          transform: scale(0.4) !important;
        }
        #star2 {
          transform: scale(0.3) !important;
        }
      }
    }
  }

  #star1 {
    opacity: 0;
  }

  @media (min-width: 850px) {
    .about {
      #star,
      #star1,
      #star2 {
        opacity: 1;
      }
    }
  }

  // @media (max-width: 1400px) and (min-width: 850px) {
  //   #star,
  //   #star1,
  //   #star2,
  //   #moon,
  //   #moon-shadow {
  //     opacity: 0;
  //   }

  //   .about {
  //     #star,
  //     #star1,
  //     #star2,
  //     #moon,
  //     #moon-shadow {
  //       opacity: 1;
  //     }
  //   }
  // }

  :global(::-webkit-full-page-media, _:future, :root) #moon {
    box-shadow: inset -25px 21px 7px -13px rgba(0, 0, 0, 0.25);
  }

  @media (max-width: 850px) {
    .moon-container {
      --moon-size: 50px;
    }
    #star,
    #star1,
    #star2 {
      opacity: 0;
    }

    .about {
      &.moon-container {
        z-index: 40001 !important;
      }
      #star,
      #star1 {
        opacity: 1;
      }

      #star {
        left: 140%;
        top: -58%;
        transform: rotate(13deg) scale(0.3) !important;
      }

      #star1 {
        left: -38% !important;
        top: -45% !important;
        transform: rotate(-13deg) scale(0.3) !important;
      }

      #star2 {
        left: 30%;
        top: 110%;
      }
    }
  }

  @keyframes glowing {
    0% {
      box-shadow: 0 0 90px #ecf0f1;
    }
    40% {
      box-shadow: 0 0 60px #ecf0f1;
    }
    60% {
      box-shadow: 0 0 60px #ecf0f1;
    }
    100% {
      box-shadow: 0 0 90px #ecf0f1;
    }
  }

  .about {
    #star {
      left: 140%;
      top: -58%;
    }

    #star1 {
      left: -33%;
      top: 45%;
    }

    #star2 {
      left: 30%;
      top: 110%;
    }
  }
</style>

<div class="moon-container" on:click class:animate class:about transition:fade>
  <div id="moon" on:mouseover={toggleAnimation}>
    <div id="star" />
    <div id="star1" />
    <div id="star2" />
  </div>
  <div id="moon-shadow" />
</div>
