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
    transition: all 2s cubic-bezier(0.215, 0.610, 0.355, 1.000);
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
    transition: all 1s cubic-bezier(0.74, -0.6, 0.455, 1.65), width,height,top,left,bottom,right 2s cubic-bezier(0.74, -0.6, 0.455, 1.65);
    transition-delay: 0.4s;
    z-index: -4;
  }

  #star {
    position: absolute;
    left: 10em;
    top: 2em;
    height: 90px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 9%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(13deg) scale(0.4);
    transition: all 2s cubic-bezier(0.215, 0.610, 0.355, 1.000);
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
    left: 7em;
    top: 8em;
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(55deg) scale(0.35);
    transition: all 2s cubic-bezier(0.215, 0.610, 0.355, 1.000);
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
    left: 4em;
    top: 8em;
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(31deg) scale(0.4);
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
    transition: all 2s cubic-bezier(0.215, 0.610, 0.355, 1.000);
  }
@media(hover: hover) and (pointer: fine) {
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
  @media (max-width: 850px) {
    .moon-container {
      --moon-size: 50px;
    }
    #star,
    #star1,
    #star2 {
      display: none;
    }
    
    .about {
      #star,
      #star1 {
        display: block;
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

  @keyframes bounce-7 {
    0%   { transform: scale(1,1)      translateY(0); }
    10%  { transform: scale(1.1,.9)   translateY(0); }
    30%  { transform: scale(.9,1.1)   translateY(-50px); }
    50%  { transform: scale(1.05,.95) translateY(0); }
    57%  { transform: scale(1,1)      translateY(-7px); }
    64%  { transform: scale(1,1)      translateY(0); }
    100% { transform: scale(1,1)      translateY(0); }
  }

  .about.bounce {
    animation: bounce-7 2s 2;
    animation-delay: 3s;
    animation-timing-function: cubic-bezier(0.280, 0.840, 0.420, 1);
  }

  .info-msg {
    display: flex;
    position: absolute;
    top: 38%;
    left: 33%;
    color: #676767;
    font-size: 0.5rem;
    font-weight: 300 !important;
    text-transform: uppercase;
    font-family: "Montserrat", sans-serif;
    letter-spacing: 6px;

    span {
      color: #c5c5c5;
    }
  }

  .about {
    &.moon-container {
      z-index: 100;
    }

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

<div class="moon-container bounce" on:click class:animate class:about transition:fade>
  <div id="moon" on:mouseover="{toggleAnimation}">
    <!-- {#if showMessage}
    <span class="info-msg" transition:slide>Tap <span>me</span></span>
    {/if} -->
    <div id="star"></div>
    <div id="star1"></div>
    <div id="star2"></div>
  </div>
  <div id="moon-shadow bounce"></div>
</div>
