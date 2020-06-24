<script>
  import { onMount } from "svelte";
  const ANIMATION_DELAY = 1000;
  let animate = true;
  const toggleAnimation = () => {
    animate = true;
    setTimeout(() => (animate = false), ANIMATION_DELAY);
  };
  onMount(() => {
    toggleAnimation();
  });
</script>
<style lang="scss">
  .moon-container {
    top: 0;
    left: 0;
    position: absolute;
    display: none;
    --moon-size: 100px;
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
    transition: all 1s cubic-bezier(0.74, 0, 0.455, 1);
    /* -webkit-transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650);
   -moz-transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650);
     -o-transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650);
        transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650);   */
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
    transition: all 1s cubic-bezier(0.74, -0.6, 0.455, 1.65);
    transition-delay: 0.4s;
    z-index: -4;
  }

  #star {
    position: absolute;
    left: 17rem;
    top: 2rem;
    height: 90px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 9%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(60deg);
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
    left: 12rem;
    top: 15rem;
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(55deg);
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
    left: 3rem;
    top: 15rem;
    height: 60px;
    width: 1px;
    background: radial-gradient(ellipse at center, #f9f9f9 29%, #1e5799 98%);
    border-radius: 50%;
    transform: rotate(31deg);
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
  }

  .animate {
    #star,
    #star1,
    #star2 {
      transform: rotate(-500deg) !important;
    }
  }

  @media (max-width: 550px) {
    .moon-container {
      --moon-size: 50px;
    }
    #star,
    #star1,
    #star2 {
      display: none;
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
</style>

<div class="moon-container" on:click class:animate>
  <div id="moon" on:mouseover="{toggleAnimation}"></div>
  <div id="moon-shadow"></div>
  <div id="star"></div>
  <div id="star1"></div>
  <div id="star2"></div>
</div>
