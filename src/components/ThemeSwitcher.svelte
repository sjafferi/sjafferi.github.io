<script>
  export let theme;

  $: dark_mode_enabled = theme !== "light";
</script>

<style lang="scss">
  .checkbox {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    z-index: 2;
    border: none;
    border-radius: 50px;
    z-index: 20000;
    &:hover {
      cursor: pointer;
    }
  }

  #switch {
    position: relative;
    width: 6.5rem;
    height: 2.5rem;
    border-radius: 50px;
    font-size: 1.125rem;
    font-weight: 700;
    background-color: #ffffff;
    box-shadow: 1px 2px 3px rgba(0, 0, 0, 0.5);
    /* Firefox */
    -moz-transition: all 0.5s ease-in;
    /* WebKit */
    -webkit-transition: all 0.5s ease-in;
    /* Opera */
    -o-transition: all 0.5s ease-in;
    /* Standard */
    transition: all 0.5s ease-in;
  }
  #switch.dark {
    background-color: rgba(51, 51, 51, 1);
    opacity: 0.68;
    &:hover {
      opacity: 0.8;
    }
  }
  #switch .icon {
    position: absolute;
    top: 6px;
    left: 7px;
    width: 27px;
    height: 27px;
    border-radius: 50%;
    background-color: #e6a555;
    transition: left 0.4s ease-out;
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
  }
  #switch:after {
    position: absolute;
    left: 3.25rem;
    top: 8px;
    color: rgba(51, 51, 51, 1);
    opacity: 1;
    content: "Day";
    text-align: right;
    line-height: 25px;

    /* Firefox */
    -moz-transition: all 0.7s ease-in;
    /* WebKit */
    -webkit-transition: all 0.7s ease-in;
    /* Opera */
    -o-transition: all 0.7s ease-in;
    /* Standard */
    transition: all 0.7s ease-in-out;
  }
  #switch.dark .icon {
    position: absolute;
    top: calc(50% - 12px);
    left: 4.3rem;
    width: 20px;
    height: 22px;
    border-radius: 50%;
    transform: rotate(60deg);
    background-color: rgba(255, 255, 255, 1);
  }

  #switch.dark .icon::before {
    position: absolute;
    top: 5px;
    left: -4px;
    opacity: 1;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: rgba(51, 51, 51, 1);
    content: "";
    transition-timing-function: ease-out;
    animation-name: leftTo;
    animation-duration: 0.5s;
    animation-delay: 0.2;
  }
  @keyframes leftTo {
    0% {
      left: -20px;
      opacity: 0;
    }
    50% {
      left: -15px;
      opacity: 0;
    }
    100% {
      left: -4px;
      opacity: 1;
    }
  }
  #switch.dark::before {
    position: absolute;
    left: 12px;
    top: 9px;
    color: #fff;
    content: "Night";
    text-align: left;
    line-height: 20px;
    opacity: 1;
    -moz-transition: all 0.7s ease-in; /* Firefox */
    -webkit-transition: all 0.7s ease-in; /* WebKit */
    -o-transition: all 0.7s ease-in; /* Opera */
    transition: all 0.7s ease-in; /* Standard */
    transition-delay: 0.2s;
  }
  #switch.dark::after,
  #switch::before {
    content: "";
    opacity: 0;
  }
  @media (max-width: 850px) {
    #switch {
      width: 65px;
      height: 26.5px;
    }
    #switch:after {
      left: 37px;
      top: 1px;
      color: #333333;
      opacity: 1;
      content: "Day";
      text-align: right;
      line-height: 25px;
      font-size: 0.75rem;
    }
    #switch .icon {
      top: 2px;
      left: 6px;
      width: 20px;
      height: 20px;
    }
    #switch.dark .icon {
      top: 3px;
      left: 36px;
      width: 18px;
      height: 21px;
    }
    #switch.dark::before {
      left: 7px;
      top: 3px;
      font-size: 0.75rem;
      z-index: 2;
    }
  }
</style>

<div id="switch" class:dark={dark_mode_enabled}>
  <input type="checkbox" class="checkbox" on:click />
  <div class="icon" />
</div>
