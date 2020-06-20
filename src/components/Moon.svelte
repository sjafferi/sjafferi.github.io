<style>
  .container {
    top: 0;
    left: 0;
    position: absolute;
    z-index: 2;
    pointer-events: none;
  }

  /* Moon is down in its initial position */
  #moon {
    position: absolute;
    height: 100px;
    width: 100px;
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
    z-index: -2;
  }

  /* this serves for the animation and it's behind the shadow   */
  #moon-shadow {
    position: absolute;
    height: 100px;
    width: 100px;
    background: silver;
    border-radius: 50%;
    left: 40%;
    top: 130%;
    transform: scale(2.5);
    z-index: -1;
    animation: glowing 5s infinite;
    /*    transition: all 1s cubic-bezier(0.740, 0, 0.455, 1);   */
    /* -webkit-transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650); 
   -moz-transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650); 
     -o-transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650);  
        transition: all 1s cubic-bezier(0.740, -0.600, 0.455, 1.650);   */
    transition-delay: 0.4s;
    z-index: -4;
  }
</style>

<div class="container">
  <div id="moon"></div>
  <div id="moon-shadow"></div>
</div>
