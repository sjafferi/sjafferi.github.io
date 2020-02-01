<script>
  import { Router, Link, Route } from "svelte-routing";
  import { Images } from "svelte-images";
  import List from "../components/List.svelte";

  const images = [];
  for (let i = 0; i < 14; i++) {
    images.push({
      src:
        "https://cdn-images-1.medium.com/max/2632/1*FPLyijI0bj-bv8wBJ78pnA.png"
    });
  }
</script>

<style>
  .container-me {
    width: 100%;
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    padding: 0 20px;
    margin-left: 30px;
    margin-top: 4vw;
  }
</style>

<svelte:head>
  <title>Me | Sibtain Jafferi</title>
</svelte:head>

<!-- <Images {images} /> -->

<div class="container-me">

  <div class="description">
    <p>Greetings. I'm a frontend engineer @ sumo logic</p>
    <p>
      Right now I'm most interested in writing svelte open source components,
      writing and side projects (check out the projects tab).
    </p>
  </div>

  <List title="Code">
    <li>
      <a href="https://github.com/sjafferi" target="_blank">
        github.com/sjafferi
      </a>
    </li>
  </List>

  <List title="Experience">
    <li>
      <p>2019 - Sumo Logic: Frontend Engineer</p>
    </li>
    <li>
      <p>2018 - Faire: Frontend Engineer</p>
    </li>
    <li>
      <p>2018 - Shoplogix: Frontend Engineer</p>
    </li>
    <li>
      <p>2016 - OpenText: Software Developer</p>
    </li>
  </List>

  <List title="Education">
    <li>
      <p>2019 - University of Waterloo: Bachelors of Computer Science</p>
    </li>
  </List>

</div>
