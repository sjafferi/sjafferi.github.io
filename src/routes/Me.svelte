<script>
  import { Router, Link, Route } from "svelte-routing";
  import List from "components/List.svelte";
</script>

<style>
  .container-me {
    width: 100%;
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    margin-left: 85px;
    margin-top: 65px;
  }
  @media (max-width: 550px) {
    .container-me {
      margin: 0;
      padding: 10px 7px;
    }
  }
</style>

<svelte:head>
  <title>Me | Sibtain Jafferi</title>
</svelte:head>

<div class="container-me">
  <List title="About">
    <div class="description list">
      <p>Greetings. I'm a frontend engineer @ Sumo Logic</p>
      <p>
        Right now I'm most interested in developing open source svelte components,
        <a href="/writings">writing</a>
        and
        <a href="/projects">a few notable projects</a>
        .
      </p>
    </div>
  </List>

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
