<script>
  import { Router, Link, Route } from "svelte-routing";
  import List from "../components/List.svelte";
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

<div class="container-me">

  <List title="Code">
    <li>
      <a href="https://github.com/sjafferi" target="_blank">
        github.com/sjafferi
      </a>
    </li>
  </List>

  <List title="Experience">
    <li>
      <p>2019 - Sumo Logic: Frontend Engineer - present job</p>
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
