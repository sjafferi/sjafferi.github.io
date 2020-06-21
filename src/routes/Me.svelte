<script>
  import { Router, Link, Route } from "svelte-routing";
  import List from "components/List.svelte";
</script>

<style lang="scss">
  .container-me {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    font-family: "Montserrat", sans-serif;
    max-width: 70ch;
    margin-left: 15%;
    margin-top: 25%;
  }

  .description {
    h1 {
      font-size: 4rem;
    }
    p {
      font-size: 2rem;
    }
    h1,
    p {
      color: #4c4c4c;
      font-weight: 400;
      line-height: 2;
    }
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
  <div class="description">
    <h1>I'm Sibtain</h1>
    <p>
      Multi-disciplinary designer working with brands in the D.C. metro area.
      Previously at The Washington Post and National Geographic.
      <a href="/writings">writing</a>
      and
      <a href="/projects">a few notable projects</a>
      .
    </p>
  </div>

  <!-- <List title="I'm Sibtain">
    <div class="description list">
      <p>Greetings. I'm a frontend engineer @ Sumo Logic</p>
      <p>
        Right now I'm most interested in developing open source svelte
        components,
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
  </List> -->
</div>
