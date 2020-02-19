<script>
  import MediaQuery from "svelte-media-query";
  import NavLink from "components/NavLink.svelte";
</script>

<style>
  .sidebar {
    display: flex;
    flex-direction: column;
    margin-top: 4vw;
  }

  @media (max-width: 550px) {
    .sidebar {
      flex-flow: row;
      margin: 0;
      margin-top: 13px;
      justify-content: space-evenly;
    }
  }
</style>

<div class="sidebar">
  <NavLink to="writings">Writings</NavLink>
  <NavLink to="projects">Projects</NavLink>
  <NavLink to="/">Me</NavLink>
</div>
