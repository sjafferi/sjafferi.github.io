<script>
  import NavLink from "../components/NavLink.svelte";
</script>

<style>
  .sidebar {
    display: flex;
    flex-direction: column;
  }
</style>

<div class="sidebar">
  <NavLink to="writings">Writings</NavLink>
  <NavLink to="projects">Projects</NavLink>
  <NavLink to="/">Me</NavLink>
</div>
