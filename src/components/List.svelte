<script>
  export let title;
</script>

<style>
  .list h3 {
    font-size: 1.5em;
    font-weight: 700;
    font-variant: small-caps;
    margin-top: 4px;
    line-height: 1.125;
  }

  :global(.list li) {
    line-height: 1.55;
  }

  :global(.list a, .list p) {
    color: var(--text-color);
  }
  :global(.list a:hover) {
    opacity: 0.75;
  }
</style>

<div class="list">
  <h3>{title}</h3>
  <ul>
    <slot />
  </ul>
</div>
