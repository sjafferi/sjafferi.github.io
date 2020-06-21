<style>
  .tag {
    display: inline-block;
    padding: 0.3em 0.9em;
    margin: 0 0.5em 0.5em 0;
    white-space: nowrap;
    background-color: #f1f8ff;
    border-radius: 3px;
  }

  :global(html.dark .tag) {
    background-color: #4f4f4f;
  }
</style>


<p class="tag" {...$$props}>
    <slot />
</p>
