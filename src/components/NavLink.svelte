<script>
  import { Link } from "svelte-routing";

  export let to = "";

  function getProps({ location, href, isPartiallyCurrent, isCurrent }) {
    const isActive = href === "/" ? isCurrent : isPartiallyCurrent || isCurrent;

    // The object returned here is spread on the anchor element's attributes
    if (isActive) {
      return { class: "active" };
    }
    return {};
  }
</script>

<style>
  :global(.link a) {
    font-weight: 600;
    font-size: 18px;
    font-variant: small-caps;
    text-decoration: none !important;
    color: #3c3c3c;
  }
  @media (max-width: 900px) {
    :global(.link a) {
      font-size: 1.5em;
    }
  }
  @media (max-width: 550px) {
    :global(.link a) {
      font-size: 1em;
    }
  }
  .link {
    margin: 2px 0;
  }
</style>

<div class="link">
  <Link {to} {getProps}>
    <slot />
  </Link>
</div>
