Please have a read through the various techniques disccused [here](https://developers.google.com/web/fundamentals/performance/lazy-loading-guidance/images-and-video) for a general understanding of lazy loading.


## Requirements

There are two things our lazy loading svelte component should be able to do. 

1. Defer loading of images until the image is in viewport
2. Show a tiny blurred placeholder until the image is loaded


## Intersection Observer

The intersection observer will enable use to observe a given set of elements and trigger actions on events such as on load. 

> Note: Intersection observer is not supported in all browsers. Consider this chart before usage.

![Caniuse Intersection Observer](https://i.imgur.com/5X5mug4.png)

To do this, we can implement a higher order Svelte component that uses the Intersection Observer API. 

[spiffy.tech](https://spiffy.tech/blog/a-lazy-loading-higher-order-component-for-svelte/) has a great article that goes over implementing this higher order component.

He creates a `VisibilityGaurd` component:

```html
<script>
  import { onMount } from "svelte";

  let el = null;

  let visible = false;
  let hasBeenVisible = false;

  onMount(() => {
    const observer = new IntersectionObserver(entries => {
      console.log("entry", entries[0]);
      visible = entries[0].isIntersecting;
      hasBeenVisible = hasBeenVisible || visible;
    });
    observer.observe(el);

    return () => observer.unobserve(el);
  });
</script>

<div bind:this={el}>
  <slot {visible} {hasBeenVisible} />
</div>
```

