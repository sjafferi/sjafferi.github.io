In order to avoid lengthy relative path imports like this:

```javascript
import Component  from "../../../../components/Component.svelte";
```

Gross.

`@rollup/plugin-alias` to the rescue!

Start by installing it into dev dependencies:

`yarn add -D @rollup/plugin-alias`

Next, add the plugin into your rollup config.

Note: Make sure to add it to both server and client bundles if you're using SSR in Svelte.

```javascript
// rollup.config.js
import alias from '@rollup/plugin-alias';

const aliases = alias({
  resolve: ['.svelte', '.js'], //optional, by default this will just look for .js files or folders
  entries: [
    { find: 'components', replacement: 'src/components' },
    { find: 'metadata', replacement: 'src/metadata' },
    { find: 'util', replacement: 'src/util' },
  ]
});

...

export default {
  ...
  plugins: [
    aliases
  ]
  ...
}
```

Now we can do:

```javascript
import Component from "components/Component.svelte";
```

yay!



