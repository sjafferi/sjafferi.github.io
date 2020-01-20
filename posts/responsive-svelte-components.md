When creating responsive web applications, my go to strategies are usually css based (flexbox, css grid, media queries). 

But what if we want to alter the content of the html based on device? In this case, some javascript must be involved to differentially render the right html.

We can make use of a higher order Svelte component to accomplish this (leveraging [named slots](https://svelte.dev/tutorial/named-slots)). 

We'll end up with something like this:

```html
<Responsive>
	<div slot="mobile">...mobile content...</div>
	<div slot="desktop">...desktop content...</div>
</Responsive>
```

`// Responsive.svelte`
```html
<script>
import { stores } from "@sapper/app";
import UAParser from "ua-parser-js";
// session is passed in server.js
const { preloading, page, session } = stores();

const parser = new UAParser();
parser.setUA($session["user-agent"]);
const mobile = parser.getResult().device["type"] == "mobile";
</script>

{#if mobile}
<slot name="mobile" />
{:else}
<slot name="desktop" />
{/if}
```

And passing the session in from `server.js`
```javascript
import sirv from 'sirv';
import polka from 'polka';
import compression from 'compression';
import * as sapper from '@sapper/server';
const { PORT, NODE_ENV } = process.env;
const dev = NODE_ENV === 'development';

polka() // You can also use Express
.use(
	compression({ threshold: 0 }),
	sirv('static', { dev }),
	sapper.middleware({
		session: (req, res) => ({
			'user-agent': req.headers['user-agent']
		})
	})
)
.listen(PORT, err => {
	if (err) console.log('error', err);
});
```

Now we can differentially load html content with a higher order Svelte component!