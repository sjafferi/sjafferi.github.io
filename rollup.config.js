import svelte from "rollup-plugin-svelte";
import resolve from "rollup-plugin-node-resolve";
import commonjs from "rollup-plugin-commonjs";
import css from 'rollup-plugin-css-porter';
import livereload from "rollup-plugin-livereload";
import { terser } from "rollup-plugin-terser";
import md from 'rollup-plugin-md';

const isDev = Boolean(process.env.ROLLUP_WATCH);

export default [
  // Browser bundle
  {
    input: "src/main.js",
    output: {
      sourcemap: true,
      format: "iife",
      name: "app",
      file: "public/bundle.js"
    },
    plugins: [
      svelte({
        hydratable: true,
        css: css => {
          css.write("public/bundle.css");
        }
      }),
      resolve(),
      commonjs(),
      // App.js will be built after bundle.js, so we only need to watch that.
      // By setting a small delay the Node server has a chance to restart before reloading.
      isDev &&
      livereload({
        watch: "public/App.js",
        delay: 200
      }),
      !isDev && terser(),
      md({
        marked: {
          //marked options
        }
      }),
      css({
        raw: 'custom.css',
        minified: 'custom.min.css',
      })
    ]
  },
  // Server bundle
  {
    input: "src/App.svelte",
    output: {
      sourcemap: false,
      format: "cjs",
      name: "app",
      file: "public/App.js"
    },
    plugins: [
      svelte({
        generate: "ssr",
      }),
      resolve(),
      commonjs(),
      !isDev && terser()
    ]
  }
];
