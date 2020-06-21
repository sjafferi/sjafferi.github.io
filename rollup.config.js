import svelte from "rollup-plugin-svelte";
import resolve from "rollup-plugin-node-resolve";
import commonjs from "rollup-plugin-commonjs";
import livereload from "rollup-plugin-livereload";
import postcss from "rollup-plugin-postcss";
import aliasPlugin from "@rollup/plugin-alias";
import { terser } from "rollup-plugin-terser";
import autoPreprocess from "svelte-preprocess";
import ts from "rollup-plugin-typescript";
import typescript from "typescript";
import md from "rollup-plugin-md";

const isDev = Boolean(process.env.ROLLUP_WATCH);

const onwarn = (warning, onwarn) =>
  (warning.code === "CIRCULAR_DEPENDENCY" &&
    /[/\\]@sapper[/\\]/.test(warning.message)) ||
  onwarn(warning);
const dedupe = (importee) =>
  importee === "svelte" || importee.startsWith("svelte/");
const alias = aliasPlugin({
  resolve: [".svelte", ".js"], //optional, by default this will just look for .js files or folders
  entries: [
    { find: "stores", replacement: "src/stores/index.js" },
    { find: "components", replacement: "src/components" },
    { find: "metadata", replacement: "src/metadata" },
    { find: "util", replacement: "src/util" },
  ],
});

export default [
  // Browser bundle
  {
    input: "src/main.js",
    output: {
      sourcemap: true,
      format: "iife",
      name: "app",
      file: "public/bundle.js",
    },
    plugins: [
      svelte({
        hydratable: true,
        css: (css) => {
          css.write("public/bundle.css");
        },
        preprocess: autoPreprocess(),
      }),
      resolve({
        browser: true,
        dedupe,
      }),
      commonjs(),
      ts({
        typescript,
      }),
      // App.js will be built after bundle.js, so we only need to watch that.
      // By setting a small delay the Node server has a chance to restart before reloading.
      isDev &&
        livereload({
          watch: "public/App.js",
          delay: 200,
        }),
      !isDev && terser(),
      md({
        marked: {
          //marked options
        },
      }),
      postcss({
        plugins: [],
      }),
      alias,
    ],
  },
  // Server bundle
  {
    input: "src/App.svelte",
    output: {
      sourcemap: false,
      format: "cjs",
      name: "app",
      file: "public/App.js",
    },
    plugins: [
      svelte({
        generate: "ssr",
        preprocess: autoPreprocess(),
      }),
      resolve(),
      commonjs(),
      !isDev && terser(),
      postcss({
        plugins: [],
      }),
      alias,
    ],
  },
];
