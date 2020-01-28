<script>
  import { toSlug } from "../../util.js";
  export let content;
  let numItems = 0;

  function generate(content) {
    const regex = /(#+) (.+)/g;
    const regex_2 = /#+/g;
    let match,
      prev,
      toc = [];
    while ((match = regex.exec(content)) != null) {
      const item = {
        hashes: match[1],
        title: match[2],
        children: [],
        parent: null
      };
      if (prev && prev.hashes.length < match[1].length) {
        prev.children.push(item);
        item.parent = prev;
      } else if (prev && prev.hashes.length == match[1].length && prev.parent) {
        prev.parent.children.push(item);
        item.parent = prev.parent;
      } else if (prev && prev.hashes.length > match[1].length && prev.parent) {
        let parent = prev.parent;
        while (parent && parent.hashes.length >= match[1].length) {
          parent = parent.parent;
        }
        if (parent) {
          parent.children.push(item);
          item.parent = parent;
        } else {
          toc.push(item);
        }
      } else {
        toc.push(item);
      }
      numItems++;
      prev = item;
    }

    return toc;
  }

  function html(toc) {
    return toc.length > 0
      ? `
      <ol>
        ${toc
          .map(
            ({ hashes, title, children }) =>
              `<li><a href="#${toSlug(title)}">${title.replace(
                /[1-9].?/g,
                ""
              )}</a> ${html(children)} </li>`
          )
          .join("")}
      </ol>
    `
      : "";
  }

  $: toc = html(generate(content));
</script>

<style>
  :global(.toc a) {
    color: #3c3c3c;
    text-decoration: none;
    padding-left: 3px;
  }
  :global(.toc a:hover) {
    color: #888;
  }
  :global(.toc > ul) {
    counter-reset: htoc_1;
  }
  :global(.toc ul) {
    list-style-type: none;
    padding-left: 0;
    margin-bottom: 0;
    margin-top: 4px;
    padding-left: 1.4em;
    text-indent: 0;
    padding: 0;
  }
  :global(ol) {
    counter-reset: item;
  }

  :global(ol > li) {
    counter-increment: item;
  }

  :global(.toc ol ol > li) {
    display: block;
    font-size: 0.9em;
    padding: 3px 5px;
  }

  :global(.toc ol ol > li:before) {
    content: counters(item, ".") ". ";
    margin-left: -20px;
  }
  .toc {
    float: left;
    max-width: 35ch;
    border: 1px solid #ccc;
    background-color: #f9f9f9;
    margin: 0 2rem 1.5rem 0;
    line-height: 1.25;
    padding: 5px 15px;
    position: relative;
    z-index: 1;
  }
</style>

{#if numItems > 3}
  <div class="toc">
    {@html toc}
  </div>
{/if}
