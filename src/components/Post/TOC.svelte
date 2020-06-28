<script>
  import { toSlug } from "util/index.js";
  export let content;
  let numItems = 0;
  let maxDepth = 0;
  const codeRegex = /(\`\`\`python(.|\n)*?\`\`\`)/gm;

  function generate(md) {
    if (!md) return [];
    const regex = /(#+)\s(.+)/g;
    const content = md.replace(codeRegex, "");
    let match,
      prev,
      toc = [];
    while ((match = regex.exec(content)) != null) {
      const item = {
        hashes: match[1].length,
        title: match[2],
        children: [],
        parent: null,
        depth: 0,
      };

      if (prev) {
        let parent = prev;
        while (parent && parent.hashes >= item.hashes) {
          parent = parent.parent;
        }
        if (parent) {
          parent.children.push(item);
          item.parent = parent;
          item.depth = parent.depth + 1;
          maxDepth = Math.max(maxDepth, item.depth);
        } else {
          toc.push(item);
        }
      } else {
        toc.push(item);
      }

      numItems += 1;
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

<style lang="scss">
  @mixin list-indices {
    .toc {
      :global(ol > li:before), :global(ol ol > li:before) {
        @content;
      }
    }
  }
  :global(.toc a) {
    color: var(--text-color);
    text-decoration: none;
    padding-left: 3px;
  }
  :global(.toc a:hover) {
    opacity: 0.75;
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
  :global(.toc ol) {
    counter-reset: item;
    list-style-type: none;
  }

  :global(.toc ol > li) {
    counter-increment: item;
  }

  :global(.toc > ol > li > a) {
    color: var(--text-color-1);
  }

  :global(.toc ol ol > li) {
    display: block;
    padding: 3px 5px;
    font-size: 0.9em !important;
  }
  @include list-indices {
    content: counters(item, ".") ". ";
    margin-left: -20px;
  }
  .toc {
    float: left;
    max-width: 40ch;
    border: 1px solid #ccc;
    background-color: #f9f9f9;
    margin: 0 2.25rem 1.75rem 0;
    line-height: 1.25;
    padding: 5px 15px;
    position: relative;
    z-index: 1;
  }
  :global(html.dark) {
    .toc {
      background-color: #2424243b;
      border-color: #686868;
    }
    @include list-indices {
      color: #8e8e8e;
    }
  }

  @media (max-width: 850px) {
    .toc {
      max-width: 85vw;
      margin: auto;
    }
  }
</style>

{#if numItems > 3}
<div class="toc" class:wide="{maxDepth > 3}">{@html toc}</div>
{/if}
