## Problem
If you're anything like me, you may have trouble coming up with designs for your project web pages. 

Translating a set of functional requirements into components can be a daunting process. 

There are some ways to make the ideation and design phase easier, which usually leads to a simpler implementation. 

I'll briefly go over my current process using [my personal projects page](https://sibta.in/projects) as an example.

![Projects page](https://i.imgur.com/PFerhK5.png)

## Ideation
I knew the basic information I wanted to show for each project: name, description, pictures and links. Now, I just had to come up with a visual format to display this info in.

I had trouble coming up with my own ideas so I looked to the internetz.

I usually bookmark websites that have interesting styles and search through this list for inspiration when confronting a design problem.

The Github repository view stuck out to me.

![Github repo view](https://i.imgur.com/UCIbUPs.png)

Another notable mention was [Samantha Ming's Code Tidbits page](https://www.samanthaming.com/tidbits/)

![Code tidbits](https://i.imgur.com/20bJ24s.png)

I decided I wanted to keep a minimalistic theme for my site and hence went with the repository view.

So, when looking for ideas on how to display your web page, list out all of the content that should be displayed. 

Then, accumulate a list of webpages that visually display the content in a way that you find appealing and relevant. List out your own ideas for the visual hierarchy as well.

This will give you a good starting point 

Next up is modification and design.

## Design

Now we need to modify our chosen design to fit the functional requirements. 

It helps to draw out a rough sketch of your designs and customize as necessary.

For the project's page, I knew I needed images and links, and hence found the appropriate spot for them in the designs.

Note: my artistic ability is very limited :(

![Rough sketch](https://i.imgur.com/YGSvJTb.jpg)

Drawing it out also helps in creating a hierarchical component structure.

For example this implies we'll need something like:

```html
<div class="project-list">
    <div class="project">
      <div class="content">
        ...
      </div>  
      <div class="links">
        ...
      </div>
    </div>
</div>
```

## Development

Once we have an idea of the component structure, development becomes a breeze. We're essentially just plugging data into the structure (and styling it).

This is the implementation using Svelte.

```html
<div class="projects">
  {#each Projects as { title, titleLink, description, images, tags, links }}
    <div class="tile">
      <div class="content">
        <a class="header-link" href={titleLink} target="_blank">{title}</a>
        <p class="description">{description}</p>
        {#if images && images.length > 0}
          <div class="images">
            <Images numCols={3} {images} />
          </div>
        {/if}
        <div class="tags">
          {#each tags as tag}
            <div class="tag">{tag}</div>
          {/each}
        </div>
      </div>
      <div class="links">
        {#each links as { link, text }}
          <a class="link-btn" target="_blank" href={link}>{text}</a>
        {/each}
      </div>
    </div>
  {/each}
</div>
```