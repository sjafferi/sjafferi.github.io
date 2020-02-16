'use strict';
const fs = require('fs');

const posts = require("../src/metadata/posts.js");


const [
  _,
  _2,
  slug,
  title,
  date,
  subtitle,
  tags
] = process.argv;

const exists = posts.find(post => post.slug == slug);


if (exists) {
  throw new Error('Slug already exists', slug);
}

const post = {
  slug,
  title,
  date,
  subtitle,
  tags: tags.split(', ')
};


console.log('Saving: ', post);

posts.push(post);


fs.writeFile('./src/metadata/posts.js', `module.exports = ${JSON.stringify(posts)};`, (err) => {
  if (err) throw err;

  fs.writeFile(`./posts/${slug}.md`, `# ${title}`, (err) => {
    if (err) throw err;
    console.log('Post saved!');
  });
});
