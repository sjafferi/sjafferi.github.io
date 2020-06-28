import { writable } from "svelte/store";

class Router {
  constructor() {
    this.go = this.go.bind(this);
    this.initialize = this.initialize.bind(this);
  }

  initialize() {
    this.currentPage = writable(location.pathname.slice(1));
  }

  go(url) {
    this.currentPage.set(url);
  }
}

export let router = new Router();
