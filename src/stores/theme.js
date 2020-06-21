import { writable } from "svelte/store";

class ThemeManager {
  constructor() {
    this.theme = "light";
  }

  toggle() {
    const html = document.getElementsByTagName("html")[0];
    html.classList.remove(this.theme);
    this.theme = this.theme === "light" ? "dark" : "light";
    html.classList.add(this.theme);
  }
}

export let themeManager = new ThemeManager();
