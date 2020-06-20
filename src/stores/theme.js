import { writable } from "svelte/store";

class ThemeManager {
  constructor() {
    this.theme = "light";
  }

  toggle() {
    document.body.classList.remove(this.theme);
    this.theme = this.theme === "light" ? "dark" : "light";
    document.body.classList.add(this.theme);
  }
}

export let themeManager = new ThemeManager();
