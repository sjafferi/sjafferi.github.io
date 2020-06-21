import { writable } from "svelte/store";

class ThemeManager {
  constructor() {
    this.theme = writable("light");
    this.toggle = this.toggle.bind(this);
    this.handleThemeChange = this.handleThemeChange.bind(this);
    this.destroy = this.theme.subscribe(this.handleThemeChange);
  }

  get html() {
    if (!this.htmlElem)
      this.htmlElem = document.getElementsByTagName("html")[0];
    return this.htmlElem;
  }

  handleThemeChange(theme) {
    const html = this.html;
    const oldTheme = html.classList[0];
    html.classList.remove(oldTheme);
    html.classList.add(theme);
  }

  toggle() {
    this.theme.update((theme) => (theme === "light" ? "dark" : "light"));
  }
}

export let themeManager = new ThemeManager();
