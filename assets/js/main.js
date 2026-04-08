(() => {
  const body = document.body;
  const wrapper = document.querySelector(".wrapper");
  const themeToggle = document.getElementById("mode");
  const menuButton = document.getElementById("menu-trigger");
  const menuPanel = document.getElementById("menu-panel");

  const getDefaultTheme = () => body.getAttribute("data-theme") || "light";

  const applyTheme = (theme) => {
    const activeTheme = theme === "dark" ? "dark" : "light";

    if (activeTheme === "dark") {
      body.setAttribute("data-theme", "dark");
    } else {
      body.removeAttribute("data-theme");
    }

    localStorage.setItem("theme", activeTheme);

    if (themeToggle) {
      const isDark = activeTheme === "dark";
      themeToggle.setAttribute("aria-pressed", String(isDark));
      themeToggle.setAttribute(
        "aria-label",
        isDark ? "Switch to light theme" : "Switch to dark theme"
      );
    }
  };

  const storedTheme = localStorage.getItem("theme") || getDefaultTheme();
  applyTheme(storedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const nextTheme =
        (localStorage.getItem("theme") || getDefaultTheme()) === "dark"
          ? "light"
          : "dark";
      applyTheme(nextTheme);
    });
  }

  const setMenuState = (isOpen) => {
    if (!menuButton || !menuPanel) return;

    menuPanel.classList.toggle("is-open", isOpen);
    menuButton.setAttribute("aria-expanded", String(isOpen));
    body.classList.toggle("menu-open", isOpen);

    if (wrapper) {
      wrapper.classList.toggle("blurry", isOpen);
    }
  };

  if (menuButton && menuPanel) {
    menuButton.addEventListener("click", () => {
      setMenuState(!menuPanel.classList.contains("is-open"));
    });

    menuPanel.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        if (window.matchMedia("(max-width: 768px)").matches) {
          setMenuState(false);
        }
      });
    });

    window.addEventListener("resize", () => {
      if (!window.matchMedia("(max-width: 768px)").matches) {
        setMenuState(false);
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        setMenuState(false);
      }
    });
  }

  const toc = document.getElementById("post-toc");
  const articleBody = document.querySelector('[itemprop="articleBody"]');

  if (toc && articleBody) {
    const headings = Array.from(articleBody.querySelectorAll("h2, h3")).filter(
      (heading) => heading.id
    );

    if (headings.length >= 3) {
      const title = document.createElement("p");
      title.className = "post-toc-title";
      title.textContent = "On this page";

      const list = document.createElement("ol");

      headings.forEach((heading) => {
        const item = document.createElement("li");
        item.className = `post-toc-depth-${heading.tagName.toLowerCase()}`;

        const link = document.createElement("a");
        link.href = `#${heading.id}`;
        link.textContent = heading.textContent.replace(/^#/, "").trim();

        item.appendChild(link);
        list.appendChild(item);
      });

      toc.appendChild(title);
      toc.appendChild(list);
      toc.hidden = false;
    }
  }
})();
