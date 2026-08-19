document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector(".copy-bibtex-btn");
  const code = document.getElementById("bibtex-code");

  if (!button || !code) return;

  button.addEventListener("click", async () => {
    const label = button.querySelector(".copy-text");
    const text = code.textContent.trim();

    try {
      await navigator.clipboard.writeText(text);
    } catch (_error) {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    button.classList.add("copied");
    label.textContent = "Copied";
    window.setTimeout(() => {
      button.classList.remove("copied");
      label.textContent = "Copy";
    }, 1800);
  });
});
