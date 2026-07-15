let diagramId = 0
let mermaidPromise

function loadMermaid() {
  mermaidPromise ??= import(
    "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.esm.min.mjs"
  ).then((module) => module.default)
  return mermaidPromise
}

function theme() {
  return document.body.getAttribute("data-md-color-scheme") === "slate"
    ? "dark"
    : "default"
}

async function renderDiagram(container) {
  try {
    const mermaid = await loadMermaid()
    const dark = theme() === "dark"
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: dark ? "dark" : "default",
      themeVariables: dark
        ? {
            edgeLabelBackground: "#343741",
            textColor: "#F8FAFC",
          }
        : undefined,
    })

    const id = `looplet-mermaid-${diagramId += 1}`
    const { svg, bindFunctions } = await mermaid.render(id, container.dataset.source)
    container.innerHTML = svg
    container.classList.remove("looplet-mermaid--error")
    bindFunctions?.(container)
  } catch (error) {
    if (!container.hasChildNodes()) {
      container.textContent = container.dataset.source
    }
    container.classList.add("looplet-mermaid--error")
    console.error("Unable to render Mermaid diagram", error)
  }
}

async function renderDiagrams() {
  const sources = document.querySelectorAll("pre.looplet-mermaid-source")
  for (const source of sources) {
    const container = document.createElement("div")
    container.className = "looplet-mermaid"
    container.dataset.source = source.textContent.trim()
    source.replaceWith(container)
    await renderDiagram(container)
  }
}

function rerenderDiagrams() {
  for (const container of document.querySelectorAll(".looplet-mermaid[data-source]")) {
    renderDiagram(container)
  }
}

renderDiagrams()

if (typeof document$ !== "undefined") {
  document$.subscribe(renderDiagrams)
} else {
  document.addEventListener("DOMContentLoaded", renderDiagrams)
}

new MutationObserver((mutations) => {
  if (mutations.some((mutation) => mutation.attributeName === "data-md-color-scheme")) {
    rerenderDiagrams()
  }
}).observe(document.body, {
  attributes: true,
  attributeFilter: ["data-md-color-scheme"],
})
