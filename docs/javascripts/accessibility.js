function labelSearchDialog() {
  const searchDialog = document.querySelector('.md-search[role="dialog"]')
  if (searchDialog && !searchDialog.hasAttribute("aria-label")) {
    searchDialog.setAttribute("aria-label", "Search")
  }
}

labelSearchDialog()
document.addEventListener("DOMContentLoaded", labelSearchDialog)

if (typeof document$ !== "undefined") {
  document$.subscribe(labelSearchDialog)
}
