const filter = document.querySelector("#filter");
if (filter) {
  const cards = Array.from(document.querySelectorAll(".episode-card"));
  filter.addEventListener("input", () => {
    const needle = filter.value.trim().toLowerCase();
    for (const card of cards) {
      const text = card.textContent.toLowerCase();
      card.hidden = needle.length > 0 && !text.includes(needle);
    }
  });
}

