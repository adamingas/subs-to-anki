const form = document.querySelector("#search-form");
const input = document.querySelector("#search-input");
const suggestionsEl = document.querySelector("#suggestions");
const resultsEl = document.querySelector("#results");
const detailEl = document.querySelector("#detail");
const resultsHeading = document.querySelector("#results-heading");
const languageFilter = document.querySelector("#language-filter");

let selectedTitle = null;
let activeSubtitles = [];
let activeEpisodeId = null;
let suggestionTimer = null;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  hideSuggestions();
  search(input.value.trim());
});

input.addEventListener("input", () => {
  window.clearTimeout(suggestionTimer);
  suggestionTimer = window.setTimeout(() => suggest(input.value.trim()), 180);
});

document.addEventListener("click", (event) => {
  if (!form.contains(event.target)) hideSuggestions();
});

languageFilter.addEventListener("change", () => renderSubtitles());

async function suggest(query) {
  if (query.length < 2) {
    hideSuggestions();
    return;
  }

  try {
    const data = await api(`/api/suggest?q=${encodeURIComponent(query)}&limit=7`);
    if (!data.results.length) {
      hideSuggestions();
      return;
    }

    suggestionsEl.innerHTML = data.results.map(renderSuggestion).join("");
    suggestionsEl.hidden = false;
    suggestionsEl.querySelectorAll(".suggestion").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.name;
        hideSuggestions();
        search(button.dataset.name);
      });
    });
  } catch {
    hideSuggestions();
  }
}

async function search(query) {
  if (query.length < 2) return;

  selectedTitle = null;
  activeSubtitles = [];
  activeEpisodeId = null;
  detailEl.hidden = true;
  resultsEl.className = "grid empty";
  resultsEl.innerHTML = "<p>Searching the catalogue...</p>";
  resultsHeading.textContent = `Searching "${query}"`;

  try {
    const data = await api(`/api/search?q=${encodeURIComponent(query)}&limit=40`);
    resultsHeading.textContent = data.results.length ? `Matches for "${query}"` : `No matches for "${query}"`;

    if (!data.results.length) {
      resultsEl.className = "grid empty";
      resultsEl.innerHTML = "<p>No movies or series came back from Cinemeta for that title.</p>";
      return;
    }

    resultsEl.className = "grid";
    resultsEl.innerHTML = data.results.map(renderCard).join("");
    resultsEl.querySelectorAll(".card").forEach((button) => {
      button.addEventListener("click", () => openTitle(JSON.parse(button.dataset.item)));
    });
  } catch (error) {
    resultsHeading.textContent = "Search failed";
    resultsEl.className = "grid empty";
    resultsEl.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  }
}

async function openTitle(item) {
  selectedTitle = item;
  activeSubtitles = [];
  activeEpisodeId = item.type === "movie" ? item.id : null;
  updateLanguageOptions([]);
  detailEl.hidden = false;
  detailEl.innerHTML = renderDetailShell(item, "<div class='status'>Loading title details...</div>");
  detailEl.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    if (item.type === "series") {
      const data = await api(`/api/meta?type=series&id=${encodeURIComponent(item.id)}`);
      const episodes = data.episodes || [];
      const firstEpisode = episodes[0];

      detailEl.innerHTML = renderDetailShell(data.meta || item, renderEpisodePicker(episodes) + "<div class='status'>Choose an episode to load subtitles.</div>");
      const select = detailEl.querySelector("#episode-select");
      if (select && firstEpisode) {
        activeEpisodeId = firstEpisode.id;
        select.value = firstEpisode.id;
        select.addEventListener("change", () => {
          activeEpisodeId = select.value;
          loadSubtitles();
        });
        await loadSubtitles();
      }
      return;
    }

    detailEl.innerHTML = renderDetailShell(item, "<div class='status'>Loading subtitles...</div>");
    await loadSubtitles();
  } catch (error) {
    detailEl.innerHTML = renderDetailShell(item, `<div class='status'>${escapeHtml(error.message)}</div>`);
  }
}

async function loadSubtitles() {
  if (!selectedTitle || !activeEpisodeId) return;

  const list = detailEl.querySelector("#subtitle-list") || ensureSubtitleList();
  list.innerHTML = "<div class='status'>Loading subtitle files from Stremio addons...</div>";

  try {
    const data = await api(`/api/subtitles?type=${selectedTitle.type}&id=${encodeURIComponent(activeEpisodeId)}`);
    activeSubtitles = data.subtitles || [];
    updateLanguageOptions(activeSubtitles);
    renderSubtitles();
  } catch (error) {
    activeSubtitles = [];
    updateLanguageOptions([]);
    list.innerHTML = `<div class='status'>${escapeHtml(error.message)}</div>`;
  }
}

function renderSubtitles() {
  const list = detailEl.querySelector("#subtitle-list") || ensureSubtitleList();
  const language = languageFilter.value;
  const visible = language === "all" ? activeSubtitles : activeSubtitles.filter((subtitle) => subtitle.lang === language);

  if (!visible.length) {
    list.innerHTML = "<div class='status'>No subtitle files found for the selected language.</div>";
    return;
  }

  list.innerHTML = visible.map((subtitle, index) => renderSubtitleRow(subtitle, index)).join("");
  list.querySelectorAll("[data-download-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const subtitle = visible[Number(button.dataset.downloadIndex)];
      downloadSubtitle(subtitle);
    });
  });
}

function downloadSubtitle(subtitle) {
  const title = selectedTitle?.name || "subtitle";
  const episode = activeEpisodeId?.includes(":") ? activeEpisodeId.split(":").slice(1).join("x") : "";
  const filename = [title, episode, subtitle.lang, subtitle.id].filter(Boolean).join(" - ") + ".srt";
  const url = `/api/download?url=${encodeURIComponent(subtitle.url)}&filename=${encodeURIComponent(filename)}`;
  window.location.href = url;
}

function ensureSubtitleList() {
  const container = document.createElement("div");
  container.id = "subtitle-list";
  container.className = "subtitle-list";
  detailEl.append(container);
  return container;
}

function updateLanguageOptions(subtitles) {
  const current = languageFilter.value;
  const options = [...new Map(subtitles.map((subtitle) => [subtitle.lang, subtitle.languageName])).entries()]
    .sort((a, b) => a[1].localeCompare(b[1]));

  languageFilter.innerHTML = `<option value="all">All languages</option>${options
    .map(([code, name]) => `<option value="${escapeHtml(code)}">${escapeHtml(name)} (${escapeHtml(code)})</option>`)
    .join("")}`;

  languageFilter.value = options.some(([code]) => code === current) ? current : "all";
}

function renderSuggestion(item) {
  return `
    <button class="suggestion" type="button" data-name="${escapeHtml(item.name)}">
      ${posterImg(item.poster, item.name)}
      <span>
        <strong>${escapeHtml(item.name)}</strong><br>
        <small>${escapeHtml(typeLabel(item.type))} ${escapeHtml(item.releaseInfo || "")}</small>
      </span>
    </button>
  `;
}

function renderCard(item) {
  return `
    <button class="card" type="button" data-item="${escapeHtml(JSON.stringify(item))}">
      ${posterImg(item.poster, item.name, "poster")}
      <div class="card-body">
        <span class="pill">${escapeHtml(typeLabel(item.type))}</span>
        <h3>${escapeHtml(item.name)}</h3>
        <p class="meta-line">${escapeHtml(item.releaseInfo || "Unknown year")}</p>
      </div>
    </button>
  `;
}

function renderDetailShell(item, body) {
  return `
    <div class="detail-header">
      ${posterImg(item.poster, item.name)}
      <div class="detail-copy">
        <span class="pill">${escapeHtml(typeLabel(item.type))}</span>
        <h2>${escapeHtml(item.name)}</h2>
        <p class="meta-line">${escapeHtml([item.releaseInfo, item.imdbRating ? `IMDb ${item.imdbRating}` : ""].filter(Boolean).join(" · "))}</p>
        ${item.description ? `<p>${escapeHtml(item.description)}</p>` : ""}
        ${body.startsWith("<label") ? body : ""}
      </div>
    </div>
    ${body.startsWith("<label") ? '<div id="subtitle-list" class="subtitle-list"></div>' : `<div id="subtitle-list" class="subtitle-list">${body}</div>`}
  `;
}

function renderEpisodePicker(episodes) {
  if (!episodes.length) return "<div class='status'>No episode list came back for this series.</div>";

  return `
    <label class="episode-picker" for="episode-select">
      Episode
      <select id="episode-select">
        ${episodes
          .map((episode) => {
            const label = `S${String(episode.season).padStart(2, "0")}E${String(episode.episode).padStart(2, "0")} - ${episode.name}`;
            return `<option value="${escapeHtml(episode.id)}">${escapeHtml(label)}</option>`;
          })
          .join("")}
      </select>
    </label>
  `;
}

function renderSubtitleRow(subtitle, index) {
  const metadata = [
    subtitle.source,
    subtitle.encoding ? `Encoding ${subtitle.encoding}` : "",
    subtitle.score ? `Score ${subtitle.score}` : ""
  ].filter(Boolean);

  return `
    <div class="subtitle-row">
      <div>
        <strong>${escapeHtml(subtitle.languageName)} (${escapeHtml(subtitle.lang)})</strong>
        <span>Subtitle ID ${escapeHtml(subtitle.id)}</span>
      </div>
      <span>${escapeHtml(metadata.join(" · "))}</span>
      <button type="button" data-download-index="${index}">Download</button>
    </div>
  `;
}

function posterImg(src, alt, className = "") {
  if (!src) return `<div class="${className} poster" role="img" aria-label="${escapeHtml(alt)}"></div>`;
  return `<img class="${className}" src="${escapeHtml(src)}" alt="${escapeHtml(alt)} poster" loading="lazy">`;
}

function hideSuggestions() {
  suggestionsEl.hidden = true;
  suggestionsEl.innerHTML = "";
}

function typeLabel(type) {
  return type === "series" ? "Series" : "Movie";
}

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
