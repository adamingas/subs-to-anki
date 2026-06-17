import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const PORT = Number(process.env.PORT || 5173);
const CINEMETA_BASE = "https://v3-cinemeta.strem.io";
const SUBTITLE_ADDONS = [
  { name: "OpenSubtitles v3", base: "https://opensubtitles-v3.strem.io" },
  { name: "OpenSubtitles v2", base: "https://subtitlesv2.strem.io" }
];
const SUBTITLE_URL_HOSTS = new Set(["subs5.strem.io"]);

const languageNames = new Intl.DisplayNames(["en"], { type: "language" });
const languageAliases = {
  alb: "sqi",
  arm: "hye",
  baq: "eus",
  bur: "mya",
  chi: "zho",
  cze: "ces",
  dut: "nld",
  ell: "el",
  fre: "fr",
  geo: "kat",
  ger: "de",
  gre: "el",
  heb: "he",
  ice: "is",
  mac: "mk",
  may: "ms",
  mao: "mi",
  per: "fa",
  pob: "pt-BR",
  rum: "ro",
  slo: "sk",
  scc: "sr",
  scr: "hr",
  spa: "es",
  swe: "sv",
  tur: "tr"
};

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml"
};

createServer(async (req, res) => {
  try {
    const requestUrl = new URL(req.url || "/", `http://${req.headers.host}`);

    if (requestUrl.pathname.startsWith("/api/")) {
      await routeApi(requestUrl, res);
      return;
    }

    await serveStatic(requestUrl.pathname, res);
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Unexpected server error" });
  }
}).listen(PORT, () => {
  console.log(`Subtitle dashboard running at http://localhost:${PORT}`);
});

async function routeApi(url, res) {
  if (url.pathname === "/api/search" || url.pathname === "/api/suggest") {
    const query = (url.searchParams.get("q") || "").trim();
    const limit = Number(url.searchParams.get("limit") || (url.pathname === "/api/suggest" ? 8 : 30));

    if (query.length < 2) {
      sendJson(res, 200, { query, results: [] });
      return;
    }

    const results = await searchCatalog(query, limit);
    sendJson(res, 200, { query, results });
    return;
  }

  if (url.pathname === "/api/meta") {
    const type = assertType(url.searchParams.get("type"));
    const id = assertId(url.searchParams.get("id"));
    const meta = await fetchJson(`${CINEMETA_BASE}/meta/${type}/${id}.json`);
    sendJson(res, 200, { meta: normalizeMeta(meta.meta), episodes: normalizeEpisodes(meta.meta?.videos || []) });
    return;
  }

  if (url.pathname === "/api/subtitles") {
    const type = assertType(url.searchParams.get("type"));
    const videoId = assertId(url.searchParams.get("id"), true);
    const subtitles = await getSubtitles(type, videoId);
    sendJson(res, 200, { type, id: videoId, subtitles });
    return;
  }

  if (url.pathname === "/api/download") {
    await proxySubtitleDownload(url, res);
    return;
  }

  sendJson(res, 404, { error: "Unknown API route" });
}

async function searchCatalog(query, limit) {
  const encodedQuery = encodeURIComponent(query);
  const requests = ["movie", "series"].map(async (type) => {
    const data = await fetchJson(`${CINEMETA_BASE}/catalog/${type}/top/search=${encodedQuery}.json`);
    return (data.metas || []).map((meta, index) => normalizeMeta(meta, index));
  });

  const settled = await Promise.allSettled(requests);
  const results = settled
    .flatMap((entry) => (entry.status === "fulfilled" ? entry.value : []))
    .filter(Boolean);

  const seen = new Set();
  return results
    .map((item) => ({ ...item, score: item.score + titleMatchBoost(query, item.name) }))
    .filter((item) => {
      const key = `${item.type}:${item.id}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, limit);
}

async function getSubtitles(type, videoId) {
  const addonResults = await Promise.allSettled(
    SUBTITLE_ADDONS.map(async (addon) => {
      const data = await fetchJson(`${addon.base}/subtitles/${type}/${videoId}.json`);
      return (data.subtitles || []).map((subtitle) => normalizeSubtitle(subtitle, addon.name));
    })
  );

  const deduped = new Map();
  for (const result of addonResults) {
    if (result.status !== "fulfilled") continue;
    for (const subtitle of result.value) {
      if (!subtitle.url) continue;
      const key = `${subtitle.lang}:${subtitle.url}`;
      if (!deduped.has(key)) deduped.set(key, subtitle);
    }
  }

  return [...deduped.values()].sort((a, b) => {
    return a.languageName.localeCompare(b.languageName) || a.source.localeCompare(b.source) || a.id.localeCompare(b.id);
  });
}

async function proxySubtitleDownload(url, res) {
  const target = url.searchParams.get("url") || "";
  const filename = safeFilename(url.searchParams.get("filename") || "subtitle.srt");
  const parsed = new URL(target);

  if (!["https:", "http:"].includes(parsed.protocol) || !SUBTITLE_URL_HOSTS.has(parsed.hostname)) {
    sendJson(res, 400, { error: "Unsupported subtitle download host" });
    return;
  }

  const response = await fetchWithTimeout(parsed.toString(), { timeoutMs: 15000 });
  if (!response.ok) {
    sendJson(res, response.status, { error: `Subtitle download failed with HTTP ${response.status}` });
    return;
  }

  res.writeHead(200, {
    "Content-Type": response.headers.get("content-type") || "application/x-subrip; charset=utf-8",
    "Content-Disposition": `attachment; filename="${filename.endsWith(".srt") ? filename : `${filename}.srt`}"`,
    "Cache-Control": "no-store"
  });

  const buffer = Buffer.from(await response.arrayBuffer());
  res.end(buffer);
}

async function serveStatic(pathname, res) {
  const requestedPath = pathname === "/" ? "/index.html" : pathname;
  const filePath = normalize(join(__dirname, requestedPath));

  if (!filePath.startsWith(__dirname)) {
    sendJson(res, 403, { error: "Forbidden" });
    return;
  }

  try {
    const data = await readFile(filePath);
    res.writeHead(200, {
      "Content-Type": contentTypes[extname(filePath)] || "application/octet-stream",
      "Cache-Control": "no-store"
    });
    res.end(data);
  } catch {
    const data = await readFile(join(__dirname, "index.html"));
    res.writeHead(200, { "Content-Type": contentTypes[".html"], "Cache-Control": "no-store" });
    res.end(data);
  }
}

async function fetchJson(url) {
  const response = await fetchWithTimeout(url, { timeoutMs: 12000 });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}

async function fetchWithTimeout(url, { timeoutMs }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      headers: { "User-Agent": "subs2anki-dashboard/0.1" },
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeout);
  }
}

function normalizeMeta(meta, index = 0) {
  if (!meta?.id || !meta?.type || !meta?.name) return null;
  return {
    id: meta.id,
    imdbId: meta.imdb_id || meta.id,
    type: meta.type,
    name: meta.name,
    poster: meta.poster || meta.background || "",
    background: meta.background || "",
    releaseInfo: meta.releaseInfo || meta.year || "",
    description: meta.description || "",
    imdbRating: meta.imdbRating || "",
    genres: meta.genres || meta.genre || [],
    score: Number(meta.score || meta.popularity || 0) - index / 100
  };
}

function titleMatchBoost(query, title) {
  const normalizedQuery = normalizeTitle(query);
  const normalizedTitle = normalizeTitle(title);

  if (normalizedTitle === normalizedQuery) return 100;
  if (normalizedTitle.startsWith(normalizedQuery)) return 55;
  if (normalizedTitle.includes(normalizedQuery)) return 25;

  const distance = levenshtein(normalizedTitle, normalizedQuery);
  const tolerance = Math.max(2, Math.floor(Math.max(normalizedTitle.length, normalizedQuery.length) * 0.25));
  if (distance <= tolerance) return 80 - distance;

  return 0;
}

function normalizeTitle(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function levenshtein(a, b) {
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  const current = Array.from({ length: b.length + 1 }, () => 0);

  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
    previous.splice(0, previous.length, ...current);
  }

  return previous[b.length];
}

function normalizeEpisodes(videos) {
  return videos
    .map((video) => ({
      id: video.id,
      name: video.name || `Episode ${video.number || video.episode || ""}`.trim(),
      season: video.season,
      episode: video.number || video.episode,
      released: video.released || video.firstAired || "",
      thumbnail: video.thumbnail || "",
      description: video.description || video.overview || ""
    }))
    .filter((episode) => episode.id && episode.season && episode.episode)
    .sort((a, b) => a.season - b.season || a.episode - b.episode);
}

function normalizeSubtitle(subtitle, source) {
  const lang = subtitle.lang || "und";
  return {
    id: String(subtitle.id || subtitle.url || ""),
    url: subtitle.url,
    lang,
    languageName: getLanguageName(lang),
    source,
    encoding: subtitle.SubEncoding || subtitle.encoding || "",
    score: subtitle.g || "",
    matchedBy: subtitle.m || ""
  };
}

function getLanguageName(code) {
  const normalized = languageAliases[code] || code;
  try {
    return languageNames.of(normalized) || code;
  } catch {
    return code.toUpperCase();
  }
}

function assertType(type) {
  if (type !== "movie" && type !== "series") throw new Error("type must be movie or series");
  return type;
}

function assertId(id, allowEpisode = false) {
  const pattern = allowEpisode ? /^tt\d+(?::\d+:\d+)?$/ : /^tt\d+$/;
  if (!id || !pattern.test(id)) throw new Error("Invalid IMDb/Stremio id");
  return id;
}

function safeFilename(value) {
  return value.replace(/[^a-z0-9._ -]/gi, "_").slice(0, 120) || "subtitle.srt";
}

function sendJson(res, status, payload) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(JSON.stringify(payload));
}
