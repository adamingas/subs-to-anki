# Subs2Anki Dashboard

A small single-page subtitle finder served by a dependency-free Node server.

## Run

```bash
cd frontend
npm start
```

Open `http://localhost:5173`.

## APIs used

- Catalogue search: Stremio Cinemeta at `https://v3-cinemeta.strem.io`
- Subtitles: public Stremio OpenSubtitles addons at `https://opensubtitles-v3.strem.io` and `https://subtitlesv2.strem.io`

The server proxies catalogue, subtitle, and download requests so the browser does not need API keys or CORS access.

## Related open-source projects found

- Stremio addon SDK: `https://github.com/Stremio/stremio-addon-sdk`
- Stremio Community Subtitles: `https://github.com/skoruppa/stremio-community-subtitles`
- SubMaker: `https://github.com/xtremexq/StremioSubMaker`
