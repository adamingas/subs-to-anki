#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 <mkv-file> <language> [output-dir]" >&2
  echo "Example: $0 episode.mkv gre data/raw" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

input_file=$1
language=$2
output_dir=${3:-data/raw}

if [[ ! -f "$input_file" ]]; then
  echo "Input file not found: $input_file" >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe is required but not installed" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed" >&2
  exit 1
fi

mkdir -p "$output_dir"

base_name=$(basename "$input_file")
base_name=${base_name%.mkv}

matching_streams=()
while IFS= read -r stream; do
  matching_streams+=("$stream")
done < <(
  ffprobe -v error \
    -select_streams s \
    -show_entries stream=index,codec_name:stream_tags=language,title \
    -of csv=p=0 "$input_file" |
    awk -F',' -v lang="$language" '
      BEGIN { count = 0 }
      $2 == "subrip" && $3 == lang {
        count++
        title = $4
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", title)
        print $1 "|" count "|" title
      }
    '
)

if [[ ${#matching_streams[@]} -eq 0 ]]; then
  echo "No subrip subtitle tracks found for language '$language' in $input_file" >&2
  exit 1
fi

for stream in "${matching_streams[@]}"; do
  IFS='|' read -r stream_index ordinal title <<<"$stream"
  suffix="$language"

  if [[ -n "$title" ]]; then
    safe_title=$(printf '%s' "$title" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '.')
    safe_title=${safe_title#.}
    safe_title=${safe_title%.}
    suffix="${suffix}.${safe_title}"
  elif [[ ${#matching_streams[@]} -gt 1 ]]; then
    suffix="${suffix}.${ordinal}"
  fi

  output_file="${output_dir}/${base_name}.${suffix}.srt"
  ffmpeg -v error -y -i "$input_file" -map "0:${stream_index}" -c copy "$output_file"
  echo "Wrote $output_file"
done
