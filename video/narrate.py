"""Synthesize scene narration with edge-tts and record cue start times."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent
CUES = json.loads((ROOT / "cues.json").read_text())
AUDIO = ROOT / "audio"
VOICE = "en-US-AndrewNeural"
RATE = "+22%"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def flatten_words(words: list[dict]) -> tuple[list[str], list[float]]:
    flat_tokens: list[str] = []
    flat_times: list[float] = []
    for word in words:
        parts = tokens(word["text"])
        if not parts:
            continue
        for part in parts:
            flat_tokens.append(part)
            flat_times.append(word["offset"] / 1e7)
    return flat_tokens, flat_times


def find_start(
    flat_tokens: list[str],
    flat_times: list[float],
    cue: str,
    search_from: int,
) -> tuple[int, float] | None:
    pattern = tokens(cue)[:8]
    if not pattern:
        return None
    need = min(5, len(pattern))
    for i in range(search_from, len(flat_tokens)):
        if flat_tokens[i] != pattern[0]:
            continue
        j = i
        matched = 0
        for piece in pattern:
            limit = i + matched + 6
            while j < len(flat_tokens) and j <= limit and flat_tokens[j] != piece:
                j += 1
            if j >= len(flat_tokens) or flat_tokens[j] != piece:
                break
            matched += 1
            j += 1
        if matched >= need:
            return j, flat_times[i]
    return None


def probe_duration(path: Path) -> float:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    )
    return float(raw.strip())


async def synthesize(scene_id: str, cues: list[str]) -> None:
    text = " ".join(cues).replace("–", "-").replace("—", "-")
    last_error: Exception | None = None
    audio = bytearray()
    words: list[dict] = []
    for attempt in range(6):
        communicate = edge_tts.Communicate(
            text,
            VOICE,
            rate=RATE,
            boundary="WordBoundary",
        )
        audio = bytearray()
        words = []
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    words.append(
                        {
                            "offset": float(chunk["offset"]),
                            "text": chunk["text"],
                        }
                    )
            if audio:
                break
            last_error = RuntimeError("empty audio")
        except Exception as exc:  # noqa: BLE001 — edge-tts drops the socket intermittently
            last_error = exc
        await asyncio.sleep(2.5 * (attempt + 1))
    else:
        raise RuntimeError(f"scene {scene_id} failed after retries: {last_error}")
    if not audio:
        raise RuntimeError(f"scene {scene_id} produced no audio: {last_error}")
    AUDIO.mkdir(parents=True, exist_ok=True)
    mp3 = AUDIO / f"scene{scene_id}.mp3"
    mp3.write_bytes(audio)
    spoken = probe_duration(mp3)
    sped = AUDIO / f"scene{scene_id}.fast.mp3"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-filter:a",
            "atempo=1.25",
            str(sped),
        ]
    )
    sped.replace(mp3)
    duration = probe_duration(mp3)
    scale = duration / spoken
    flat_tokens, flat_times = flatten_words(words)
    starts: list[float] = []
    cursor = 0
    for index, cue in enumerate(cues):
        found = find_start(flat_tokens, flat_times, cue, cursor)
        if found is None:
            raise RuntimeError(f"scene {scene_id} cue {index} did not align: {cue[:80]}")
        cursor, start = found
        starts.append(round(start * scale, 3))
    if starts[0] > 0.35:
        starts[0] = 0.0
    payload = {
        "duration": duration,
        "starts": starts,
        "cues": cues,
    }
    (AUDIO / f"scene{scene_id}.json").write_text(json.dumps(payload, indent=2))
    print(
        f"scene {scene_id}: {duration:.2f}s  starts={starts}",
        flush=True,
    )


async def main() -> None:
    total = 0.0
    for scene_id in [str(i) for i in range(1, 9)]:
        done = AUDIO / f"scene{scene_id}.json"
        if done.exists() and (AUDIO / f"scene{scene_id}.mp3").exists():
            meta = json.loads(done.read_text())
            print(f"scene {scene_id}: reuse {meta['duration']:.2f}s", flush=True)
        else:
            await synthesize(scene_id, CUES[scene_id])
            meta = json.loads(done.read_text())
        total += meta["duration"]
    print(f"total narration {total:.1f}s ({total / 60:.2f} min)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
