# Explainer video

Eight scenes, about five minutes, on the Lab II bone-marrow survival analysis.
Narration is `en-US-AndrewNeural`. Pictures are Manim Community Edition.
Nothing in this folder changes the analysis code.

## Install

Homebrew packages. `pkgconf` is required so `pycairo` can find Cairo.
`cairo` and `pango` are required by Manim; install them if they are not already present.
`ffmpeg` muxes the scenes.

```bash
brew install ffmpeg pkgconf cairo pango
```

Python 3.13 virtualenv, kept separate from the project `.venv`:

```bash
python3.13 -m venv video/.venv
source video/.venv/bin/activate
pip install -U pip
pip install manim edge-tts
```

The versions used to render this cut are Manim Community 0.22.1 and edge-tts 7.3.1.

## Regenerate the plot data

`video/chart_data.json` is already written. Rebuild it only if the fitted curves should be refreshed.
This uses the project environment, which has `lifelines` and `scikit-learn`, and it only writes `video/chart_data.json`.

```bash
.venv/bin/python video/extract_chart_data.py
```

## Narration

From the repository root, with `video/.venv` activated:

```bash
python video/narrate.py
```

That reads `video/cues.json`, writes `video/audio/scene1.mp3` through `scene8.mp3`, and writes a matching `sceneN.json` with the cue start times.
The voice is `en-US-AndrewNeural` at `+22%`, then ffmpeg `atempo=1.25`, so the eight scenes land just under five minutes.
`video/scenes.py` reads those JSON files and holds each scene until the audio ends.

## Render

Still from the repository root:

```bash
manim -qh --media_dir video/media video/scenes.py Scene1 Scene2 Scene3 Scene4 Scene5 Scene6 Scene7 Scene8
```

`-qh` is 1920x1080 at 60 fps. Silent scene files land in:

```text
video/media/videos/scenes/1080p60/Scene1.mp4
...
video/media/videos/scenes/1080p60/Scene8.mp4
```

## Mux and join

```bash
mkdir -p video/build
for i in 1 2 3 4 5 6 7 8; do
  ffmpeg -y \
    -i video/media/videos/scenes/1080p60/Scene${i}.mp4 \
    -i video/audio/scene${i}.mp3 \
    -map 0:v:0 -map 1:a:0 \
    -c:v copy -c:a aac -b:a 192k \
    -shortest \
    video/build/scene${i}.mp4
done

printf "file '%s'\n" "$PWD"/video/build/scene{1,2,3,4,5,6,7,8}.mp4 > video/build/list.txt

ffmpeg -y -f concat -safe 0 -i video/build/list.txt -c copy -movflags +faststart video/explainer.mp4
```

`video/explainer.mp4` is the finished video.
