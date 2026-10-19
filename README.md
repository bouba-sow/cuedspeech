# Cued Speech

Generate French cued speech videos (a hand is added next to the speaker's
face) and decode cued speech videos into subtitles.

## Install

Requires Python 3.11 and [pixi](https://pixi.sh) (used to install the
Montreal Forced Aligner).

```bash
mkdir cued-speech-env && cd cued-speech-env
pixi init
pixi add "python==3.11" montreal-forced-aligner=3.3.4
pixi run python -m pip install cued-speech
pixi shell
```

## Generate a cued speech video

```bash
# Text transcribed automatically with Whisper
cued-speech generate video.mp4 --output_path out.mp4

# Or give the text yourself
cued-speech generate video.mp4 --skip-whisper --text "Bonjour tout le monde" --output_path out.mp4
```

Nothing else to download: the hand images ship with the package and the
French MFA models are downloaded automatically on first use.

Useful options: `--timing-model [rules|revised|attina|base]`,
`--no-morphing`, `--no-transparency`, `--debug-landmarks`
(see `cued-speech generate --help`).

From Python:

```python
from cued_speech import generate_cue

generate_cue(
    text="Bonjour tout le monde",  # or None to transcribe with Whisper
    video_path="video.mp4",
    output_path="out.mp4",
    config={"skip_whisper": True},
)
```

## Decode a cued speech video

```bash
cued-speech download-data   # models and test videos, into ./download
cued-speech decode --video_path video.mp4
```

## Credits

Hand positions, angles and transition timing follow the models published
by [Auto-CS](https://auto-cuedspeech.org) (B. Bigi, CNRS-LPL) and
V. Attina (2005), reimplemented independently.

## License

MIT. Contact: boubasow.pro@gmail.com
