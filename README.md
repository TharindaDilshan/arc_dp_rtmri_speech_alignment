# rtmri-align: combine rtMRI videos with audio

This tool is independent of the reconstruction package (`rtmri_recon`). After reconstruction, the in-scanner audio recordings are synchronised with the silent videos and combined
into final MP4 files. Point the tool at the **root folder** that holds all the data folders:

```
python -m rtmri_align C:\rtMRI\data --subject 275 --date 2026-10-07
```

Expected layout (`wav`, `avi` and `mp4` are created if they do not exist):

```
<root>/
  In-scanner-recordings/   DSP_OUT*.wav and TTL*.wav (raw recordings, mono)
  unaligned/               silent videos from the reconstruction step (meas_*.mp4)
  wav/                     (output) recordings cropped to the scan start
  avi/                     (output) uncompressed AVI with video + audio
  mp4/                     (output) final lossless H.264 MP4 with audio
```

The reconstruction tool writes straight into `unaligned/` when run with `-o <root>\unaligned`.

What it does, in order:

1. **Rename** `meas_*.mp4` in `unaligned/` to `<subject>_<date>_task<T>_<rep>.mp4`. Task numbers come from the
   file name: `Intervocalic_Consonants` = task 2, `Vowels_bVt` = 3, `Vowels_dVn` = 4, `Polysyl_Words` = 5,
   `Con_Speech` = 6 (a single run, so repetition 1). The repetition is the `Run_<n>` in the name.
2. **Assign recordings to tasks** from the `DSP_OUT*.wav` files: the two largest are task 1, the next five
   largest are task 2, the last file (about 2 MB) is task 6, and exactly 15 files between task 2 and task 6
   are tasks 3-5 (five repetitions each). If these assumptions fail, it stops with a message (for example
   when the number of files in between is not 15: remove failed runs first).
3. **Crop at the TTL pulse.** For each task 2-6 recording, the matching `TTL*.wav` is searched for its first
   sample with absolute value above 0.007, and the audio is cropped to start 1800 samples after it. Cropped
   files are written to `wav/` as `<subject>_<date>_task<T>_<rep>.wav` (16-bit PCM).
4. **Combine.** Each video's audio is resampled to 24996 Hz, trimmed or zero-padded to the video duration,
   then split into one chunk of 1250 samples per video frame and written with the frames into
   `avi/<name>.avi`. The video frame count and rate are taken from the video file.
5. **Convert** every AVI in `avi/` to `mp4/<name>.mp4` with
   `ffmpeg -i <avi> -c:v libx264 -preset ultrafast -crf 0 -c:a copy <mp4>` (lossless).

Options: `--subject` and `--date` are required (they are used in the output names); `--ffmpeg PATH` selects an
ffmpeg executable.

Notes:

* Recordings must be mono. Every video must have exactly one matching `wav/*task<T>_<rep>.wav`; if `wav/`
  holds files from other sessions with the same task/repetition numbers the step stops with an error, so keep
  one session per root folder.
* Re-running is safe: already renamed videos are left alone and outputs are overwritten.

## Installation

```
pip install -r requirements-align.txt
```

Needs `ffmpeg`: the one on your `PATH` is used if present, otherwise the bundled one from `imageio-ffmpeg`
(or pass `--ffmpeg <path>`).

## Layout and tests

```
rtmri_align/
  __main__.py   python -m rtmri_align
  core.py       all steps
tests/test_align.py   end-to-end test on a synthetic root folder
```

```
python tests/test_align.py
```
