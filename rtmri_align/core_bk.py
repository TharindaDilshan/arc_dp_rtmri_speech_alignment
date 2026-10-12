"""Align in-scanner audio recordings with reconstructed rtMRI videos.

Expected layout under ROOT (``wav``, ``avi`` and ``mp4`` are created if missing)::

    ROOT/In-scanner-recordings/   DSP_OUT*.wav and TTL*.wav  (raw recordings)
    ROOT/unaligned/               silent videos from the reconstruction (meas_*.mp4)
    ROOT/wav/                     cropped audio (output)
    ROOT/avi/                     video + audio, uncompressed AVI (output)
    ROOT/mp4/                     final lossless MP4 with audio (output)

Steps
1. rename ``meas_*.mp4`` to ``<subject>_<date>_task<T>_<rep>.mp4``;
2. work out which DSP recording belongs to which task/repetition;
3. crop every recording to start 1800 samples after the first TTL sample above 0.007;
4. resample the audio to 24996 Hz, trim/pad it to the video length and write it frame-by-frame
   (1250 samples per frame) together with the video into an uncompressed AVI;
5. convert every AVI to a lossless H.264 MP4 (``-preset ultrafast -crf 0``, audio copied).
"""
from __future__ import annotations

import argparse
import fnmatch
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

AUDIO_DIR = "In-scanner-recordings"
WAV_DIR = "wav"
UNALIGNED_DIR = "unaligned"
AVI_DIR = "avi"
MP4_DIR = "mp4"

# (name contained in the reconstructed file name, repetitions); task number = TASK_START + index
TASKS = [("Intervocalic_Consonants", 5), ("Vowels_bVt", 5), ("Vowels_dVn", 5),
         ("Polysyl_Words", 5), ("Con_Speech", 1)]
TASK_START = 2

TTL_THRESHOLD = 0.007          # first |TTL| sample above this marks the start
TTL_OFFSET = 1800              # samples after that first sample
FS_OUT = 24996                 # audio is resampled to this rate
FREE_SPEECH_BYTES = 2_000_000  # expected size of the last (free speech) recording ...
FREE_SPEECH_TOL = 300_000      # ... +/- this many bytes


# --------------------------------------------------------------------------- helpers
def mround(x: float) -> int:
    """Round half away from zero (what the original processing uses), not banker's rounding."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def _list(folder: str, pattern: str):
    """Sorted file names in `folder` matching a glob pattern (case-insensitive, like Windows)."""
    pat = pattern.lower()
    return sorted((f for f in os.listdir(folder) if fnmatch.fnmatchcase(f.lower(), pat)),
                  key=str.lower)


def find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it, add it to PATH, pass --ffmpeg, or `pip install imageio-ffmpeg`")


def write_wav_16bit(path: str, x: np.ndarray, fs: int):
    """Write float audio in [-1, 1] as 16-bit PCM (scale 2^15, round half away, clip)."""
    import soundfile as sf
    y = np.asarray(x, dtype=np.float64) * 32768.0
    q = np.copysign(np.floor(np.abs(y) + 0.5), y)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    sf.write(path, q, int(fs), subtype="PCM_16")


def read_wav(path: str):
    """float64 samples in [-1, 1], shape (n, channels), and the sample rate."""
    import soundfile as sf
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    return x, fs


def resample_audio(x: np.ndarray, fs_in: int, fs_out: int = FS_OUT) -> np.ndarray:
    """Polyphase resampling with the usual anti-aliasing design: Kaiser window (beta=5), filter
    half-length 10*max(p, q), zero-phase (delay compensated), output length ceil(n*p/q)."""
    from scipy.signal import resample_poly
    g = math.gcd(int(fs_out), int(fs_in))
    p, q = int(fs_out) // g, int(fs_in) // g
    if p == q:
        return x.copy()
    return resample_poly(x, p, q, axis=0, window=("kaiser", 5.0))


# --------------------------------------------------------------------------- step 1
def rename_mp4s(unaligned: str, subject: int, date: str):
    for old in _list(unaligned, "meas_*.mp4"):
        for t, (name, _reps) in enumerate(TASKS, start=1):
            if name in old:
                task = TASK_START + t - 1
                m = re.search(r"Run_(\d+)", old)
                run = int(m.group(1)) if m else 1          # Con_Speech has a single run, no "Run_"
                new = f"{subject}_{date}_task{task}_{run}.mp4"
                os.replace(os.path.join(unaligned, old), os.path.join(unaligned, new))
                break


# --------------------------------------------------------------------------- step 2
def assign_task_files(audio_dir: str):
    """Returns {task: [DSP file names]} for tasks 2..6 using the recording sizes/order.

    Assumptions (as in the acquisition protocol): the two largest recordings are task 1, the next five
    largest are task 2, the last recording is the ~2 MB free-speech task 6, and exactly 15 recordings
    lie between task 2 and task 6 (tasks 3-5, five repetitions each).
    """
    files = _list(audio_dir, "DSP_OUT*.wav")
    sizes = [os.path.getsize(os.path.join(audio_dir, f)) for f in files]
    n = len(files)
    if n < 8:
        raise RuntimeError(f"Found only {n} DSP_OUT*.wav files in {audio_dir}")
    info = [(i + 1, sizes[i]) for i in range(n)]                  # 1-based file number, bytes
    desc = sorted(info, key=lambda r: -r[1])                      # stable: ties keep file order
    top_two = sorted(desc[0:2], key=lambda r: r[0])
    task1 = [files[r[0] - 1] for r in top_two]
    task2_info = sorted(desc[2:7], key=lambda r: r[0])
    task2 = [files[r[0] - 1] for r in task2_info]
    last_task2 = task2_info[4][0]

    last = n
    if not abs(sizes[last - 1] - FREE_SPEECH_BYTES) < FREE_SPEECH_TOL:
        raise AssertionError("Free Speech File is an odd size!")
    task6 = [files[last - 1]]

    between = last - last_task2 - 1
    if between != 15:
        raise RuntimeError(
            f"Found {between} recordings between Task 2 and Task 6 (expected 15). "
            "Please try to read the notes and delete error runs between Task 3 and Task 5")
    tasks = {1: task1, 2: task2, 6: task6}
    for k, task in enumerate((3, 4, 5)):
        tasks[task] = [files[i - 1] for i in range(last_task2 + 1 + 5 * k, last_task2 + 6 + 5 * k)]
    return tasks


# --------------------------------------------------------------------------- step 3
def crop_to_ttl(audio_dir: str, wav_dir: str, subject: int, tasks: dict):
    """Crop each task 2-6 recording to start TTL_OFFSET samples after the first TTL pulse."""
    for task in (2, 3, 4, 5, 6):
        for rep, dsp in enumerate(tasks[task], start=1):
            ttl_name = "TTL" + dsp[7:]
            audio, _ = read_wav(os.path.join(audio_dir, dsp))
            ttl, fs = read_wav(os.path.join(audio_dir, ttl_name))
            if audio.shape[1] != 1 or ttl.shape[1] != 1:
                raise RuntimeError(f"{dsp}: expected mono recordings")
            hits = np.flatnonzero(np.abs(ttl[:, 0]) > TTL_THRESHOLD)
            if hits.size == 0:
                raise RuntimeError("Check TTL file: " + ttl_name)
            start = int(hits[0]) + TTL_OFFSET                       # 0-based start index
            new = audio[start:, :]
            if new.shape[0] == 0:
                raise RuntimeError(f"TTL start is beyond the end of {dsp}")
            out = os.path.join(wav_dir, f"{subject}_{dsp[8:18]}_task{task}_{rep}.wav")
            write_wav_16bit(out, new, fs)


# --------------------------------------------------------------------------- step 4
def _probe_video(path: str):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        if c.duration is not None:
            duration = float(c.duration) / av.time_base
        else:
            duration = float(s.duration * s.time_base)
        rate = Fraction(s.average_rate or s.guessed_rate)
        n = sum(1 for _ in c.decode(video=0))                       # number of decodable frames
        return duration, rate, n, s.codec_context.width, s.codec_context.height


def combine_one(video_path: str, wav_path: str, avi_path: str, ffmpeg: str):
    import av
    import soundfile as sf

    duration, rate, n_decoded, width, height = _probe_video(video_path)
    audio, fs_orig = read_wav(wav_path)
    audio = resample_audio(audio, fs_orig, FS_OUT)
    audio_fs = FS_OUT

    audio_duration = audio.shape[0] / audio_fs
    print(f"Video Duration: {duration:.2f} seconds")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    if audio_duration > duration:
        n = mround(duration * audio_fs)
        audio = audio[:n, :]
        print(f"Trimmed Audio Duration: {audio.shape[0] / audio_fs:.2f} seconds")
    elif audio_duration < duration:
        n = mround(duration * audio_fs)
        audio = np.concatenate([audio, np.zeros((n - audio.shape[0], audio.shape[1]))], axis=0)
        print(f"Padded Audio Duration: {audio.shape[0] / audio_fs:.2f} seconds")

    frame_rate = float(rate)
    samples_per_frame = mround(audio_fs / frame_rate)               # 1250 at 24996 Hz / 19.9968 fps
    num_frames = mround(duration * frame_rate)
    required = num_frames * samples_per_frame
    if audio.shape[0] < required:
        audio = np.concatenate([audio, np.zeros((required - audio.shape[0], audio.shape[1]))], axis=0)
    else:
        audio = audio[:required, :]

    n_write = min(n_decoded, num_frames)                            # frames actually written
    audio = audio[: n_write * samples_per_frame, :]
    # audio rate stored in the AVI: one frame's worth of samples per video frame
    avi_audio_rate = mround(frame_rate * samples_per_frame)

    tmp_wav = None
    try:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="rtmri_audio_")
        os.close(fd)
        write_wav_16bit(tmp_wav, audio, avi_audio_rate)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-framerate", f"{rate.numerator}/{rate.denominator}", "-i", "pipe:0",
               "-i", tmp_wav, "-map", "0:v", "-map", "1:a",
               "-c:v", "rawvideo", "-pix_fmt", "bgr24", "-c:a", "pcm_s16le", avi_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            with av.open(video_path) as c:
                for i, frame in enumerate(c.decode(video=0)):
                    if i >= n_write:
                        break
                    proc.stdin.write(frame.to_ndarray(format="rgb24").tobytes())
        finally:
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed writing {avi_path}")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    print(f"Final AVI video with audio saved as: {avi_path}")


def combine_all(unaligned: str, wav_dir: str, avi_dir: str, ffmpeg: str):
    for video_name in _list(unaligned, "*.mp4"):
        m = re.search(r"task(\d+)_(\d+)", video_name)
        if not m:
            print("Pattern not found in filename")
            continue
        task, rep = int(m.group(1)), int(m.group(2))
        matches = _list(wav_dir, f"*task{task}_{rep}.wav")
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one audio file matching '*task{task}_{rep}.wav' in "
                               f"{wav_dir} for {video_name}, found {len(matches)}: {matches}")
        wav_name = matches[0]
        out = os.path.join(avi_dir, wav_name[:-4] + ".avi")
        combine_one(os.path.join(unaligned, video_name), os.path.join(wav_dir, wav_name), out, ffmpeg)


# --------------------------------------------------------------------------- step 5
def avi_to_mp4(avi_dir: str, mp4_dir: str, ffmpeg: str):
    for name in _list(avi_dir, "*.avi"):
        out = os.path.join(mp4_dir, name[:-4] + ".mp4")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-i", os.path.join(avi_dir, name),
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-c:a", "copy", out]
        subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Align in-scanner audio with reconstructed rtMRI videos (see module docstring).")
    ap.add_argument("root", help="root folder containing In-scanner-recordings/, unaligned/, wav/, avi/, mp4/")
    ap.add_argument("--subject", type=int, required=True, help="subject ID, e.g. 275")
    ap.add_argument("--date", required=True, help="session date used in file names, e.g. 2026-10-07")
    ap.add_argument("--ffmpeg", default=None, help="path to ffmpeg (default: ffmpeg on PATH, else bundled)")
    a = ap.parse_args(argv)

    root = a.root
    audio_dir, unaligned = os.path.join(root, AUDIO_DIR), os.path.join(root, UNALIGNED_DIR)
    wav_dir, avi_dir, mp4_dir = (os.path.join(root, d) for d in (WAV_DIR, AVI_DIR, MP4_DIR))
    for d in (audio_dir, unaligned):
        if not os.path.isdir(d):
            sys.exit(f"Missing folder: {d}")
    for d in (wav_dir, avi_dir, mp4_dir):
        os.makedirs(d, exist_ok=True)
    ffmpeg = find_ffmpeg(a.ffmpeg)

    print("== Renaming videos")
    rename_mp4s(unaligned, a.subject, a.date)
    print("== Assigning recordings to tasks")
    tasks = assign_task_files(audio_dir)
    for t in sorted(tasks):
        print(f"  task {t}: {', '.join(tasks[t])}")
    print("== Cropping audio at the TTL pulse")
    crop_to_ttl(audio_dir, wav_dir, a.subject, tasks)
    print("== Combining video and audio")
    combine_all(unaligned, wav_dir, avi_dir, ffmpeg)
    print("== Converting AVI to MP4")
    avi_to_mp4(avi_dir, mp4_dir, ffmpeg)
    print("Done.")


if __name__ == "__main__":
    main()
