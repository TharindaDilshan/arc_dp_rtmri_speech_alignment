"""Tests for rtmri_align on a synthetic root folder.

Run:  python tests/test_align.py
"""
import math, os, subprocess, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtmri_align import core as align

FS = 8000
FPS = 379.94 / 19          # 19.9968 fps, as produced by the reconstruction


def write_gray_video(frames01, path, fps):
    """Minimal silent gray H.264 writer for test inputs (frames in [0, 1], shape (T, H, W))."""
    T, H, W = frames01.shape
    cmd = [align.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write((np.clip(frames01, 0, 1) * 255).round().astype(np.uint8).tobytes()); p.stdin.close(); p.wait()


def synth_audio(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(n)
    return np.clip(x, -0.99, 0.99)


def to_int16(x):
    return np.round(x * 32768).clip(-32768, 32767).astype(np.int16)


def build_root(root, subject=275, date="2026-10-07"):
    ad = os.path.join(root, "In-scanner-recordings"); os.makedirs(ad)
    os.makedirs(os.path.join(root, "unaligned"))
    nbytes = lambda b: b // 2                      # samples for a given file size (16-bit mono)
    plan = ([("t1", 3_000_000)] * 2 + [("t2", 2_600_000)] * 5 + [("t3", 1_200_000)] * 5 +
            [("t4", 48_000)] * 5 + [("t5", 48_000)] * 5 + [("t6", 2_000_000)])
    pulses, sources = {}, {}
    for i, (kind, size) in enumerate(plan, start=1):
        stamp = f"{date}_12-{i:02d}-00"
        n = nbytes(size) - 22                      # header is ~44 bytes
        x = synth_audio(n, i)
        sf.write(os.path.join(ad, f"DSP_OUT_{stamp}.wav"), to_int16(x), FS, subtype="PCM_16")
        ttl = 0.002 * np.random.default_rng(100 + i).standard_normal(n).clip(-3, 3)   # < 0.007
        p = 8000 + 37 * i
        ttl[p] = 0.5; ttl[p + 5] = 0.4
        sf.write(os.path.join(ad, f"TTL_{stamp}.wav"), ttl.astype(np.float32), FS, subtype="FLOAT")
        pulses[i] = p
        sources[i] = (f"DSP_OUT_{stamp}.wav", to_int16(x))
    # silent videos: 100 frames, gray, 146x146
    names = ["meas_MID1_FID1_Vowels_bVt_Run_2.mp4", "meas_MID2_FID2_Polysyl_Words_Run_5.mp4",
             "meas_MID3_FID3_Con_Speech.mp4", "other_video.mp4"]
    for k, name in enumerate(names):
        yy, xx = np.mgrid[0:146, 0:146]
        frames = np.stack([0.5 + 0.4 * np.sin((xx + 3 * f + 10 * k) / 9.0) * np.cos(yy / 11.0)
                           for f in range(100)]).astype(np.float32)
        write_gray_video(frames, os.path.join(root, "unaligned", name), FPS)
    return pulses, sources


def decode_audio_s16(path):
    r = subprocess.run([align.find_ffmpeg(), "-loglevel", "error", "-i", path, "-vn", "-f", "s16le",
                        "-ac", "1", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.int16)


def stream_info(path):
    import av
    with av.open(path) as c:
        v, a = c.streams.video[0], c.streams.audio[0]
        return dict(fps=float(v.average_rate), a_rate=a.rate, a_codec=a.codec_context.name,
                    v_pix=v.codec_context.pix_fmt, v_codec=v.codec_context.name)


def decode_rgb(path):
    import av
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def reference_resample(x, p, q, n=10, beta=5.0):
    """Independent re-implementation of the textbook resample algorithm (windowed LS-lowpass design,
    delay compensation, exact output length) used to check scipy-based `resample_audio`."""
    from scipy.signal import upfirdn
    g = math.gcd(p, q); p //= g; q //= g
    pqmax = max(p, q); fc = 1.0 / 2 / pqmax; L = 2 * n * pqmax + 1
    k = np.arange(L) - (L - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * k) * np.kaiser(L, beta)       # least-squares brick-wall design x window
    h = p * h / h.sum()
    Lhalf = (L - 1) // 2
    Lx = len(x)
    nz = int(math.floor(q - Lhalf % q))
    h = np.concatenate([np.zeros(nz), h]); Lhalf += nz
    delay = int(math.floor(math.ceil(Lhalf) / q))
    nz1 = 0
    while math.ceil(((Lx - 1) * p + len(h) + nz1) / q) - delay < math.ceil(Lx * p / q):
        nz1 += 1
    h = np.concatenate([h, np.zeros(nz1)])
    y = upfirdn(h, x, p, q)
    return y[delay: delay + math.ceil(Lx * p / q)]


def test_resample_matches_reference_algorithm():
    rng = np.random.default_rng(0)
    for fs_in, n in ((8000, 3000), (48000, 4000), (44100, 3000)):
        x = rng.standard_normal(n)
        ref = reference_resample(x, align.FS_OUT, fs_in)
        got = align.resample_audio(x[:, None], fs_in)[:, 0]
        assert len(ref) == len(got) == math.ceil(n * align.FS_OUT / fs_in)
        assert np.abs(ref - got).max() < 1e-9, (fs_in, np.abs(ref - got).max())


def test_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        pulses, sources = build_root(root)
        align.main([root, "--subject", "275", "--date", "2026-10-07"])
        P = lambda *a: os.path.join(root, *a)

        # 1. renaming
        assert sorted(os.listdir(P("unaligned"))) == sorted([
            "275_2026-10-07_task3_2.mp4", "275_2026-10-07_task5_5.mp4", "275_2026-10-07_task6_1.mp4",
            "other_video.mp4"]), os.listdir(P("unaligned"))

        # 2. task assignment + TTL crop: file numbers 8..12 = task3, 13..17 = task4, 18..22 = task5, 23 = task6
        tasks = align.assign_task_files(P("In-scanner-recordings"))
        assert [len(tasks[t]) for t in (1, 2, 3, 4, 5, 6)] == [2, 5, 5, 5, 5, 1]
        assert tasks[3][1] == sources[9][0] and tasks[5][4] == sources[22][0] and tasks[6][0] == sources[23][0]
        for task, num, rep in ((3, 9, 2), (5, 22, 5), (6, 23, 1)):
            w, fs = sf.read(P("wav", f"275_2026-10-07_task{task}_{rep}.wav"), dtype="int16")
            exp = sources[num][1][pulses[num] + 1800:]           # first TTL sample + 1800
            assert fs == FS and np.array_equal(w, exp), (task, rep)

        # 3. combined audio/video
        for task, num, rep, trim in ((3, 9, 2, True), (5, 22, 5, False), (6, 23, 1, True)):
            name = f"275_2026-10-07_task{task}_{rep}"
            src = P("unaligned", name + ".mp4")
            fin = P("mp4", name + ".mp4")
            v_src, v_fin = decode_rgb(src), decode_rgb(fin)
            n_frames = len(v_src)
            assert len(v_fin) == n_frames == 100
            assert np.abs(v_src.astype(int) - v_fin.astype(int)).max() <= 2
            info = stream_info(fin)
            assert info["v_codec"] == "h264" and info["a_codec"] == "pcm_s16le", info
            assert info["a_rate"] == 24996, info
            a_fin = decode_audio_s16(fin)
            assert len(a_fin) == n_frames * 1250, (len(a_fin), n_frames)
            # expected audio: read cropped wav -> resample -> trim/pad to frames*1250 -> int16
            w = sf.read(P("wav", name + ".wav"), dtype="float64")[0]
            ref = reference_resample(w, 24996, FS)
            need = n_frames * 1250
            ref = ref[:need] if len(ref) >= need else np.concatenate([ref, np.zeros(need - len(ref))])
            exp = np.clip(np.copysign(np.floor(np.abs(ref * 32768) + .5), ref), -32768, 32767).astype(np.int16)
            assert np.abs(a_fin.astype(int) - exp.astype(int)).max() <= 1, (name, np.abs(a_fin.astype(int) - exp).max())
            if not trim:                                          # padded case ends in zeros
                assert (a_fin[-100:] == 0).all()
        assert sorted(os.listdir(P("avi"))) == sorted(f[:-4] + ".avi" for f in os.listdir(P("mp4")))


if __name__ == "__main__":
    test_resample_matches_reference_algorithm(); print("resample == reference algorithm OK")
    test_end_to_end(); print("end-to-end OK")
