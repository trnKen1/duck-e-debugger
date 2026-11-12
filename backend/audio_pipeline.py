"""Phase 1 — active listening pipeline.

Continuously captures the microphone, uses Silero VAD to segment speech from
silence/noise, gates each segment against the enrolled voice profile
(Resemblyzer cosine similarity) so only the user's own speech proceeds, then
transcribes accepted segments with faster-whisper (GPU).

If no voice_profile.npy exists yet (see enroll_voice.py), speaker filtering
is skipped and every detected utterance is transcribed — useful for testing
the VAD/STT stages before enrolling, but not the intended steady state.

Two details keep the input clean:
- Pre-roll: the last ~300ms of audio before VAD fires is kept and prepended
  to each utterance, so soft word onsets ("h", "f", "s") aren't clipped.
- Playback mute: while the shell is playing a reply (see mute_for_playback,
  called by server.py), mic audio is dropped, so the assistant never hears
  itself. Trade-off: you can't talk over a reply.
"""

import asyncio
import collections
import logging
import os
import queue
import sys
import sysconfig
import time

import numpy as np
import sounddevice as sd

log = logging.getLogger("duck-e-bugger-backend.audio")


def _register_nvidia_dll_dirs() -> None:
    """faster-whisper's ctranslate2 backend loads cuBLAS/cuDNN at inference
    time. The pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 install
    those DLLs into site-packages but don't add them to PATH, so ctranslate2
    can't find them unless we register the directories first — must happen
    before `faster_whisper` (and therefore ctranslate2) is imported.

    `os.add_dll_directory` alone isn't enough here: it only affects
    LoadLibraryEx calls made with the extended search flags, and
    ctranslate2's loader doesn't use them. Prepending to PATH also covers
    the classic DLL search order, which is what actually works."""
    if sys.platform != "win32":
        return
    site_packages = sysconfig.get_path("purelib")
    for pkg in ("cublas", "cudnn"):
        dll_dir = os.path.join(site_packages, "nvidia", pkg, "bin")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


_register_nvidia_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 — must follow DLL registration
from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: E402
from silero_vad import VADIterator, load_silero_vad  # noqa: E402

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512  # Silero VAD's required frame size at 16kHz
MIN_UTTERANCE_SECONDS = 0.3  # discard shorter blips (coughs, clicks)
# Silero's default (100ms) ends an utterance at every mid-sentence pause,
# turning one spoken request into several fragmentary agent turns.
VAD_MIN_SILENCE_MS = 700
# Silero only fires "start" once speech is already underway; without this,
# the first syllable of a softly spoken word can be lost.
PREROLL_SECONDS = 0.3
PREROLL_CHUNKS = round(PREROLL_SECONDS * SAMPLE_RATE / VAD_CHUNK_SAMPLES)
# Extra mute time after a reply's audio should have finished: covers room
# echo and small shell-side scheduling delays.
PLAYBACK_TAIL_SECONDS = 0.4
SPEAKER_MATCH_THRESHOLD = 0.75
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "voice_profile.npy")
WHISPER_MODEL_SIZE = "small"


class ActiveListeningPipeline:
    def __init__(self, vocabulary: list[str] | None = None) -> None:
        # Names Whisper should prefer when the audio is ambiguous (e.g.
        # "Duck-e-bugger", which it otherwise hears as "Duckley Debugger").
        # Passed as faster-whisper hotwords; from access_config.json.
        self._vocabulary = [v.lower() for v in vocabulary or []]
        self._hotwords = ", ".join(vocabulary) if vocabulary else None
        self._vad_model = load_silero_vad()
        self._vad_iterator = VADIterator(
            self._vad_model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )
        self._encoder = VoiceEncoder()  # auto-selects CUDA if available, else CPU
        self._profile = self._load_profile()
        self._whisper = self._load_whisper()
        # (capture time from time.monotonic(), samples)
        self._audio_queue: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue()
        self._stream: sd.InputStream | None = None
        # Mic audio captured in [_mute_from, _playback_end + tail] is dropped.
        self._mute_from = 0.0
        self._playback_end = 0.0

    def _load_whisper(self) -> WhisperModel:
        try:
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
            # ctranslate2 loads its CUDA libs lazily — construction can
            # succeed even when cuBLAS/cuDNN aren't actually loadable. Force
            # one real inference here so a missing DLL fails at startup
            # instead of on the user's first spoken query.
            list(model.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), language="en")[0])
            log.info("faster-whisper running on GPU (%s)", WHISPER_MODEL_SIZE)
            return model
        except Exception as exc:  # noqa: BLE001 — no usable CUDA runtime for ctranslate2
            log.warning(
                "GPU unavailable for faster-whisper (%s), falling back to CPU (slower): %s",
                WHISPER_MODEL_SIZE,
                exc,
            )
            return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    def _load_profile(self) -> np.ndarray | None:
        if os.path.exists(PROFILE_PATH):
            log.info("loaded voice profile from %s", PROFILE_PATH)
            return np.load(PROFILE_PATH)
        log.warning(
            "no voice_profile.npy found — speaker filtering disabled, every "
            "detected utterance will be transcribed. Run enroll_voice.py to fix this."
        )
        return None

    def start(self) -> None:
        def callback(indata, frames, time_info, status):
            if status:
                log.warning("sounddevice status: %s", status)
            self._audio_queue.put((time.monotonic(), indata[:, 0].copy()))

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=callback,
        )
        self._stream.start()
        log.info("mic stream started (%s)", sd.query_devices(kind="input")["name"])

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("mic stream stopped")

    def mute_for_playback(self, seconds: float) -> None:
        """Called by server.py when it sends a reply's audio to the shell.
        Mirrors the shell's back-to-back scheduling (handleAudioEnd in
        main.js): this clip plays when the previous one ends, or now if
        nothing is playing. Mic audio captured until then (plus a short
        echo tail) is dropped."""
        now = time.monotonic()
        if now > self._playback_end + PLAYBACK_TAIL_SECONDS:
            self._mute_from = now
        self._playback_end = max(self._playback_end, now) + seconds

    def _is_muted(self, captured_at: float) -> bool:
        return self._mute_from <= captured_at <= self._playback_end + PLAYBACK_TAIL_SECONDS

    def _speaker_matches(self, segment: np.ndarray) -> bool:
        if self._profile is None:
            return True
        wav = preprocess_wav(segment, source_sr=SAMPLE_RATE)
        if wav.size == 0:
            return False
        embedding = self._encoder.embed_utterance(wav)
        similarity = float(
            np.dot(embedding, self._profile)
            / (np.linalg.norm(embedding) * np.linalg.norm(self._profile))
        )
        matched = similarity >= SPEAKER_MATCH_THRESHOLD
        # INFO, not DEBUG: a silent rejection is otherwise indistinguishable
        # from "the mic heard nothing" — and similarity drops noticeably when
        # the capture mic differs from the one used in enroll_voice.py.
        log.info(
            "utterance (%.1fs) speaker similarity %.3f %s threshold %.2f — %s",
            segment.size / SAMPLE_RATE,
            similarity,
            ">=" if matched else "<",
            SPEAKER_MATCH_THRESHOLD,
            "accepted" if matched else "discarded",
        )
        return matched

    def _transcribe(self, segment: np.ndarray) -> str:
        segments, _info = self._whisper.transcribe(
            segment, language="en", hotwords=self._hotwords
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if self._is_only_vocabulary(text):
            log.info("dropped transcript %r: only hotwords, likely a Whisper hallucination", text)
            return ""
        return text

    def _is_only_vocabulary(self, text: str) -> bool:
        """With nothing intelligible to transcribe, Whisper tends to echo
        the hotwords back verbatim (checked on 1s of silence)."""
        if not self._vocabulary or not text:
            return False
        rest = text.lower()
        for word in sorted(self._vocabulary, key=len, reverse=True):
            rest = rest.replace(word, " ")
        return not any(ch.isalnum() for ch in rest)

    async def listen(self):
        """Async generator yielding transcribed text for accepted utterances."""
        loop = asyncio.get_running_loop()
        preroll: collections.deque[np.ndarray] = collections.deque(maxlen=PREROLL_CHUNKS)
        buffer: list[np.ndarray] = []
        speech_start_s = 0.0
        in_speech = False
        muted = False

        while True:
            captured_at, chunk = await loop.run_in_executor(None, self._audio_queue.get)

            if self._is_muted(captured_at):
                if not muted:
                    muted = True
                    if in_speech:
                        log.info("utterance dropped — reply playback started mid-speech")
                    buffer, in_speech = [], False
                    preroll.clear()  # would otherwise hold the start of the reply
                    self._vad_iterator.reset_states()
                continue
            muted = False

            event = self._vad_iterator(chunk, return_seconds=True)
            if in_speech:
                buffer.append(chunk)
            elif event and "start" in event:
                in_speech = True
                buffer = [*preroll, chunk]
                speech_start_s = event["start"]
            preroll.append(chunk)

            if not (event and "end" in event and in_speech):
                continue
            in_speech = False
            segment = np.concatenate(buffer)
            buffer = []
            # Judge length on Silero's own speech timestamps, not the
            # segment: the segment also holds pre-roll and the
            # VAD_MIN_SILENCE_MS of trailing silence, which would let any
            # cough or click past the minimum.
            speech_s = event["end"] - speech_start_s
            if speech_s < MIN_UTTERANCE_SECONDS:
                # INFO, not DEBUG: a short "yes" to a permission prompt lands
                # here too, and was otherwise dropped without a trace.
                log.info("discarded %.2fs blip (below %.1fs minimum)", speech_s, MIN_UTTERANCE_SECONDS)
                continue

            t_vad_end = time.monotonic()
            matched = await loop.run_in_executor(None, self._speaker_matches, segment)
            t_speaker = time.monotonic()
            if not matched:
                continue

            text = await loop.run_in_executor(None, self._transcribe, segment)
            t_done = time.monotonic()
            log.info(
                "timing: transcript ready %.2fs after VAD end (speaker check %.2fs, "
                "whisper %.2fs); VAD itself waits %dms of silence before ending",
                t_done - t_vad_end,
                t_speaker - t_vad_end,
                t_done - t_speaker,
                VAD_MIN_SILENCE_MS,
            )
            if text:
                yield text
