#!/usr/bin/env python3
"""
Helper script to validate a MIDI file and render it to WAV so it can be
played easily (e.g., QuickTime will open the WAV). Uses pretty_midi + FluidSynth.

Python requirements (already mostly in your env):
    pip install pretty_midi soundfile

System requirement (FluidSynth):
    brew install fluid-synth

SoundFont note:
Homebrew does NOT ship a separate "fluid-soundfont" formula. You must manually
download a General MIDI SoundFont (.sf2 or compressed .sf3). Two easy options:
    1. FluidR3 GM (LGPL) — search "FluidR3_GM.sf2" and download; place it somewhere like ~/soundfonts/FluidR3_GM.sf2
    2. MuseScore General (sf3) — download with:
             curl -L -o ~/soundfonts/MuseScore_General.sf3 \
                 https://github.com/musescore/MuseScore/raw/master/share/sound/MuseScore_General.sf3

Then either set:
    export SOUND_FONT=~/soundfonts/FluidR3_GM.sf2
or let this script auto-detect common paths.

Usage examples:
    python listen_midi.py reconstructed.mid
    python listen_midi.py reconstructed.mid --wav out.wav --play
    python listen_midi.py reconstructed.mid --sf2 ~/soundfonts/MuseScore_General.sf3 --play

It tries to locate a SoundFont automatically if SOUND_FONT is not set.
Supports .sf2 and .sf3 files.
"""

import os
import sys
import argparse
import pretty_midi
import soundfile as sf

# Try both module names: project name is pyFluidSynth, import name usually 'fluidsynth'.
try:
    from pyfluidsynth import Synth as FluidSynth  # some installs expose this name
except ImportError:
    try:
        from fluidsynth import Synth as FluidSynth  # common case
    except ImportError:
        FluidSynth = None

COMMON_SF2_PATHS = [
    os.environ.get("SOUND_FONT"),
    # User soundfonts directory (if user followed instructions)
    os.path.expanduser("~/soundfonts/FluidR3_GM.sf2"),
    os.path.expanduser("~/soundfonts/MuseScore_General.sf3"),
    # Homebrew typical installs (FluidR3 sometimes manually placed here)
    "/opt/homebrew/share/sounds/sf2/FluidR3_GM.sf2",  # Apple Silicon Homebrew
    "/usr/local/share/sounds/sf2/FluidR3_GM.sf2",      # Intel Homebrew
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",           # Linux fallback
]

def find_soundfont():
    for p in COMMON_SF2_PATHS:
        if p and os.path.isfile(p):
            return p
    return None

def validate_midi(path: str) -> bool:
    try:
        pm = pretty_midi.PrettyMIDI(path)
        # Basic sanity: at least one instrument with notes
        note_count = sum(len(inst.notes) for inst in pm.instruments)
        if note_count == 0:
            print(f"[WARN] No notes found in {path} (blank MIDI).")
        return True
    except Exception as e:
        print(f"[ERROR] Cannot load MIDI with pretty_midi: {e}")
        return False

def render_to_wav(midi_path: str, wav_path: str, sf2_path: str, sample_rate: int = 44100):
    # FluidSynth does NOT support .sf3 (Ogg-compressed) soundfonts directly; require .sf2.
    if sf2_path.lower().endswith('.sf3'):
        print('[ERROR] .sf3 soundfonts are not supported by fluidsynth here. Please download an .sf2 (e.g. FluidR3_GM.sf2).')
        print('Download example:')
        print('  mkdir -p ~/soundfonts')
        print('  curl -L -o ~/soundfonts/FluidR3_GM.sf2 https://archive.org/download/FluidR3_GM/FluidR3_GM.sf2')
        print('Then rerun with --sf2 ~/soundfonts/FluidR3_GM.sf2 or set SOUND_FONT.')
        sys.exit(1)
    if FluidSynth is None:
        # Fallback: use fluidsynth CLI if library bindings missing.
        cli_cmd = f"fluidsynth -ni '{sf2_path}' '{midi_path}' -F '{wav_path}' -r {sample_rate}"
        print("[WARN] Python FluidSynth bindings not found; attempting CLI fallback:")
        print("       " + cli_cmd)
        ret = os.system(cli_cmd)
        if ret != 0:
            print("[ERROR] CLI fluidsynth failed. Install with 'brew install fluid-synth' and ensure it's in PATH.")
            print("To install bindings: pip install pyfluidsynth")
            sys.exit(1)
        print(f"[OK] Rendered via CLI fluidsynth: {wav_path}")
        return
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        synth = FluidSynth(samplerate=sample_rate)
        sfid = synth.sfload(sf2_path)
        synth.program_select(0, sfid, 0, 0)
        # New pretty_midi API uses 'synthesizer' not 'synth'
        audio = pm.fluidsynth(fs=sample_rate, synthesizer=synth)
        synth.delete()
        sf.write(wav_path, audio, sample_rate)
        print(f"[OK] Rendered audio: {wav_path}")
    except Exception as e:
        print(f"[ERROR] Rendering failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Validate and render a MIDI file to WAV")
    parser.add_argument("midi", help="Path to input MIDI file")
    parser.add_argument("--wav", default=None, help="Output WAV path (default: <midi_basename>.wav)")
    parser.add_argument("--sf2", default=None, help="Path to .sf2 SoundFont (overrides auto-detect)")
    parser.add_argument("--play", action="store_true", help="Open the WAV after rendering (macOS 'open')")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate for synthesis (default 44100)")
    args = parser.parse_args()

    if not os.path.isfile(args.midi):
        print(f"[ERROR] MIDI file not found: {args.midi}")
        sys.exit(1)

    if not validate_midi(args.midi):
        print("[ERROR] MIDI validation failed; aborting.")
        sys.exit(1)

    sf2_path = args.sf2 or find_soundfont()
    if not sf2_path:
        print("[ERROR] No SoundFont found. Set SOUND_FONT or use --sf2 to specify one.")
        print("Example download (MuseScore General sf3):")
        print("  curl -L -o ~/soundfonts/MuseScore_General.sf3 \\")
        print("    https://github.com/musescore/MuseScore/raw/master/share/sound/MuseScore_General.sf3")
        print("Then: export SOUND_FONT=~/soundfonts/MuseScore_General.sf3")
        sys.exit(1)
    else:
        print(f"[INFO] Using SoundFont: {sf2_path}")

    wav_out = args.wav or os.path.splitext(os.path.basename(args.midi))[0] + ".wav"
    render_to_wav(args.midi, wav_out, sf2_path, sample_rate=args.sr)

    if args.play and sys.platform == "darwin":
        os.system(f"open '{wav_out}'")

if __name__ == "__main__":
    main()
