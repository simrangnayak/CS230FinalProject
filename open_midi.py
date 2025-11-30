import os
from glob import glob
import sys
import miditoolkit
import numpy as np

def check_midi_file(file_path):
    print(f"\n--- Checking: {file_path} ---")

    if not os.path.exists(file_path):
        print("❌ File not found.")
        return False

    try:
        midi_obj = miditoolkit.MidiFile(file_path)
    except Exception as e:
        print(f"❌ Could not parse MIDI: {e}")
        return False

    ok = True

    # ---- 1. Basic sanity checks ----
    if not hasattr(midi_obj, "ticks_per_beat") or midi_obj.ticks_per_beat <= 0:
        print(f"❌ Invalid ticks_per_beat: {midi_obj.ticks_per_beat}")
        ok = False
    else:
        print(f"✓ ticks_per_beat = {midi_obj.ticks_per_beat}")

    # ---- 2. Instruments and notes ----
    instruments = midi_obj.instruments
    print(f"✓ Number of instruments: {len(instruments)}")

    total_notes = sum(len(i.notes) for i in instruments)
    if total_notes == 0:
        print("❌ No notes found (ERROR(BLANK))")
        ok = False
    else:
        print(f"✓ Total notes: {total_notes}")

    for i, inst in enumerate(instruments):
        if len(inst.notes) == 0:
            continue
        for note in inst.notes:
            if note.start < 0 or note.end < 0:
                print(f"❌ Negative note time in instrument {i}")
                ok = False
                break
            if note.end <= note.start:
                print(f"❌ Bad note duration (end <= start) in instrument {i}")
                ok = False
                break
            if not (0 <= note.pitch <= 127):
                print(f"❌ Invalid pitch: {note.pitch}")
                ok = False
                break
            if not (0 <= note.velocity <= 127):
                print(f"❌ Invalid velocity: {note.velocity}")
                ok = False
                break

    # ---- 3. Time signatures ----
    tsc = midi_obj.time_signature_changes
    if len(tsc) == 0:
        print("⚠️ No explicit time signature changes (using default 4/4).")
    else:
        valid_ts = True
        for ts in tsc:
            if ts.numerator <= 0 or ts.denominator <= 0:
                print(f"❌ Invalid time signature: {ts.numerator}/{ts.denominator}")
                valid_ts = False
        if valid_ts:
            print("✓ Time signatures valid.")

    # ---- 4. Tempos ----
    tpc = midi_obj.tempo_changes
    if len(tpc) == 0:
        print("⚠️ No explicit tempo changes (default 120 BPM).")
    else:
        tempos = [tp.tempo for tp in tpc]
        if any(t <= 0 or t > 1000 for t in tempos):
            print(f"❌ Abnormal tempo values: {tempos}")
            ok = False
        else:
            print(f"✓ Tempos OK (range: {min(tempos)}–{max(tempos)} BPM)")

    # ---- 5. Duration sanity ----
    total_ticks = midi_obj.max_tick
    if total_ticks <= 0:
        print("❌ No measurable duration (max_tick <= 0).")
        ok = False
    else:
        print(f"✓ Total duration (ticks): {total_ticks}")

    print("✅ MIDI appears valid.\n" if ok else "❌ MIDI has issues.\n")
    return ok




tokens_per_note = 8

def get_vocab_sizes(folder):
    max_vals = [0]*tokens_per_note

    for path in glob(os.path.join(folder, '*.txt')):
        with open(path, 'r') as f:
            for line in f:
                # remove <s> and </s> tokens
                toks = [t for t in line.strip().split() if "<s>" not in t and "</s>" not in t]
                if not toks:
                    continue

                # convert <j-k> to integer k
                encoding = [int(t.split('-')[1][:-1]) for t in toks]
                octuples = [encoding[i:i+tokens_per_note]
                            for i in range(0, len(encoding), tokens_per_note)]

                # update max for each of the 8 channels
                for octuple in octuples:
                    for j in range(tokens_per_note):
                        max_vals[j] = max(max_vals[j], octuple[j])

    vocab_sizes = [m+1 for m in max_vals]    # +1 because values start at 0
    return vocab_sizes

# Example:
folder = "val_octuples"
print(get_vocab_sizes(folder))