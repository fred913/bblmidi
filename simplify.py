#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pretty_midi

# 一些常见三和弦模板（按 pitch class 表示）
CHORD_TEMPLATES: Dict[str, List[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}


@dataclass
class MelodyCandidate:
    note: pretty_midi.Note
    group_size: int
    gap_to_second: int
    other_max_dur: float


def gather_all_notes(pm: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
    notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    return notes


def add_note_monophonic(
    notes_out: Union[List[pretty_midi.Note], pretty_midi.Instrument],
    note: pretty_midi.Note,
    min_dur: float = 0.01,
) -> None:
    if isinstance(notes_out, pretty_midi.Instrument):
        notes_out = notes_out.notes
    if note.end - note.start < min_dur:
        return
    if not notes_out:
        notes_out.append(note)
        return
    prev = notes_out[-1]
    if note.start < prev.end:
        prev.end = min(prev.end, note.start)
        if prev.end - prev.start < min_dur:
            notes_out.pop()
    if notes_out and note.start < notes_out[-1].end:
        return
    notes_out.append(note)


def clone(n: pretty_midi.Note) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end
    )


def group_by_onset(
    notes: List[pretty_midi.Note], onset_tol: float
) -> List[List[pretty_midi.Note]]:
    notes = sorted(notes, key=lambda x: (x.start, x.pitch, x.end))
    groups = []
    i = 0
    while i < len(notes):
        t0 = notes[i].start
        g = [notes[i]]
        i += 1
        while i < len(notes) and abs(notes[i].start - t0) <= onset_tol:
            g.append(notes[i])
            i += 1
        groups.append(sorted(g, key=lambda x: x.pitch))
    return groups


def extract_theme_by_top_onset(
    all_notes: List[pretty_midi.Note],
    onset_tol: float = 0.03,
) -> Tuple[List[pretty_midi.Note], List[pretty_midi.Note]]:
    groups = group_by_onset(all_notes, onset_tol)
    theme_out: List[pretty_midi.Note] = []
    remaining: List[pretty_midi.Note] = []

    for g in groups:
        if len(g) == 1:
            add_note_monophonic(theme_out, clone(g[0]))
            continue
        top = max(g, key=lambda n: n.pitch)
        add_note_monophonic(theme_out, clone(top))
        for n in g:
            if n is not top:
                remaining.append(n)

    remaining = sorted(remaining, key=lambda x: (x.start, x.pitch, x.end))
    return theme_out, remaining


def guess_chord_root(pitch_classes: Set[int], bass_pitch: int) -> Tuple[int, str]:
    """
    返回 (root_pc, quality)
    用“模板命中数”打分，平分时偏向 bass_pitch 所在 pitch class
    """
    bass_pc = bass_pitch % 12
    best = None  # (score, bass_bonus, root_pc, quality)
    for root_pc in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            tpl = {(root_pc + x) % 12 for x in intervals}
            score = len(tpl & pitch_classes)
            bass_bonus = 1 if root_pc == bass_pc else 0
            cand = (score, bass_bonus, root_pc, quality)
            if best is None or cand > best:
                best = cand
    assert best is not None
    return best[2], best[3]


def choose_note_with_pc(
    group: List[pretty_midi.Note], pc: int, exclude_pitches: Set[int], prefer: str
) -> Optional[pretty_midi.Note]:
    cands = [
        n for n in group if (n.pitch % 12) == pc and n.pitch not in exclude_pitches
    ]
    if not cands:
        return None
    if prefer == "low":
        return min(cands, key=lambda n: n.pitch)
    if prefer == "high":
        return max(cands, key=lambda n: n.pitch)
    # prefer == "mid"
    pitches = sorted([n.pitch for n in group])
    mid = pitches[len(pitches) // 2]
    return min(cands, key=lambda n: abs(n.pitch - mid))


def choose_chord_tone(
    group: List[pretty_midi.Note], root_pc: int, quality: str, exclude_pitches: Set[int]
) -> Optional[pretty_midi.Note]:
    """
    chord 轨优先拿“三度音”。sus 和弦没有三度就退化拿“第二优先音”
    """
    intervals = CHORD_TEMPLATES[quality]
    target_pc = None

    if quality in ("maj", "aug", "sus4", "sus2"):
        # maj/aug 的“中间音”是大三度，sus 系列中间音是 2 或 5
        target_pc = (root_pc + intervals[1]) % 12
    elif quality in ("min", "dim"):
        target_pc = (root_pc + intervals[1]) % 12
    else:
        target_pc = (root_pc + 4) % 12  # 兜底

    n = choose_note_with_pc(group, target_pc, exclude_pitches, prefer="mid")
    if n is not None:
        return n

    # 找不到目标音就拿“除 bass 外最中间”的一个音
    remaining = [x for x in group if x.pitch not in exclude_pitches]
    if not remaining:
        return None
    remaining.sort(key=lambda x: x.pitch)
    return remaining[len(remaining) // 2]


def clone_note(src: pretty_midi.Note) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=src.velocity, pitch=src.pitch, start=src.start, end=src.end
    )


def split_midi(
    in_path: str,
    out_path: str,
    onset_tol: float = 0.03,
    bass_split_pitch: int = 48,  # 低于 C3 更倾向 bass
    melody_min_pitch: int = 60,  # C4 以上更倾向 melody
    melody_min_gap: int = 7,  # 最高音比次高音至少高 7 半音更像旋律
    melody_max_step: int = 12,  # 旋律连贯性阈值，允许最大跳进
    melody_dur_bonus: float = 0.25,  # 旋律音明显更长也更像旋律
) -> None:
    pm = pretty_midi.PrettyMIDI(in_path)
    notes = gather_all_notes(pm)
    groups = group_by_onset(notes, onset_tol)

    out = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())

    inst_mel = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Lead 1 (square)"), name="melody"
    )
    inst_chd = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Lead 1 (square)"), name="chord"
    )
    inst_bas = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Lead 1 (square)"), name="bass"
    )

    melody_candidates: List[Optional[MelodyCandidate]] = []

    for g in groups:
        g_sorted = sorted(g, key=lambda n: n.pitch)
        pcs = {n.pitch % 12 for n in g_sorted}
        group_size = len(g_sorted)

        # 单音簇先按音区粗分
        if group_size == 1:
            n = g_sorted[0]
            if n.pitch < bass_split_pitch:
                add_note_monophonic(inst_bas, clone_note(n))
                melody_candidates.append(None)
            else:
                melody_candidates.append(MelodyCandidate(clone_note(n), 1, 127, 0.0))
            continue

        # 多音簇当作和弦簇
        root_pc, quality = guess_chord_root(pcs, bass_pitch=g_sorted[0].pitch)

        exclude: Set[int] = set()

        # bass 轨优先拿根音所在 pitch class 的最低音
        bass_note = choose_note_with_pc(g_sorted, root_pc, exclude, prefer="low")
        if bass_note is None:
            bass_note = g_sorted[0]
        exclude.add(bass_note.pitch)
        add_note_monophonic(inst_bas, clone_note(bass_note))

        # chord 轨优先拿三度音
        chord_note = choose_chord_tone(g_sorted, root_pc, quality, exclude)
        if chord_note is not None:
            exclude.add(chord_note.pitch)
            add_note_monophonic(inst_chd, clone_note(chord_note))

        # melody 候选只从剩余音里拿最高音
        remaining = [n for n in g_sorted if n.pitch not in exclude]
        if not remaining:
            melody_candidates.append(None)
            continue

        remaining.sort(key=lambda n: n.pitch)
        top = remaining[-1]
        second = remaining[-2] if len(remaining) >= 2 else None
        gap = top.pitch - (second.pitch if second is not None else top.pitch)

        other_max_dur = 0.0
        for n in g_sorted:
            if n.pitch != top.pitch:
                other_max_dur = max(other_max_dur, n.end - n.start)

        melody_candidates.append(
            MelodyCandidate(clone_note(top), group_size, gap, other_max_dur)
        )

    # 第二遍用“音区 + 高度差 + 连贯性 + 时值优势”过滤旋律候选
    prev_pitch: Optional[int] = None
    for cand in melody_candidates:
        if cand is None:
            continue
        n = cand.note
        dur = n.end - n.start

        keep = False
        if cand.group_size == 1:
            keep = True
        elif n.pitch >= melody_min_pitch:
            keep = True
        elif cand.gap_to_second >= melody_min_gap and n.pitch >= (melody_min_pitch - 5):
            keep = True
        elif (
            prev_pitch is not None
            and abs(n.pitch - prev_pitch) <= melody_max_step
            and n.pitch >= (melody_min_pitch - 5)
        ):
            keep = True
        elif dur >= cand.other_max_dur + melody_dur_bonus and n.pitch >= (
            melody_min_pitch - 5
        ):
            keep = True

        if keep:
            add_note_monophonic(inst_mel, n)
            prev_pitch = n.pitch

    out.instruments.append(inst_mel)
    out.instruments.append(inst_chd)
    out.instruments.append(inst_bas)
    out.write(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_midi", nargs="?", help="input single-track midi file")
    ap.add_argument("output_midi", nargs="?", help="output midi file with 3 tracks")
    ap.add_argument(
        "--all",
        action="store_true",
        help="process all *.mid/*.midi files from inputs/ into outputs/ with same base names",
    )
    ap.add_argument("--onset_tol", type=float, default=0.03)
    ap.add_argument("--bass_split_pitch", type=int, default=48)
    ap.add_argument("--melody_min_pitch", type=int, default=60)
    ap.add_argument("--melody_min_gap", type=int, default=7)
    ap.add_argument("--melody_max_step", type=int, default=12)
    ap.add_argument("--melody_dur_bonus", type=float, default=0.25)
    args = ap.parse_args()

    if args.all:
        input_dir = Path("inputs")
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        midi_files = sorted(
            [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".mid", ".midi")
            ]
        )

        if not midi_files:
            ap.error("--all was set but no .mid/.midi files were found in inputs/")

        for in_file in midi_files:
            out_file = output_dir / in_file.name
            split_midi(
                str(in_file),
                str(out_file),
                onset_tol=args.onset_tol,
                bass_split_pitch=args.bass_split_pitch,
                melody_min_pitch=args.melody_min_pitch,
                melody_min_gap=args.melody_min_gap,
                melody_max_step=args.melody_max_step,
                melody_dur_bonus=args.melody_dur_bonus,
            )
            print(f"processed: {in_file} -> {out_file}")
        return

    if not args.input_midi or not args.output_midi:
        ap.error("input_midi and output_midi are required unless --all is used")

    split_midi(
        args.input_midi,
        args.output_midi,
        onset_tol=args.onset_tol,
        bass_split_pitch=args.bass_split_pitch,
        melody_min_pitch=args.melody_min_pitch,
        melody_min_gap=args.melody_min_gap,
        melody_max_step=args.melody_max_step,
        melody_dur_bonus=args.melody_dur_bonus,
    )


if __name__ == "__main__":
    main()
