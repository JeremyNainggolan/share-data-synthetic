import json
import re
from copy import deepcopy


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_sr_from_output(output: str) -> str | None:
    """Extract SR content from ```SR ... ``` block in output."""
    match = re.search(r'```SR\n(.*?)```', output, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_erroneous_sr_from_instruction(instruction: str) -> str | None:
    """
    Extract the erroneous SR from instruction.
    In LOM, the instruction contains a ```SR ... ``` block (the erroneous one)
    before the 'Now generate' prompt. Take only the first SR block found.
    """
    matches = re.findall(r'```SR\n(.*?)```', instruction, re.DOTALL)
    # First block is the erroneous SR, second block is [Your Answer] placeholder
    for m in matches:
        if '[Your Answer]' not in m:
            return m.strip()
    return None


def extract_schema_from_instruction(instruction: str) -> set[str]:
    """Extract all table.column entries from schema list in instruction."""
    match = re.search(r"schema\s*=\s*\[(.*?)\]", instruction, re.DOTALL)
    if not match:
        return set()
    return set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)'", match.group(1)))


def extract_schema_used_in_sr(sr: str) -> set[str]:
    """Extract table.column references used inside SR (excluding df variables)."""
    all_refs = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b', sr))
    return {s for s in all_refs if not s.startswith('df') and not s.startswith('res')}


def normalize_sr_for_comparison(sr: str) -> str:
    """Normalize SR for structural comparison (collapse spaces, lowercase)."""
    sr = re.sub(r'\s+', ' ', sr.strip().lower())
    return sr


def get_sr_operations(sr: str) -> list[str]:
    """Return list of operation types per line in SR."""
    ops = []
    for line in sr.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if '.where(' in line:
            ops.append('where')
        elif '.groupby(' in line:
            ops.append('groupby')
        elif '.orderby(' in line or '.sort(' in line:
            ops.append('orderby')
        elif 'select_distinct' in line or ('select(' in line and 'distinct' in line):
            ops.append('select_distinct')
        elif '.select(' in line:
            ops.append('select')
        elif '.having(' in line:
            ops.append('having')
        elif '.limit(' in line:
            ops.append('limit')
        else:
            ops.append('other')
    return ops


def normalize_output_format(output: str) -> str | None:
    """Ensure output is wrapped in ```SR ... ```."""
    output = output.strip()
    if output.startswith('```SR'):
        return output
    if output and not output.startswith('```'):
        return f"```SR\n{output}\n```"
    return None


# ─────────────────────────────────────────────
# STAGE 1: DATA CLEANING
# ─────────────────────────────────────────────

def clean_lom(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {'original': len(data), 'removed_null': 0, 'removed_duplicate': 0}
    cleaned = []
    seen = set()

    for sample in data:
        # Remove null/empty fields
        if not all(sample.get(k, '').strip() for k in ['instruction', 'output', 'system']):
            stats['removed_null'] += 1
            continue

        # Remove duplicates based on instruction + output
        key = (sample['instruction'].strip(), sample['output'].strip())
        if key in seen:
            stats['removed_duplicate'] += 1
            continue
        seen.add(key)

        # Strip leading/trailing whitespace from each field
        s = deepcopy(sample)
        s['instruction'] = s['instruction'].strip()
        s['output'] = s['output'].strip()
        s['system'] = s['system'].strip()
        cleaned.append(s)

    stats['after_cleaning'] = len(cleaned)
    return cleaned, stats


# ─────────────────────────────────────────────
# STAGE 2: FORMAT NORMALIZATION
# ─────────────────────────────────────────────

def normalize_lom(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {'input': len(data), 'removed_invalid_format': 0}
    normalized = []

    for sample in data:
        s = deepcopy(sample)
        norm_output = normalize_output_format(s['output'])
        if norm_output is None:
            stats['removed_invalid_format'] += 1
            continue
        s['output'] = norm_output
        normalized.append(s)

    stats['after_normalization'] = len(normalized)
    return normalized, stats


# ─────────────────────────────────────────────
# STAGE 3: ANOMALOUS DATA FILTERING
# ─────────────────────────────────────────────

def filter_anomalies_lom(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        'input': len(data),
        'removed_no_erroneous_sr': 0,
        'removed_no_output_sr': 0,
        'removed_identical_sr': 0,
    }
    filtered = []

    for sample in data:
        instruction = sample['instruction']
        output = sample['output']

        # Extract erroneous SR from instruction
        erroneous_sr = extract_erroneous_sr_from_instruction(instruction)
        if not erroneous_sr:
            stats['removed_no_erroneous_sr'] += 1
            continue

        # Extract corrected SR from output
        corrected_sr = extract_sr_from_output(output)
        if not corrected_sr:
            stats['removed_no_output_sr'] += 1
            continue

        # Check erroneous SR and corrected SR are NOT identical (structural comparison)
        if normalize_sr_for_comparison(erroneous_sr) == normalize_sr_for_comparison(corrected_sr):
            stats['removed_identical_sr'] += 1
            continue

        filtered.append(sample)

    stats['after_filter'] = len(filtered)
    return filtered, stats


# ─────────────────────────────────────────────
# STAGE 4: STRUCTURE & CONSISTENCY VALIDATION
# ─────────────────────────────────────────────

def validate_lom(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        'input': len(data),
        'removed_no_res': 0,
        'removed_schema_changed': 0,
        'alignment_scores': [],
    }
    validated = []

    for sample in data:
        instruction = sample['instruction']
        output = sample['output']

        corrected_sr = extract_sr_from_output(output)
        if not corrected_sr:
            continue

        lines = [l.strip() for l in corrected_sr.strip().split('\n') if l.strip()]

        # Validate last line is res = ...
        if not lines[-1].startswith('res'):
            stats['removed_no_res'] += 1
            continue

        # Validate corrected SR only uses schema from instruction
        available_schema = extract_schema_from_instruction(instruction)
        if available_schema:
            sr_schema_used = extract_schema_used_in_sr(corrected_sr)
            invalid_schema = sr_schema_used - available_schema
            if invalid_schema:
                stats['removed_schema_changed'] += 1
                continue

        # Compute alignment score: how many operations changed (meaningful correction)
        erroneous_sr = extract_erroneous_sr_from_instruction(instruction)
        if erroneous_sr:
            err_ops = get_sr_operations(erroneous_sr)
            cor_ops = get_sr_operations(corrected_sr)
            # Score: ratio of lines that are different (correction coverage)
            err_lines = [l.strip() for l in erroneous_sr.split('\n') if l.strip()]
            cor_lines = [l.strip() for l in corrected_sr.split('\n') if l.strip()]
            changed = sum(
                1 for e, c in zip(err_lines, cor_lines)
                if normalize_sr_for_comparison(e) != normalize_sr_for_comparison(c)
            )
            total = max(len(err_lines), len(cor_lines))
            score = round(changed / total, 4) if total > 0 else 0.0
        else:
            score = 1.0

        stats['alignment_scores'].append(score)
        s = deepcopy(sample)
        s['alignment_score'] = score
        validated.append(s)

    avg_score = (
        round(sum(stats['alignment_scores']) / len(stats['alignment_scores']), 4)
        if stats['alignment_scores'] else 0
    )
    stats['avg_alignment_score'] = avg_score
    stats['after_validation'] = len(validated)
    return validated, stats


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def preprocess_lom(input_path: str, output_path: str):
    print("=" * 50)
    print("LOM PREPROCESSING PIPELINE")
    print("=" * 50)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n[LOAD] {len(data)} samples loaded from {input_path}")

    # Stage 1: Cleaning
    data, s1 = clean_lom(data)
    print(f"\n[STAGE 1] Data Cleaning")
    print(f"  Original         : {s1['original']}")
    print(f"  Removed null     : {s1['removed_null']}")
    print(f"  Removed duplicate: {s1['removed_duplicate']}")
    print(f"  After cleaning   : {s1['after_cleaning']}")

    # Stage 2: Normalization
    data, s2 = normalize_lom(data)
    print(f"\n[STAGE 2] Format Normalization")
    print(f"  Input            : {s2['input']}")
    print(f"  Removed invalid format: {s2['removed_invalid_format']}")
    print(f"  After normalization: {s2['after_normalization']}")

    # Stage 3: Anomaly Filtering
    data, s3 = filter_anomalies_lom(data)
    print(f"\n[STAGE 3] Anomalous Data Filtering")
    print(f"  Input                    : {s3['input']}")
    print(f"  Removed no erroneous SR  : {s3['removed_no_erroneous_sr']}")
    print(f"  Removed no output SR     : {s3['removed_no_output_sr']}")
    print(f"  Removed identical SR     : {s3['removed_identical_sr']}")
    print(f"  After filter             : {s3['after_filter']}")

    # Stage 4: Validation
    data, s4 = validate_lom(data)
    print(f"\n[STAGE 4] Structure & Consistency Validation")
    print(f"  Input                  : {s4['input']}")
    print(f"  Removed no res         : {s4['removed_no_res']}")
    print(f"  Removed schema changed : {s4['removed_schema_changed']}")
    print(f"  Avg alignment score    : {s4['avg_alignment_score']}")
    print(f"  After validation       : {s4['after_validation']}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(data)} samples saved to {output_path}")
    print(f"[SUMMARY] {s1['original']} → {len(data)} samples retained")
    print("=" * 50)


if __name__ == '__main__':
    preprocess_lom(
        input_path='./dataset/lom_train_data.json',
        output_path='./output/lom_train_data.json'
    )