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


def extract_masked_sr_from_instruction(instruction: str) -> str | None:
    """Extract masked SR content from ```Masked SR ... ``` block."""
    match = re.search(r'```Masked SR\n(.*?)```', instruction, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_all_schema_from_instruction(instruction: str) -> set[str]:
    """Extract all table.column entries from Schema section in instruction."""
    return set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)'", instruction))


def extract_schema_used_in_sr(sr: str) -> set[str]:
    """Extract table.column references used inside SR (excluding df variables)."""
    all_refs = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b', sr))
    return {s for s in all_refs if not s.startswith('df') and not s.startswith('res')}


def count_masks(text: str) -> int:
    """Count the number of [MASK] tokens in text."""
    return text.count('[MASK]')


def count_sr_operations(sr: str) -> int:
    """Count the number of operation lines in SR."""
    return len([l for l in sr.strip().split('\n') if l.strip() and '=' in l])


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

def clean_sam(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {'original': len(data), 'removed_null': 0, 'removed_duplicate': 0}
    cleaned = []
    seen = set()

    for sample in data:
        # Remove null/empty fields
        if not all(sample.get(k, '').strip() for k in ['instruction', 'output', 'system']):
            stats['removed_null'] += 1
            continue

        # Remove duplicates
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

def normalize_sam(data: list[dict]) -> tuple[list[dict], dict]:
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

def filter_anomalies_sam(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        'input': len(data),
        'removed_no_masked_sr': 0,
        'removed_no_sr_output': 0,
        'removed_mask_remaining': 0,
        'removed_schema_invalid': 0,
    }
    filtered = []

    for sample in data:
        instruction = sample['instruction']
        output = sample['output']

        # Extract masked SR from instruction
        masked_sr = extract_masked_sr_from_instruction(instruction)
        if not masked_sr:
            stats['removed_no_masked_sr'] += 1
            continue

        # Extract SR from output
        sr = extract_sr_from_output(output)
        if not sr:
            stats['removed_no_sr_output'] += 1
            continue

        # Check no [MASK] remaining in output
        if count_masks(output) > 0:
            stats['removed_mask_remaining'] += 1
            continue

        # Extract available schema from instruction
        available_schema = extract_all_schema_from_instruction(instruction)

        # Check all schema used in output SR exist in available schema
        sr_schema_used = extract_schema_used_in_sr(sr)
        invalid_schema = sr_schema_used - available_schema
        if invalid_schema:
            stats['removed_schema_invalid'] += 1
            continue

        filtered.append(sample)

    stats['after_filter'] = len(filtered)
    return filtered, stats


# ─────────────────────────────────────────────
# STAGE 4: STRUCTURE & CONSISTENCY VALIDATION
# ─────────────────────────────────────────────

def validate_sam(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        'input': len(data),
        'removed_no_res': 0,
        'removed_operation_count_mismatch': 0,
        'alignment_scores': [],
    }
    validated = []

    for sample in data:
        instruction = sample['instruction']
        output = sample['output']

        sr = extract_sr_from_output(output)
        if not sr:
            continue

        lines = [l.strip() for l in sr.strip().split('\n') if l.strip()]

        # Validate last line is res = ...
        if not lines[-1].startswith('res'):
            stats['removed_no_res'] += 1
            continue

        # Validate operation count: output SR should match masked SR operation count
        masked_sr = extract_masked_sr_from_instruction(instruction)
        if masked_sr:
            masked_op_count = count_sr_operations(masked_sr)
            output_op_count = count_sr_operations(sr)
            if masked_op_count != output_op_count:
                stats['removed_operation_count_mismatch'] += 1
                continue

        # Compute alignment score: ratio of schema used in output that are valid
        available_schema = extract_all_schema_from_instruction(instruction)
        sr_schema_used = extract_schema_used_in_sr(sr)
        if sr_schema_used:
            valid_count = len(sr_schema_used & available_schema)
            score = valid_count / len(sr_schema_used)
        else:
            score = 1.0

        stats['alignment_scores'].append(score)
        s = deepcopy(sample)
        s['alignment_score'] = round(score, 4)
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

def preprocess_sam(input_path: str, output_path: str):
    print("=" * 50)
    print("SAM PREPROCESSING PIPELINE")
    print("=" * 50)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n[LOAD] {len(data)} samples loaded from {input_path}")

    # Stage 1: Cleaning
    data, s1 = clean_sam(data)
    print(f"\n[STAGE 1] Data Cleaning")
    print(f"  Original         : {s1['original']}")
    print(f"  Removed null     : {s1['removed_null']}")
    print(f"  Removed duplicate: {s1['removed_duplicate']}")
    print(f"  After cleaning   : {s1['after_cleaning']}")

    # Stage 2: Normalization
    data, s2 = normalize_sam(data)
    print(f"\n[STAGE 2] Format Normalization")
    print(f"  Input            : {s2['input']}")
    print(f"  Removed invalid format: {s2['removed_invalid_format']}")
    print(f"  After normalization: {s2['after_normalization']}")

    # Stage 3: Anomaly Filtering
    data, s3 = filter_anomalies_sam(data)
    print(f"\n[STAGE 3] Anomalous Data Filtering")
    print(f"  Input                   : {s3['input']}")
    print(f"  Removed no masked SR    : {s3['removed_no_masked_sr']}")
    print(f"  Removed no SR in output : {s3['removed_no_sr_output']}")
    print(f"  Removed [MASK] remaining: {s3['removed_mask_remaining']}")
    print(f"  Removed invalid schema  : {s3['removed_schema_invalid']}")
    print(f"  After filter            : {s3['after_filter']}")

    # Stage 4: Validation
    data, s4 = validate_sam(data)
    print(f"\n[STAGE 4] Structure & Consistency Validation")
    print(f"  Input                        : {s4['input']}")
    print(f"  Removed no res               : {s4['removed_no_res']}")
    print(f"  Removed op count mismatch    : {s4['removed_operation_count_mismatch']}")
    print(f"  Avg alignment score          : {s4['avg_alignment_score']}")
    print(f"  After validation             : {s4['after_validation']}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(data)} samples saved to {output_path}")
    print(f"[SUMMARY] {s1['original']} → {len(data)} samples retained")
    print("=" * 50)


if __name__ == '__main__':
    preprocess_sam(
        input_path='./dataset/sam_train_data.json',
        output_path='./output/sam_train_data.json'
    )