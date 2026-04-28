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


def extract_sql_from_instruction(instruction: str) -> str | None:
    """Extract SQL string from instruction field."""
    match = re.search(r'sql\s*=\s*"(.*?)"', instruction, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_schema_from_instruction(instruction: str) -> list[str]:
    """Extract list of schema columns from instruction field."""
    match = re.search(r'schema\s*=\s*(\[.*?\])', instruction, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1).replace("'", '"'))
    except Exception:
        return re.findall(r"'([A-Za-z_]+\.[A-Za-z_]+)'", match.group(1))


def normalize_output_format(output: str) -> str | None:
    """
    Ensure output is wrapped in ```SR ... ```.
    Returns None if SR block not found and content is empty.
    """
    output = output.strip()
    if output.startswith('```SR'):
        return output
    # Try to wrap bare SR content
    if output and not output.startswith('```'):
        return f"```SR\n{output}\n```"
    return None


def get_sql_operations(sql: str) -> set[str]:
    """Extract key SQL operations from a SQL string."""
    sql_upper = sql.upper()
    ops = set()
    if 'WHERE' in sql_upper:
        ops.add('WHERE')
    if 'GROUP BY' in sql_upper:
        ops.add('GROUP BY')
    if 'ORDER BY' in sql_upper:
        ops.add('ORDER BY')
    if 'DISTINCT' in sql_upper:
        ops.add('DISTINCT')
    if re.search(r'\bCOUNT\s*\(', sql_upper):
        ops.add('COUNT')
    if re.search(r'\bSUM\s*\(', sql_upper):
        ops.add('SUM')
    if re.search(r'\bAVG\s*\(', sql_upper):
        ops.add('AVG')
    if re.search(r'\bMAX\s*\(', sql_upper):
        ops.add('MAX')
    if re.search(r'\bMIN\s*\(', sql_upper):
        ops.add('MIN')
    if 'HAVING' in sql_upper:
        ops.add('HAVING')
    if 'LIMIT' in sql_upper:
        ops.add('LIMIT')
    return ops


def get_sr_operations(sr: str) -> set[str]:
    """Extract key SR operations from SR string."""
    ops = set()
    if 'df.where' in sr or '.where(' in sr:
        ops.add('WHERE')
    if '.groupby(' in sr:
        ops.add('GROUP BY')
    if '.orderby(' in sr or '.sort(' in sr:
        ops.add('ORDER BY')
    if 'select_distinct' in sr or 'distinct(' in sr:
        ops.add('DISTINCT')
    if 'count(' in sr.lower():
        ops.add('COUNT')
    if 'sum(' in sr.lower():
        ops.add('SUM')
    if 'avg(' in sr.lower():
        ops.add('AVG')
    if 'max(' in sr.lower():
        ops.add('MAX')
    if 'min(' in sr.lower():
        ops.add('MIN')
    if '.having(' in sr:
        ops.add('HAVING')
    if '.limit(' in sr:
        ops.add('LIMIT')
    return ops


def extract_schema_used_in_sr(sr: str) -> set[str]:
    """Extract table.column references used inside SR."""
    return set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b', sr))


# ─────────────────────────────────────────────
# STAGE 1: DATA CLEANING
# ─────────────────────────────────────────────

def clean_bam(data: list[dict]) -> tuple[list[dict], dict]:
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

def normalize_bam(data: list[dict]) -> tuple[list[dict], dict]:
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

def filter_anomalies_bam(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        'input': len(data),
        'removed_no_sql': 0,
        'removed_no_sr': 0,
        'removed_operation_mismatch': 0,
        'removed_schema_mismatch': 0,
    }
    filtered = []

    for sample in data:
        instruction = sample['instruction']
        output = sample['output']

        # Extract SQL
        sql = extract_sql_from_instruction(instruction)
        if not sql:
            stats['removed_no_sql'] += 1
            continue

        # Extract SR
        sr = extract_sr_from_output(output)
        if not sr:
            stats['removed_no_sr'] += 1
            continue

        # Extract schema from instruction
        schema_list = extract_schema_from_instruction(instruction)

        # Check SQL operation → SR operation alignment
        sql_ops = get_sql_operations(sql)
        sr_ops = get_sr_operations(sr)
        missing_ops = sql_ops - sr_ops
        if missing_ops:
            stats['removed_operation_mismatch'] += 1
            continue

        # Check schema used in SR exists in available schema
        if schema_list:
            sr_schema_used = extract_schema_used_in_sr(sr)
            # Filter out SR internal variables like df1.where (not real schema)
            sr_schema_used = {
                s for s in sr_schema_used
                if not s.startswith('df') and not s.startswith('res')
            }
            invalid_schema = sr_schema_used - set(schema_list)
            if invalid_schema:
                stats['removed_schema_mismatch'] += 1
                continue

        filtered.append(sample)

    stats['after_filter'] = len(filtered)
    return filtered, stats


# ─────────────────────────────────────────────
# STAGE 4: STRUCTURE & CONSISTENCY VALIDATION
# ─────────────────────────────────────────────

def validate_bam(data: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        'input': len(data),
        'removed_no_res': 0,
        'removed_broken_chain': 0,
        'alignment_scores': [],
    }
    validated = []

    for sample in data:
        sr = extract_sr_from_output(sample['output'])
        if not sr:
            continue

        lines = [l.strip() for l in sr.strip().split('\n') if l.strip()]

        # Validate last line is res = ...
        if not lines[-1].startswith('res'):
            stats['removed_no_res'] += 1
            continue

        # Validate variable chaining (df1, df2, ... must be assigned before used)
        assigned_vars = {'df'}
        broken = False
        for line in lines:
            # Get right-hand side references
            rhs_vars = re.findall(r'\b(df\d*|res)\b', line.split('=', 1)[-1] if '=' in line else line)
            for v in rhs_vars:
                if v != 'df' and v not in assigned_vars:
                    broken = True
                    break
            # Get left-hand side assignment
            if '=' in line:
                lhs = line.split('=')[0].strip()
                assigned_vars.add(lhs)
            if broken:
                break

        if broken:
            stats['removed_broken_chain'] += 1
            continue

        # Compute alignment score: ratio of SQL ops covered by SR ops
        sql = extract_sql_from_instruction(sample['instruction'])
        if sql:
            sql_ops = get_sql_operations(sql)
            sr_ops = get_sr_operations(sr)
            score = len(sql_ops & sr_ops) / len(sql_ops) if sql_ops else 1.0
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

def preprocess_bam(input_path: str, output_path: str):
    print("=" * 50)
    print("BAM PREPROCESSING PIPELINE")
    print("=" * 50)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n[LOAD] {len(data)} samples loaded from {input_path}")

    # Stage 1: Cleaning
    data, s1 = clean_bam(data)
    print(f"\n[STAGE 1] Data Cleaning")
    print(f"  Original       : {s1['original']}")
    print(f"  Removed null   : {s1['removed_null']}")
    print(f"  Removed duplicate: {s1['removed_duplicate']}")
    print(f"  After cleaning : {s1['after_cleaning']}")

    # Stage 2: Normalization
    data, s2 = normalize_bam(data)
    print(f"\n[STAGE 2] Format Normalization")
    print(f"  Input          : {s2['input']}")
    print(f"  Removed invalid format: {s2['removed_invalid_format']}")
    print(f"  After normalization: {s2['after_normalization']}")

    # Stage 3: Anomaly Filtering
    data, s3 = filter_anomalies_bam(data)
    print(f"\n[STAGE 3] Anomalous Data Filtering")
    print(f"  Input                    : {s3['input']}")
    print(f"  Removed no SQL           : {s3['removed_no_sql']}")
    print(f"  Removed no SR            : {s3['removed_no_sr']}")
    print(f"  Removed operation mismatch: {s3['removed_operation_mismatch']}")
    print(f"  Removed schema mismatch  : {s3['removed_schema_mismatch']}")
    print(f"  After filter             : {s3['after_filter']}")

    # Stage 4: Validation
    data, s4 = validate_bam(data)
    print(f"\n[STAGE 4] Structure & Consistency Validation")
    print(f"  Input              : {s4['input']}")
    print(f"  Removed no res     : {s4['removed_no_res']}")
    print(f"  Removed broken chain: {s4['removed_broken_chain']}")
    print(f"  Avg alignment score: {s4['avg_alignment_score']}")
    print(f"  After validation   : {s4['after_validation']}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(data)} samples saved to {output_path}")
    print(f"[SUMMARY] {s1['original']} → {len(data)} samples retained")
    print("=" * 50)


if __name__ == '__main__':
    preprocess_bam(
        input_path='./dataset/bam_train_data.json',
        output_path='./output/bam_train_data.json'
    )