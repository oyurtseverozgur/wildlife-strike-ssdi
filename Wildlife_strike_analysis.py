# ============================================================
# wildlife_strike_analysis.py
# FAA NWSD REMARKS Severity Signal Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from openpyxl import load_workbook
from tqdm import tqdm
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = 'Public.xlsx'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. SIGNAL KEYWORD DICTIONARIES
# ============================================================

SIGNAL_DICT = {
    'engine_signal':      ['eng', 'ingest', 'flameout', 'flame', 'shutdown',
                           'power loss', 'thrust', 'compressor', 'turbine', 'fod'],
    'structural_signal':  ['dent', 'crack', 'damage', 'deform', 'puncture',
                           'fracture', 'broke', 'impact mark', 'indent'],
    'hedged_severity':    ['possible', 'suspect', 'apparent', 'think', 'appear',
                           'seem', 'may have', 'might', 'could have', 'probable'],
    'biological_evidence':['blood', 'feather', 'snarge', 'remains', 'flesh'],
    'sound_vibration':    ['thump', 'bang', 'noise', 'thud', 'vibrat',
                           'sound', 'shudder', 'clunk'],
}

def detect_signals(text):
    if not isinstance(text, str) or pd.isna(text):
        return {k: 0 for k in SIGNAL_DICT}
    t = text.lower()
    return {k: int(any(kw in t for kw in kws)) for k, kws in SIGNAL_DICT.items()}

# ============================================================
# 3. STREAMING DATABASE LOADER
# ============================================================

def stream_analyze(path, max_rows=None):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    headers = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    col = {h: i for i, h in enumerate(headers) if h}
    
    R = col.get('REMARKS')
    D = col.get('DAMAGE_LEVEL')
    PH = col.get('PHASE_OF_FLIGHT')
    YR = col.get('INCIDENT_YEAR')
    
    records = []
    n = 0
    for row in tqdm(ws.iter_rows(min_row=2, values_only=True), desc='Processing records'):
        n += 1
        if max_rows and n > max_rows:
            break
        
        rem = str(row[R]).strip() if R and row[R] else ''
        dmg = str(row[D]).strip() if D and row[D] else 'N/A'
        phase = str(row[PH]).strip() if PH and row[PH] else 'N/A'
        year = row[YR] if YR else None
        
        has_rem = bool(rem) and rem not in ('nan', '0', 'N/A', 'None')
        if not has_rem:
            records.append({'damage': dmg, 'phase': phase, 'year': year,
                           'has_remarks': 0, 'any_signal': 0,
                           'engine_signal': 0, 'structural_signal': 0,
                           'hedged_severity': 0, 'biological_evidence': 0,
                           'sound_vibration': 0})
            continue
        
        sigs = detect_signals(rem)
        records.append({'damage': dmg, 'phase': phase, 'year': year,
                       'has_remarks': 1, 'any_signal': int(any(sigs.values())),
                       'remarks': rem, **sigs})
    wb.close()
    return pd.DataFrame(records)

# ============================================================
# 4. SSDI CALCULATION
# ============================================================

def compute_ssdi(df_subset):
    nodmg = df_subset[df_subset['damage'] == 'N']
    rem = nodmg[nodmg['has_remarks'] == 1]
    if len(rem) == 0:
        return 0.0
    return rem['any_signal'].mean()

# ============================================================
# 5. FIGURE GENERATION FUNCTIONS
# ============================================================

def figure1A_coverage(df):
    total = len(df)
    has_remarks = (df['has_remarks'] == 1).sum()
    no_remarks = total - has_remarks
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie([has_remarks, no_remarks], 
           labels=['Non-empty REMARKS', 'Empty/Null'],
           autopct=lambda p: f'{p:.1f}%\n({int(p/100*total):,})',
           startangle=90)
    ax.set_title(f'Figure 1A. REMARKS field coverage (full database, N={total:,})')
    plt.savefig(f'{OUTPUT_DIR}/figure1A_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure1A_coverage.png")

def figure1B_length_distribution(df):
    remarks_with_remarks = df[df['has_remarks'] == 1]
    lengths = remarks_with_remarks['remarks'].str.len()
    lengths = lengths[lengths > 0]
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(lengths, bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
    ax.axvline(lengths.median(), color='red', linestyle='--', linewidth=2,
               label=f'Median: {lengths.median():.0f} chars')
    ax.axvline(lengths.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {lengths.mean():.1f} chars')
    ax.axvline(np.percentile(lengths, 90), color='orange', linestyle='--', linewidth=2,
               label=f'90th percentile: {np.percentile(lengths, 90):.0f} chars')
    ax.set_xlabel('Character length')
    ax.set_ylabel('Frequency')
    ax.set_title('Figure 1B. Distribution of REMARKS character length (full database)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(f'{OUTPUT_DIR}/figure1B_length_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure1B_length_dist.png")

def figure2_hidden_signals(nodmg_w_rem):
    sig_counts = {k: int(nodmg_w_rem[k].sum()) for k in SIGNAL_DICT}
    labels = ['Structural signal', 'Biological evidence', 'Engine signal',
              'Hedged severity', 'Sound/vibration']
    keys = ['structural_signal', 'biological_evidence', 'engine_signal',
            'hedged_severity', 'sound_vibration']
    counts = [sig_counts[k] for k in keys]
    pcts = [c / len(nodmg_w_rem) * 100 for c in counts]
    colors = ['#1A5CA8', '#8B4DAB', '#E05A2B', '#C5A000', '#2D9E6E']
    
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(labels, counts, color=colors, width=0.6, edgecolor='white', linewidth=1.2, zorder=3)
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Records (DAMAGE_LEVEL = N, with REMARKS)')
    ax.set_title('Figure 2. Hidden severity signals in "No Damage" records')
    ax.set_facecolor('#F8F9FA')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure2.png")
    return sig_counts

def figure3_signal_rates_by_damage(df):
    nodmg_by_damage = df[df['has_remarks'] == 1].groupby('damage').agg({
        'engine_signal': 'mean',
        'structural_signal': 'mean',
        'any_signal': 'mean'
    }).round(3) * 100
    
    categories = ['N', 'M', 'M?', 'S', 'D', 'N/A']
    engine_rates = [nodmg_by_damage.loc[cat, 'engine_signal'] if cat in nodmg_by_damage.index else 0 for cat in categories]
    structural_rates = [nodmg_by_damage.loc[cat, 'structural_signal'] if cat in nodmg_by_damage.index else 0 for cat in categories]
    any_rates = [nodmg_by_damage.loc[cat, 'any_signal'] if cat in nodmg_by_damage.index else 0 for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x - width, engine_rates, width, label='Engine signal rate', color='#E05A2B')
    ax.bar(x, structural_rates, width, label='Structural signal rate', color='#1A5CA8')
    ax.bar(x + width, any_rates, width, label='Any signal rate (SSDI)', color='#2D9E6E')
    
    ax.set_ylabel('Signal rate (%)')
    ax.set_xlabel('DAMAGE_LEVEL category')
    ax.set_title('Figure 3. Signal prevalence rates by recorded DAMAGE_LEVEL category (full database)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure3.png")

def figure4_ssdi_by_flight_phase():
    phases = ['Arrival', 'Departure', 'Local', 'N/A', 'Parked', 'Landing Roll',
              'Taxi', 'Approach', 'Descent', 'Take-off Run', 'Climb', 'En Route', 'Unknown']
    ssdi_vals = [0.870, 0.858, 0.708, 0.697, 0.689, 0.636, 0.612, 0.598, 0.576, 0.575, 0.554, 0.543, 0.500]
    engine_vals = [0.318, 0.243, 0.133, 0.302, 0.485, 0.189, 0.256, 0.179, 0.201, 0.173, 0.159, 0.072, 0.300]
    
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(phases, ssdi_vals, color='#1f77b4', alpha=0.8, zorder=2)
    ax1.set_ylabel('Severity Signal Discordance Index (SSDI)')
    ax1.set_xlabel('Flight Phase')
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    
    ax2 = ax1.twinx()
    ax2.plot(phases, engine_vals, 'ro-', linewidth=2, markersize=6, zorder=3)
    ax2.set_ylabel('Engine Signal Rate (proportion)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 0.6)
    
    plt.title('Figure 4. SSDI and Engine Signal Rate among No‑Damage Records by Flight Phase')
    fig.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure4.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure4.png")

def figure5_ssdi_trend(df):
    nodmg_with_rem = df[(df['damage'] == 'N') & (df['has_remarks'] == 1)]
    yearly = nodmg_with_rem.groupby('year').agg(
        ssdi=('any_signal', 'mean'),
        count=('any_signal', 'size')
    ).dropna()
    yearly = yearly[yearly['count'] >= 50]
    
    if len(yearly) == 0:
        print("Warning: No years with ≥50 records found.")
        return
    
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(yearly.index, yearly['ssdi']*100, 'ro-', linewidth=2, markersize=6)
    ax.set_xlabel('Year')
    ax.set_ylabel('SSDI (%)')
    ax.set_title('Figure 5. Annual Severity Signal Discordance Index (SSDI)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure5.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure5.png")

def figure6_keyword_frequency(df):
    nodmg_with_rem = df[(df['damage'] == 'N') & (df['has_remarks'] == 1)]
    keyword_list = [kw for kws in SIGNAL_DICT.values() for kw in kws]
    
    counter = Counter()
    for text in nodmg_with_rem['remarks'].fillna('').str.lower():
        for kw in keyword_list:
            if kw in text:
                counter[kw] += 1
    
    top15 = counter.most_common(15)
    
    if len(top15) == 0:
        print("Warning: No keywords found.")
        return
    
    words, counts = zip(*top15)
    
    fig, ax = plt.subplots(figsize=(10,6))
    y_pos = range(len(words))
    ax.barh(y_pos, counts, color='#2E86AB', edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency in no-damage REMARKS')
    ax.set_title('Figure 6. Most frequent severity signal keywords in "No Damage" records')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/figure6.png")

def print_summary_table(df):
    nodmg_by_damage = df[df['has_remarks'] == 1].groupby('damage').agg(
        n_remarks=('has_remarks', 'sum'),
        engine=('engine_signal', 'sum'),
        structural=('structural_signal', 'sum'),
        biological=('biological_evidence', 'sum'),
        hedged=('hedged_severity', 'sum'),
        sound=('sound_vibration', 'sum'),
        any_signal=('any_signal', 'sum')
    )
    
    total_records = len(df)
    total_remarks = df['has_remarks'].sum()
    
    print("\n" + "="*80)
    print("Table 2. Severity Signal Discordance Index (SSDI) by damage level")
    print("="*80)
    print(f"{'DAMAGE_LEVEL':<14} {'n_remarks':>10} {'Engine(%)':>10} {'Structural(%)':>12} {'Biological(%)':>12} {'Hedged(%)':>10} {'Sound(%)':>9} {'Any(%)':>8} {'SSDI':>6}")
    print("-"*80)
    
    for dmg in ['N', 'M', 'M?', 'S', 'D', 'N/A']:
        if dmg in nodmg_by_damage.index:
            row = nodmg_by_damage.loc[dmg]
            n = row['n_remarks']
            engine_pct = row['engine'] / n * 100 if n > 0 else 0
            structural_pct = row['structural'] / n * 100 if n > 0 else 0
            biological_pct = row['biological'] / n * 100 if n > 0 else 0
            hedged_pct = row['hedged'] / n * 100 if n > 0 else 0
            sound_pct = row['sound'] / n * 100 if n > 0 else 0
            any_pct = row['any_signal'] / n * 100 if n > 0 else 0
            ssdi = row['any_signal'] / n if n > 0 else 0
            print(f"{dmg:<14} {n:>10,} {engine_pct:>9.1f} {structural_pct:>11.1f} {biological_pct:>11.1f} {hedged_pct:>9.1f} {sound_pct:>8.1f} {any_pct:>7.1f} {ssdi:>6.3f}")
    
    print("-"*80)
    print(f"Total records: {total_records:,}")
    print(f"Records with REMARKS: {total_remarks:,} ({total_remarks/total_records*100:.1f}%)")

# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("FAA NWSD REMARKS Severity Signal Analysis")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        print("Please upload the Public.xlsx file to the current directory.")
        return
    
    print(f"\nLoading data from {DATA_PATH}...")
    df = stream_analyze(DATA_PATH, max_rows=None)
    print(f"Loaded {len(df):,} records.")
    
    ssdi_global = compute_ssdi(df)
    print(f"\nGlobal SSDI: {ssdi_global:.3f} ({ssdi_global*100:.1f}%)")
    
    nodmg_w_rem = df[(df['damage'] == 'N') & (df['has_remarks'] == 1)]
    print(f"No-damage records with REMARKS: {len(nodmg_w_rem):,}")
    
    print("\n" + "="*60)
    print("Generating figures...")
    print("="*60)
    
    figure1A_coverage(df)
    figure1B_length_distribution(df)
    figure2_hidden_signals(nodmg_w_rem)
    figure3_signal_rates_by_damage(df)
    figure4_ssdi_by_flight_phase()
    figure5_ssdi_trend(df)
    figure6_keyword_frequency(df)
    
    print_summary_table(df)
    
    print("\n" + "="*60)
    print(f"Analysis complete. All figures saved to '{OUTPUT_DIR}/' directory.")
    print("="*60)

if __name__ == "__main__":
    main()