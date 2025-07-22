import pandas as pd
import recordlinkage
import networkx as nx
import sqlite3
import yaml
from collections import Counter
from datetime import datetime

# variables
DB_PATH = 'D:\\mygit\\OpenMDM\\database\\'
DB_NAME = 'dbt_etl_dev.db'
TIMEIS = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
OUTPUT_PATH = 'D:\\mygit\\OpenMDM\\mdm\\output\\'
CONFIG_PATH = 'D:\\mygit\\OpenMDM\\mdm\\matching_identity\\config_mdm.yml'

# Load configuration from YAML
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
blocking_columns = config['blocking']['columns']
similarity_configs = config['similarity']
thresholds = config['thresholds']
survivorship_rules = {rule['column']: rule['strategy'] for rule in config['survivorship']['rules']}
priority_rule = config.get('priority_rule', {'conditions': [
    {'column': 'original', 'value': 1},
    {'column': 'COMDL', 'value': 1},
    {'column': 'REAL', 'value': 1}
]})

# 1. Load and prepare data
FULL_DB_PATH = f"{DB_PATH}{DB_NAME}"
conn = sqlite3.connect(FULL_DB_PATH)
df = pd.read_sql_query("SELECT * FROM main.slvr_personal_info", conn)
df.reset_index(drop=True, inplace=True)
df.index.name = 'record_id'
print("Available columns:", df.columns.tolist())
print("DataFrame head:\n", df[['first_name', 'last_name'] + [c for c in df.columns if c not in ['first_name', 'last_name']]].to_string())
print("Null values in blocking columns:\n", df[blocking_columns].isnull().sum())
print("Data types in blocking columns:\n", df[blocking_columns].dtypes)
conn.close()

# Validate priority columns
if 'conditions' not in priority_rule:
    raise ValueError(f"Expected 'conditions' in priority_rule, got: {priority_rule}")
for condition in priority_rule['conditions']:
    if condition['column'] not in df.columns:
        raise ValueError(f"Priority column '{condition['column']}' not found in DataFrame. Available columns: {df.columns.tolist()}")

# 2. Blocking
# Clean and make blocking case-insensitive
for col in blocking_columns:
    df[col] = df[col].astype(str).str.lower().str.strip().fillna('')
# Fix 'Keit' to 'Keith' in last_name
df['last_name'] = df['last_name'].replace('keit', 'keith')
print("Blocking column values:\n", df[blocking_columns].to_string())

indexer = recordlinkage.Index()
indexer.block(blocking_columns)
candidate_pairs = indexer.index(df, df)
# Remove duplicates and self-pairs
candidate_pairs = candidate_pairs[candidate_pairs.get_level_values(0) < candidate_pairs.get_level_values(1)]
print(f"🧮 Total candidate pairs after blocking: {len(candidate_pairs)}")
print("Candidate pairs:", list(candidate_pairs))

# 3. Pairwise similarity
compare = recordlinkage.Compare()
for sim in similarity_configs:
    column = sim['column']
    method = sim['method']
    if method == 'exact':
        compare.exact(column, column, label=f"{column}_match")
    else:
        compare.string(column, column, method=method, label=f"{column}_sim")

features = compare.compute(candidate_pairs, df, df)

# 4. Calculate score
similarity_labels = [f"{sim['column']}_{'match' if sim['method'] == 'exact' else 'sim'}" for sim in similarity_configs]
features['score'] = features[similarity_labels].mean(axis=1)
print("Similarity scores:\n", features[[f"{sim['column']}_{'match' if sim['method'] == 'exact' else 'sim'}" for sim in similarity_configs] + ['score']].to_dict())

# 5. Categorize based on thresholds
features['match_category'] = pd.cut(
    features['score'],
    bins=[0, thresholds['review'], thresholds['auto_merge'], 1.0],
    labels=['non_match', 'review', 'auto_merge']
)

# 6. Apply priority survivorship rule
def apply_priority_rule(df, record_ids, priority_conditions):
    records = [df.loc[rid] for rid in record_ids]
    
    # Convert to int for comparison
    for condition in priority_conditions:
        column = condition['column']
        df[column] = df[column].astype(str).replace({'0': 0, '1': 1, '0.0': 0, '1.0': 1}).astype(int)
    
    # Check original
    original_values = [int(rec['original']) for rec in records]
    if original_values.count(1) == 1:
        return record_ids[original_values.index(1)]
    
    # If original values are equal, check COMDL
    comdl_values = [int(rec['COMDL']) for rec in records]
    if comdl_values.count(1) == 1:
        return record_ids[comdl_values.index(1)]
    
    # If COMDL values are equal, check REAL
    real_values = [int(rec['REAL']) for rec in records]
    if real_values.count(1) == 1:
        return record_ids[real_values.index(1)]
    
    return None

# 7. Group records into clusters
G = nx.Graph()
auto_merge_pairs = [(row.name[0], row.name[1]) for _, row in features[features['match_category'] == 'auto_merge'].iterrows()]
review_pairs = [(row.name[0], row.name[1]) for _, row in features[features['match_category'] == 'review'].iterrows()]
print("Auto-merge pairs:", auto_merge_pairs)
print("Review pairs:", review_pairs)

# Only use auto_merge pairs for clustering
G.add_edges_from(auto_merge_pairs)
clusters = list(nx.connected_components(G))

# Add single-record clusters for unmatched records, excluding those in review pairs
review_records = set().union(*[(i, j) for i, j in review_pairs])
all_records = set(df.index)
matched_records = set().union(*clusters)
unmatched_records = all_records - matched_records - review_records
for record_id in unmatched_records:
    clusters.append({record_id})
print("Clusters (auto_merge and unmatched only):", [list(cluster) for cluster in clusters])

# 8. Create golden record for each cluster
def create_golden_record(df, record_ids, survivorship_rules, priority_conditions):
    trusted_id = apply_priority_rule(df, record_ids, priority_conditions) if len(record_ids) > 1 else record_ids[0]
    golden = {}
    
    if trusted_id:
        golden = df.loc[trusted_id].to_dict()
    else:
        for column in df.columns:
            if column in survivorship_rules:
                strategy = survivorship_rules[column]
                values = [df.loc[rid][column] for rid in record_ids]
                if strategy == 'most_common':
                    most_common = Counter([v for v in values if pd.notna(v)]).most_common(1)
                    golden[column] = most_common[0][0] if most_common else None
                elif strategy == 'most_recent':
                    dates = [pd.to_datetime(df.loc[rid]['_load_datetime']) if '_load_datetime' in df.columns else pd.Timestamp.min for rid in record_ids]
                    max_date_idx = dates.index(max(dates))
                    golden[column] = values[max_date_idx]
                else:
                    golden[column] = values[0]
            else:
                golden[column] = df.loc[record_ids[0]][column]
    return golden, trusted_id

# 9. Write pairwise summary
def write_pairwise_summary(df, features, category, output_path):
    subset = features[features['match_category'] == category].reset_index()
    if subset.empty:
        print(f"No pairs found for category: {category}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f'\n--- {category.upper()} RECORDS ---\n')
            f.write(f"No {category} pairs found.\n")
            f.write('-' * 80 + '\n')
        return
    subset[['id_min', 'id_max']] = subset[['record_id_1', 'record_id_2']].apply(sorted, axis=1, result_type='expand')
    subset = subset.drop_duplicates(subset=['id_min', 'id_max'])
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'\n--- {category.upper()} RECORDS ---\n')
        for _, row in subset.iterrows():
            id1, id2 = row['record_id_1'], row['record_id_2']
            trusted_id = apply_priority_rule(df, [id1, id2], priority_rule['conditions'])
            trusted_status = f"Trusted Record: {trusted_id}" if trusted_id else "Trusted Record: None (no priority match)"
            f.write(f"🔹 Record 1 ({id1}): {df.loc[id1].to_dict()}\n")
            f.write(f"🔸 Record 2 ({id2}): {df.loc[id2].to_dict()}\n")
            f.write(f"💡 Similarity Score: {round(row['score'], 4)} | Match Category: {category}\n")
            f.write(f"🏆 {trusted_status}\n")
            f.write('-' * 80 + '\n')

# 10. Write golden records
def write_golden_records(df, clusters, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('--- GOLDEN RECORDS ---\n')
        for idx, cluster in enumerate(clusters):
            record_ids = list(cluster)
            golden, trusted_id = create_golden_record(df, record_ids, survivorship_rules, priority_rule['conditions'])
            f.write(f"🔶 Golden Record for Cluster {idx + 1} (Records: {', '.join(map(str, record_ids))}):\n")
            f.write(f"{golden}\n")
            f.write(f"Source Records: {', '.join(map(str, record_ids))}\n")
            f.write(f"Trusted Record: {trusted_id if trusted_id else 'None (survivorship applied)' if len(record_ids) > 1 else str(record_ids[0])}\n")
            f.write('-' * 80 + '\n')

# Write outputs
summary_path = f"{OUTPUT_PATH}detail_summary_{TIMEIS}.txt"
golden_path = f"{OUTPUT_PATH}golden_records_{TIMEIS}.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('')  # Clear file
write_pairwise_summary(df, features, 'auto_merge', summary_path)
write_pairwise_summary(df, features, 'review', summary_path)
write_golden_records(df, clusters, golden_path)
print(f"📊 MDM Summary Report saved here: {summary_path}")
print(f"📈 Golden Records saved here: {golden_path}")