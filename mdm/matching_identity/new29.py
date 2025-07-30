import pandas as pd
import recordlinkage
import networkx as nx
import sqlite3
import yaml
from collections import Counter
from datetime import datetime
from fuzzywuzzy import fuzz

# Suppress FutureWarning for replace
pd.set_option('future.no_silent_downcasting', True)

# variables
DB_PATH = 'D:\\mygit\\OpenMDM\\database\\'
DB_NAME = 'dbt_etl_dev.db'
TIMEIS = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
OUTPUT_PATH = 'D:\\mygit\\OpenMDM\\mdm\\output\\'
CONFIG_PATH = 'D:\\mygit\\OpenMDM\\mdm\\matching_identity\\config_mdm29.yml'

# Load configuration from YAML
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
blocking_columns = config['blocking']['columns']
similarity_configs = config['similarity']
thresholds = config['thresholds']
survivorship_rules = {rule['column']: rule['strategy'] for rule in config['survivorship']['rules']}
# date_column = yaml_config['survivorship']['rules'][0]['column']

priority_rule = config['priority_rule']
date_column = config['survivorship']['rules'][0]['column']

# Validate priority_rule
if not priority_rule or 'conditions' not in priority_rule:
    raise ValueError("YAML must contain 'priority_rule' with 'conditions' key.")
for condition in priority_rule['conditions']:
    if 'column' not in condition or 'value' not in condition:
        raise ValueError(f"Each condition in priority_rule must have 'column' and 'value'. Got: {condition}")

# 1. Load and prepare data
FULL_DB_PATH = f"{DB_PATH}{DB_NAME}"
conn = sqlite3.connect(FULL_DB_PATH)
df = pd.read_sql_query("SELECT * FROM main.slvr_personal_info", conn)
df.reset_index(drop=True, inplace=True)
df.index.name = 'record_id'
# print("Available columns:", df.columns.tolist())
# print("DataFrame head:\n", df[['first_name', 'last_name'] + [c for c in df.columns if c not in ['first_name', 'last_name']]].to_string())
# print("Null values in blocking columns:\n", df[blocking_columns].isnull().sum())
# print("Data types in blocking columns:\n", df[blocking_columns].dtypes)
# print("Unique first_name values:\n", df['first_name'].unique())
# print("Unique last_name values:\n", df['last_name'].unique())
conn.close()

# Validate priority columns
for condition in priority_rule['conditions']:
    if condition['column'] not in df.columns:
        raise ValueError(f"Priority column '{condition['column']}' not found in DataFrame. Available columns: {df.columns.tolist()}")

# 2. Blocking
# Clean and make blocking case-insensitive

# indexer = recordlinkage.Index()
# indexer.block(blocking_columns)
# candidate_pairs = indexer.index(df, df)
# # Remove duplicates and self-pairs
# candidate_pairs = candidate_pairs[candidate_pairs.get_level_values(0) < candidate_pairs.get_level_values(1)]
# # print(f"🧮 Total candidate pairs after blocking: {len(candidate_pairs)}")
# # print("Candidate pairs:", list(candidate_pairs))

# Create a new column by concatenating the specified columns
df['concat_column'] = df[blocking_columns].apply(lambda x: ' '.join(x.astype(str)).lower(), axis=1)

# Define a function to compute fuzzy similarity
def fuzzy_match(row1, row2, column='concat_column', threshold=80):
    """
    Compute fuzzy match score between two rows for the specified column.
    Returns True if score is above threshold, False otherwise.
    """
    score = fuzz.token_sort_ratio(row1[column], row2[column])
    return score >= threshold

# Initialize recordlinkage indexer
indexer = recordlinkage.Index()

# Custom blocking based on fuzzy matching
# Since recordlinkage doesn't natively support fuzzy blocking, we implement a custom approach
candidate_pairs = []

# Compare all pairs (this can be optimized for large datasets)
for i, row1 in df.iterrows():
    for j, row2 in df[df.index > i].iterrows():  # Avoid self-pairs and duplicates
        if fuzzy_match(row1, row2, 'concat_column', threshold=80):
            candidate_pairs.append((i, j))

# Convert candidate pairs to MultiIndex
candidate_pairs = pd.MultiIndex.from_tuples(candidate_pairs, names=['first', 'second'])
# Print the number of candidate pairs
print(f"🧮 Total candidate pairs after fuzzy blocking: {len(candidate_pairs)}")



# 3. Pairwise similarity
compare = recordlinkage.Compare()
for sim in similarity_configs:
    column = sim['column']
    method = sim['method']
    if method == 'exact':
        compare.exact(column, column, label=f"{column}_match")
    else:
        compare.string(column, column, method=method, label=f"{column}_sim")

# Use the MultiIndex directly, not the DataFrame conversion
features = compare.compute(candidate_pairs, df, df)

# # Debug missing pairs
# missing_pairs = [(0, 1), (3, 4)]
# for pair in missing_pairs:
#     if pair in features.index:
#         # print(f"Detailed similarity for {pair}:\n", features.loc[pair].to_dict())
#         pass
#     else:
#         print(f"Pair {pair} not in candidate pairs.")

# 4. Calculate score
similarity_labels = [f"{sim['column']}_{'match' if sim['method'] == 'exact' else 'sim'}" for sim in similarity_configs]
features['score'] = features[similarity_labels].mean(axis=1)
# print("Similarity scores:\n", features[[f"{sim['column']}_{'match' if sim['method'] == 'exact' else 'sim'}" for sim in similarity_configs] + ['score']].to_dict())

# 5. Categorize based on thresholds
features['match_category'] = pd.cut(
    features['score'],
    bins=[0, thresholds['review'], thresholds['auto_merge'], 1.0],
    labels=['non_match', 'review', 'auto_merge']
)

# 6. Apply priority survivorship rule
def apply_priority_rule(df, record_ids, priority_conditions):
    records = [df.loc[rid] for rid in record_ids]
    
    # Convert priority columns to match expected value type
    for condition in priority_conditions:
        column = condition['column']
        expected_value = condition['value']
        
        # Skip conversion if column doesn't exist
        if column not in df.columns:
            continue
            
        # Handle numeric or string values, but be more careful about conversion
        try:
            # Only convert if the column contains values that can be converted
            # Check if all non-null values are numeric-like
            non_null_values = df[column].dropna().astype(str)
            
            # Check if values look like they should be converted to the expected type
            if isinstance(expected_value, int):
                # Only convert if values look like integers (not dates)
                convertible_values = []
                for val in non_null_values:
                    # Skip date-like patterns
                    if '/' in str(val) or '-' in str(val) and len(str(val)) > 4:
                        continue
                    try:
                        int(val)
                        convertible_values.append(val)
                    except ValueError:
                        continue
                
                # Only proceed with conversion if we have convertible values
                if convertible_values:
                    df[column] = df[column].astype(str).replace(
                        {'0': 0, '1': 1, '0.0': 0, '1.0': 1, str(expected_value): expected_value}
                    )
                    # Convert only the convertible values
                    mask = df[column].astype(str).isin(['0', '1', '0.0', '1.0', str(expected_value)])
                    df.loc[mask, column] = df.loc[mask, column].astype(type(expected_value))
            
        except (ValueError, TypeError) as e:
            # Silently continue if conversion fails - this is expected for some columns
            continue
    
    # Check each condition in order
    for condition in priority_conditions:
        column = condition['column']
        expected_value = condition['value']
        
        if column not in df.columns:
            continue
            
        values = [rec[column] for rec in records]
        if values.count(expected_value) == 1:
            return record_ids[values.index(expected_value)]
    
    return None

# 7. Group records into clusters
G = nx.Graph()
auto_merge_pairs = [(row.name[0], row.name[1]) for _, row in features[features['match_category'] == 'auto_merge'].iterrows()]
review_pairs = [(row.name[0], row.name[1]) for _, row in features[features['match_category'] == 'review'].iterrows()]
# print("Auto-merge pairs:", auto_merge_pairs)
# print("Review pairs:", review_pairs)

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
# print("Clusters (auto_merge and unmatched only):", [list(cluster) for cluster in clusters])

# # 8. Create golden record for each cluster
# def create_golden_record(df, record_ids, survivorship_rules, priority_conditions):
#     trusted_id = apply_priority_rule(df, record_ids, priority_conditions) if len(record_ids) > 1 else record_ids[0]
#     golden = {}
    
#     if trusted_id:
#         golden = df.loc[trusted_id].to_dict()
#     else:
#         for column in df.columns:
#             if column in survivorship_rules:
#                 strategy = survivorship_rules[column]
#                 values = [df.loc[rid][column] for rid in record_ids]
#                 if strategy == 'most_common':
#                     most_common = Counter([v for v in values if pd.notna(v)]).most_common(1)
#                     golden[column] = most_common[0][0] if most_common else None
#                 elif strategy == 'longest_if_null':
#                     non_null_values = [v for v in values if pd.notna(v)]
#                     if non_null_values:
#                         most_common = Counter(non_null_values).most_common(1)
#                         golden[column] = most_common[0][0] if most_common else None
#                     else:
#                         # Fall back to longest string from all records
#                         all_values = []
#                         for rid in record_ids:
#                             val = df.loc[rid][column]
#                             all_values.append((val, len(str(val)) if pd.notna(val) else 0))
#                         if all_values:
#                             max_len_val = max(all_values, key=lambda x: x[1])[0]
#                             golden[column] = max_len_val if pd.notna(max_len_val) else None
#                         else:
#                             golden[column] = None
#                 elif strategy == 'most_recent':
#                     dates = [pd.to_datetime(df.loc[rid]['_load_datetime']) if '_load_datetime' in df.columns else pd.Timestamp.min for rid in record_ids]
#                     max_date_idx = dates.index(max(dates))
#                     golden[column] = values[max_date_idx]
#                 else:
#                     golden[column] = values[0]
#             else:
#                 golden[column] = df.loc[record_ids[0]][column]
#     return golden, trusted_id

def create_golden_record(df, record_ids, survivorship_rules, priority_conditions):
    trusted_id = apply_priority_rule(df, record_ids, priority_conditions) if len(record_ids) > 1 else record_ids[0]
    golden = {}
    
    # Get the date column from survivorship_rules where strategy is 'most_recent'
    date_column = next((col for col, strategy in survivorship_rules.items() if strategy == 'most_recent'), None)
    
    if trusted_id:
        return df.loc[trusted_id].to_dict(), trusted_id
    
    for column in df.columns:
        values = [df.loc[rid][column] for rid in record_ids]
        non_null_values = [v for v in values if pd.notna(v)]
        
        if non_null_values:
            if date_column and column.lower() == date_column.lower():
                dates = [pd.to_datetime(df.loc[rid][column]) if pd.notna(df.loc[rid][column]) else pd.Timestamp.min for rid in record_ids]
                max_date_idx = dates.index(max(dates))
                golden[column] = values[max_date_idx]
            else:
                golden[column] = non_null_values[0]
        else:
            golden[column] = None
            
    return golden, trusted_id


# 9. Write pairwise summary
def write_pairwise_summary(df, features, category, output_path):
    subset = features[features['match_category'] == category].reset_index()
    if subset.empty:
        # print(f"No pairs found for category: {category}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f'\n--- {category.upper()} RECORDS ---\n')
            f.write(f"No {category} pairs found.\n")
            f.write('-' * 80 + '\n')
        return
    # The MultiIndex columns are named 'first' and 'second' after reset_index()
    subset[['id_min', 'id_max']] = subset[['first', 'second']].apply(sorted, axis=1, result_type='expand')
    subset = subset.drop_duplicates(subset=['id_min', 'id_max'])
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'\n--- {category.upper()} RECORDS ---\n')
        for _, row in subset.iterrows():
            id1, id2 = row['first'], row['second']
            trusted_id = apply_priority_rule(df, [id1, id2], priority_rule['conditions'])
            trusted_status = f"Trusted Record: {trusted_id}" if trusted_id else "Trusted Record: None (no priority match)"
            f.write(f"🔹 Record 1 ({id1}): {df.loc[id1].to_dict()}\n")
            f.write(f"🔸 Record 2 ({id2}): {df.loc[id2].to_dict()}\n")
            f.write(f"💡 Similarity Score: {round(row['score'], 4)} | Match Category: {category}\n")
            f.write(f"🏆 {trusted_status}\n")
            f.write('-' * 80 + '\n')

# 10. Write single records summary
def write_single_records_summary(df, unmatched_records, output_path):
    if not unmatched_records:
        # print("No single records found.")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f'\n--- SINGLE RECORDS ---\n')
            f.write("No single records found.\n")
            f.write('-' * 80 + '\n')
        return
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'\n--- SINGLE RECORDS ---\n')
        for record_id in sorted(unmatched_records):
            record = df.loc[record_id].to_dict()
            f.write(f"🔶 Single Record ({record_id}):\n")
            f.write(f"{record}\n")
            f.write(f"Source Record: {record_id}\n")
            f.write(f"Trusted Record: {record_id}\n")
            f.write('-' * 80 + '\n')

# 11. Write golden records
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
write_single_records_summary(df, unmatched_records, summary_path)
write_golden_records(df, clusters, golden_path)
print(f"📊 MDM Summary Report saved here: {summary_path}")
print(f"📈 Golden Records saved here: {golden_path}")
