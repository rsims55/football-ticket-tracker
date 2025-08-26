import csv, sys, os, collections
CSV = r'C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\data\daily\price_snapshots.csv'
today = r'2025-08-26'
print('FILE:', CSV)
print('EXISTS:', os.path.isfile(CSV))
rows = 0
by_date = collections.Counter()
examples = []
with open(CSV, 'r', encoding='utf-8', newline='') as f:
    r = csv.reader(f)
    header = next(r, None)
    # Detect header (if present) by looking for any alpha in the last 2 cells
    if header and any(c.isalpha() for c in ''.join(header[-2:])):
        # header was read as a data row; rewind
        f.seek(0); r = csv.reader(f); header = None
    for row in r:
        rows += 1
        if len(row) >= 2:
            snap_date = row[-2].strip()
            by_date[snap_date] += 1
            if snap_date == today and len(examples) < 5:
                examples.append(row)
print('Total data rows:', rows)
print('Top snapshot dates:', by_date.most_common(5))
print(f"Rows with snapshot_date == {today}:", by_date[today])
for i, ex in enumerate(examples, 1):
    print(f'example[{i}]:', ex[:6] + ex[-4:])  # show a few key fields
