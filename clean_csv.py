import csv, os

# Clean CSV - keep only benchmark rows (seqlen >= 500)
csv_path = 'results/benchmark.csv'
rows = []
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['SeqLen']) >= 500:
            rows.append(row)

# Keep last occurrence of each size (most recent benchmark run)
seen = {}
for row in rows:
    seen[row['SeqLen']] = row

final = sorted(seen.values(), key=lambda r: int(r['SeqLen']))

fieldnames = ['SeqLen','CPU_ms','GPUKernel_ms','GPUTotal_ms',
              'SpeedupKernel','SpeedupTotal','CPU_Score','GPU_Score','Validation']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final)

print('CSV cleaned. Final rows:')
for r in final:
    sz  = r['SeqLen']
    cpu = float(r['CPU_ms'])
    gpu = float(r['GPUKernel_ms'])
    val = r['Validation']
    print(f'  SeqLen={sz:>5}  CPU={cpu:>8.2f}ms  Wavefront={gpu:>8.2f}ms  {val}')
