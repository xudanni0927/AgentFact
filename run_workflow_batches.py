import subprocess
from concurrent.futures import ThreadPoolExecutor

script = "main_workflow.py"
input_file = "Dataset_RW-Post/demo/demo.jsonl"  # replace with your full dataset path for real runs
dataset = "rwpost"

batch_size = 100
start_id = 0
max_workers = 16   # 并行进程数（根据GPU/CPU调整）

# 统计 jsonl 行数
with open(input_file, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

END = total_lines
print(f"Total samples: {END}")


def run_batch(start, end):
    cmd = [
        "python", script,
        "--start_id", str(start),
        "--end_id", str(end),
        "--input_file", input_file,
        "--dataset", dataset
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd)


# 生成任务列表
tasks = []
current = start_id

while current < END:
    batch_end = min(current + batch_size, END)
    tasks.append((current, batch_end))
    current = batch_end


# 并行执行
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for start, end in tasks:
        executor.submit(run_batch, start, end)