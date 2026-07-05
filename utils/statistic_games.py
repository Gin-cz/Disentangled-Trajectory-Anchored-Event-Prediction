import os
import pandas as pd
from pathlib import Path
from collections import Counter

# 设置数据目录
data_dir = os.environ.get("DISTRAJ_NODE_EDGE_DIR", os.path.join("data", "interim", "syn_node_edge_data"))

print("="*60)
print("足球比赛事件数据统计")
print("="*60)

# 统计CSV文件数量（比赛数量）
csv_files = list(Path(data_dir).glob("*.csv"))
num_matches = len(csv_files)

if num_matches == 0:
    print(f"错误：在 {data_dir} 中没有找到CSV文件")
    exit(1)

print(f"\n📊 总比赛数: {num_matches}")

# 统计总事件数和事件类型
total_events = 0
event_type_counter = Counter()

print(f"\n正在统计事件数据...")

for i, csv_file in enumerate(csv_files, 1):
    try:
        df = pd.read_csv(csv_file)
        num_events = len(df)
        total_events += num_events
        
        # 尝试找到事件类型列
        event_col = None
        for col_name in ['event_type', 'event type', 'eventType', 'type']:
            if col_name in df.columns:
                event_col = col_name
                break
        
        if event_col:
            event_type_counter.update(df[event_col].tolist())
        
        # 显示进度
        if i % 20 == 0 or i == num_matches:
            print(f"  已处理: {i}/{num_matches} 场比赛")
            
    except Exception as e:
        print(f"  警告：读取文件 {csv_file.name} 失败: {e}")

print("\n" + "="*60)
print("统计结果")
print("="*60)

# 基本统计
print(f"\n📈 基本统计:")
print(f"  总比赛数:        {num_matches:>10}")
print(f"  总事件数:        {total_events:>10}")
print(f"  平均每场事件数:  {total_events / num_matches:>10.2f}")

# 事件类型统计
if event_type_counter:
    print(f"\n📋 事件类型统计 (共 {len(event_type_counter)} 种):")
    print(f"\n{'事件类型':<30} {'数量':>12} {'占比':>10}")
    print("-" * 54)
    
    for event_type, count in sorted(event_type_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_events) * 100
        print(f"{str(event_type):<30} {count:>12,} {percentage:>9.2f}%")
    
    print("-" * 54)
    print(f"{'总计':<30} {total_events:>12,} {100.0:>9.2f}%")
else:
    print(f"\n⚠️  警告：未找到事件类型列")
    print(f"   第一个CSV文件的列名: {pd.read_csv(csv_files[0]).columns.tolist()}")

print("\n" + "="*60)
