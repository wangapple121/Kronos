import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# 设置股票代码
stock_code = "601995"
stock_name = "中金公司"

# 计算日期范围：最近一年
end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

print(f"正在获取 {stock_name} ({stock_code}) 从 {start_date} 到 {end_date} 的日K数据...")

try:
    # 使用 akshare 获取日K数据（前复权）
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权
    )

    if df.empty:
        raise ValueError("获取的数据为空，请检查股票代码、日期范围或网络连接。")

    # 检查必要字段是否存在
    required_columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"缺失以下列：{missing_cols}")

    # 重命名列
    df.rename(columns={
        '日期': 'timestamps',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',     # 单位：手（1手 = 100股）
        '成交额': 'amount'      # 单位：元
    }, inplace=True)

    # 转换时间格式为 datetime
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 确保数值列是 float 类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 检查是否有 NaN
    null_count = df[numeric_cols].isnull().sum().sum()
    if null_count > 0:
        print(f"⚠️ 检测到 {null_count} 个 NaN 值，使用前向填充 + 后向填充处理...")
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        # 再次检查
        if df[numeric_cols].isnull().any().any():
            raise ValueError("数据仍包含 NaN，请检查原始数据源。")

    # 按时间排序并重置索引
    df.sort_values('timestamps', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 保留指定列，确保顺序正确
    final_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
    df = df[final_columns]

    # 格式化 timestamps 为 'YYYY-MM-DD 00:00:00'
    df['timestamps'] = df['timestamps'].dt.strftime('%Y-%m-%d 00:00:00')

    # 数值列保留两位小数（volume 和 amount 是整数单位，但保留 .00 形式）
    df['open'] = df['open'].round(2)
    df['high'] = df['high'].round(2)
    df['low'] = df['low'].round(2)
    df['close'] = df['close'].round(2)
    df['volume'] = df['volume'].round(2)   # 成交量（手），允许小数（如机构数据）
    df['amount'] = df['amount'].round(2)   # 成交额（元）

    # 保存为 CSV
    filename = f"{stock_code}_daily_kline_last_year.csv"
    df.to_csv(filename, index=False)

    print(f"\n✅ 成功获取 {len(df)} 天日K数据")
    print("数据预览：")
    print(df.head())
    print(f"\n已保存至 '{filename}'")
    print("\n📌 数据说明：")
    print("  - 时间戳格式：YYYY-MM-DD 00:00:00")
    print("  - 价格保留两位小数")
    print("  - volume: 成交量（单位：手）")
    print("  - amount: 成交额（单位：元）")
    print("  - 无空值，已做完整性校验")

except Exception as e:
    print(f"❌ 获取数据失败：{e}")