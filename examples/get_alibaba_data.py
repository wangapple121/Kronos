import yfinance as yf
import pandas as pd

# ================== 配置参数 ==================
symbol = "BABA"           # 阿里巴巴美股代码
start_date = "2023-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")  # 自动设为今天

# =============================================
# 获取日K线数据
# =============================================
print(f"正在下载 {symbol} 从 {start_date} 到 {end_date} 的日K线数据...")

stock = yf.Ticker(symbol)
data = stock.history(start=start_date, end=end_date, interval="1d")

if data.empty:
    raise ValueError(f"❌ 未获取到 {symbol} 的数据，请检查股票代码或网络连接。")

# =============================================
# 数据处理
# =============================================
data_reset = data.reset_index()

# 重命名时间列
data_reset.rename(columns={'Date': 'timestamps'}, inplace=True)

# 确保 timestamps 是 datetime 类型
data_reset['timestamps'] = pd.to_datetime(data_reset['timestamps'])

# 强制时间部分为 00:00:00（因为日K线没有具体分钟信息）
# 这样格式就是：2024-06-18 00:00:00
data_reset['timestamps'] = data_reset['timestamps'].dt.floor('D')  # 或 .dt.normalize()

# 计算成交额（amount）: close * volume
data_reset['amount'] = data_reset['Close'] * data_reset['Volume']

# 重命名其他列
final_data = data_reset.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]

# 按时间排序
final_data.sort_values('timestamps', inplace=True)
final_data.reset_index(drop=True, inplace=True)

# =============================================
# 保存为 CSV（时间格式化为 YYYY-MM-DD HH:MM:SS）
# =============================================
# 使用 date_format 参数控制输出格式
filename = f"{symbol}_daily_kline_2023_to_now.csv"
final_data.to_csv(
    filename,
    index=False,
    date_format='%Y-%m-%d %H:%M:%S'  # 强制输出为 2024-06-18 00:00:00 格式
)

# =============================================
# 输出预览（前5行）
# =============================================
print(f"✅ 成功获取 {len(final_data)} 天的数据！")
print(f"📊 数据时间范围：{final_data['timestamps'].min()} 到 {final_data['timestamps'].max()}")
print(f"💾 已保存为：{filename}")
print("\n📊 前5行数据预览：")
print(final_data.head().to_string(index=False))