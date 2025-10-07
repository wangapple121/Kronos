from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. 检查并设置设备（适配Mac ARM芯片）
def get_device():
    # 检查是否为Apple Silicon (M1/M2/M3等)
    if torch.backends.mps.is_available():
        return "mps"  # 使用Apple Metal加速
    elif torch.cuda.is_available():
        return "cuda:0"  # 备选：如果有NVIDIA GPU
    else:
        return "cpu"  # 最后使用CPU

device = get_device()
print(f"使用设备: {device}")


# 2. 加载模型和分词器
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 将模型移动到合适的设备
model = model.to(device)


# 3. 实例化预测器
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)


# 4. 准备数据
#df = pd.read_csv("./data/XSHG_5min_600977.csv")
df = pd.read_csv("./data/BABA_daily_kline_2023_to_now.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 根据用户要求，使用最新的数据往前推200个交易日作为训练数据
latest_date = df['timestamps'].max()
start_date = latest_date - timedelta(days=200)
# 1. 训练数据：最后200天
x_df = df[df['timestamps'] <= latest_date].tail(200)[['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df[df['timestamps'] <= latest_date].tail(200)['timestamps']


# 生成足够多的日期，然后手动过滤掉节假日
y_timestamp_full = pd.date_range(start='2025-10-02', periods=60, freq='D')

y_timestamp_filtered = []
count = 0

for date in y_timestamp_full:
    # 跳过周末 (Saturday=5, Sunday=6)
    if date.weekday() >= 5:
        continue
    y_timestamp_filtered.append(date)
    count += 1

    if count >= 30:  # 只需要30个交易日
        break

# 检查是否生成了足够的日期
if len(y_timestamp_filtered) < 30:
    print(f"警告：只生成了 {len(y_timestamp_filtered)} 个交易日的日期，少于所需的30个")

y_timestamp = pd.DatetimeIndex(y_timestamp_filtered)

# 4. Make Prediction
# 将y_timestamp转换为Series以匹配模型期望的输入格式
y_timestamp_series = pd.Series(y_timestamp)

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp_series,
    pred_len=len(y_timestamp_filtered),
    T=1.0,
    top_p=0.9,
    sample_count=5,  # 增加采样次数以获得更稳定的预测
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# 4. 绘图：历史 vs 预测
kline_df = df[df['timestamps'] <= latest_date].tail(200).set_index('timestamps')  # 只有历史

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(kline_df['close'], label='Historical Close', color='blue', linewidth=1.5)
ax1.plot(pred_df['close'], label='Predicted Close', color='red', linewidth=1.5, linestyle='--')
ax1.set_ylabel('Close Price')
ax1.legend()
ax1.grid(True)

ax2.plot(kline_df['volume'], label='Historical Volume', color='blue', linewidth=1.5)
ax2.plot(pred_df['volume'], label='Predicted Volume', color='red', linewidth=1.5, linestyle='--')
ax2.set_ylabel('Volume')
ax2.legend()
ax2.grid(True)

plt.xlabel('Date')
plt.tight_layout()
plt.show()