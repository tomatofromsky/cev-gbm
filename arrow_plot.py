import os
import numpy as np
import matplotlib.pyplot as plt

def plot_with_envelope(i, price_series, save_folder="envelope_time_series", width_ratio=0.1):
    os.makedirs(save_folder, exist_ok=True)
    x = np.arange(len(price_series))
    y = np.array(price_series)

    lower = y * (1 - width_ratio)
    upper = y * (1 + width_ratio)

    fig, ax = plt.subplots()
    ax.plot(x, y, color='black', label='Price')
    ax.fill_between(x, lower, upper, color='gray', alpha=0.3)

    max_idx = np.argmax(y)
    min_idx = np.argmin(y)

    # 기준 길이 및 최소 길이 설정
    half_noise = (upper - lower) / 2
    base_len_mean = np.mean(half_noise)
    min_len = base_len_mean * 0.5
    scale_factor = 1.3

    for idx in (max_idx, min_idx):
        y_base = y[idx]
        arrow_len = max(half_noise[idx] * scale_factor, min_len)

        # ★ 가운데 점: 파란색, 크고 위에 그리기
        ax.scatter(x[idx], y_base, s=80, c='black', alpha=0.8, zorder=5)

        for direction in (+1, -1):
            # RGBA로 반투명한 파란색 지정
            rgba_blue = (0/255, 114/255, 189/255, 0.5)

            ax.annotate(
                '',
                xy=(x[idx], y_base + direction * arrow_len),
                xytext=(x[idx], y_base),
                arrowprops=dict(
                    color=rgba_blue,
                    linewidth=4,      # 몸통 굵기
                    headwidth=8,      # 머리 폭
                    headlength=8,     # 머리 길이
                    shrink=0
                )
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_ylim(y.min() * 0.7, y.max() * 1.3)
    ax.grid(False)
    # ax.legend()

    out = os.path.join(save_folder, f"time_series_envelope_{i+1}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out
