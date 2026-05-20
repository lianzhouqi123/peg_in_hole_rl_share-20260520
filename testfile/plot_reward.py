import pandas as pd
import matplotlib.pyplot as plt
from developsuit.utils.plot_utils import average_every_n
from datetime import datetime


def plot_reward(load_file, if_save, show_mode, fix_axis=False):
    save_file = f"../fig/reward/"

    try:
        df_l = pd.read_csv('{}/reward.csv'.format(load_file), header=None)
        reward_list = df_l.values.reshape(-1)
    except:
        reward_list = []

    # 图片设置
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 6.5
    fig, ax = plt.subplots(figsize=(88 / 25.4, 50 / 25.4), dpi=1200)
    fig.subplots_adjust(left=0.16, right=0.85, bottom=0.18, top=0.98)

    # 遍历图形的四个边框，并分别设置它们的线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # reward_list = [x for x in reward_list if abs(x) <= 5.5e3]
    reward_list_average = average_every_n(reward_list, 50, keep_len=True)
    episodes_reward_list = list(range(len(reward_list)))

    ax.plot(episodes_reward_list, reward_list, linewidth=0.5, color=(0, 0.5, 0.8, 0.4))
    line1, = ax.plot(episodes_reward_list, reward_list_average, linewidth=0.5, color=(0, 0.5, 0.8))

    plt.legend([line1], [line1.get_label()],loc="lower right", frameon=False)

    if fix_axis:
        ax.set_xlim(0, 4000)
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        # ax.set_ylim(-5000, 1000)
        # ax.set_yticks([-5000, -4000, -3000, -2000, -1000, 0, 1000])
        ax.set_ylim(-1500, 500)
        ax.set_yticks([-1500, -1000, -500, 0, 500])
    ax.set_xlabel(r"Episodes")
    ax.set_ylabel(r"Reward")

    if save_file is not None and if_save:
        # 文件名
        log_time = datetime.now()
        formatted_log_time = log_time.strftime("%Y-%m-%d %H-%M-%S")
        fig_file_name = save_file + f"reward" + "【" + formatted_log_time + "】.png"
        plt.savefig(fig_file_name, transparent=True, dpi=1200)

    if show_mode == "show":
        plt.show()

    return fig

if __name__ == "__main__":
    load_file1 = f"../result/"

    if_save1 = True
    # if_save1 = False

    show_mode1 = "no_show"
    # show_mode1 = "show"

    plot_reward_arm(load_file1, if_save1, show_mode1, fix_axis=False)