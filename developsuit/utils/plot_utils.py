import numpy as np
import math as m


def average_every_n(plot_list, n, keep_len=False):
    if len(plot_list) <= n:
        averages = plot_list
    else:
        if not keep_len:
            averages = [sum(plot_list[i:i + n]) / len(plot_list[i:i + n]) for i in range(0, len(plot_list), n)]
        else:
            if n % 2 == 0:
                n += 1
            averages_start = [sum(plot_list[0: int(i + (n + 1) / 2)]) / (i + (n + 1) / 2)
                              for i in range(int((n - 1) / 2))]

            averages_mid = [sum(plot_list[i - int((n - 1) / 2): i + int((n + 1) / 2)]) / n
                            for i in range(int((n - 1) / 2), len(plot_list) - int((n - 1) / 2))]

            averages_end = [sum(plot_list[i - int((n - 1) / 2):]) / len(plot_list[i - int((n - 1) / 2):])
                            for i in range(len(plot_list) - int((n - 1) / 2), len(plot_list))]

            averages = averages_start + averages_mid + averages_end

    return averages


def std_every_n(plot_list, n, keep_len=False):
    if len(plot_list) <= n:
        std_list = plot_list
    else:
        std_list = []
        if not keep_len:
            for i in range(0, len(plot_list), n):
                cal_list = plot_list[i:i + n]
                mean = sum(cal_list) / len(cal_list)
                var = sum((x - mean) ** 2 for x in cal_list) / len(cal_list)
                std = m.sqrt(var)
                std_list.append(std)
        else:
            if n % 2 == 0:
                n += 1

            for i in range(int((n - 1) / 2)):
                cal_list = plot_list[0: int(i + (n + 1) / 2)]
                mean = sum(cal_list) / len(cal_list)
                var = sum((x - mean) ** 2 for x in cal_list) / len(cal_list)
                std = m.sqrt(var)
                std_list.append(std)

            for i in range(int((n - 1) / 2), len(plot_list) - int((n - 1) / 2)):
                cal_list = plot_list[i - int((n - 1) / 2): i + int((n + 1) / 2)]
                mean = sum(cal_list) / len(cal_list)
                var = sum((x - mean) ** 2 for x in cal_list) / len(cal_list)
                std = m.sqrt(var)
                std_list.append(std)

            for i in range(len(plot_list) - int((n - 1) / 2), len(plot_list)):
                cal_list = plot_list[i - int((n - 1) / 2):]
                mean = sum(cal_list) / len(cal_list)
                var = sum((x - mean) ** 2 for x in cal_list) / len(cal_list)
                std = m.sqrt(var)
                std_list.append(std)

    return std_list