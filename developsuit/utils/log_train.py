import inspect
from datetime import datetime
import os
from developsuit.envs.arm7d.single_arm7d_obsavoid import SingleArm7dObsAvoid


# from developsuit.envs.MAPPO_env4_2 import env


def separate_word_end(text):
    text += "\n#########################################################################\n\n"
    return text


def log_train(file_info, train_info, param_train, text_func,
              log_file_all, fig_all, fig_name_all):
    text = ""

    # 加入训练日志
    text += train_info
    text = separate_word_end(text)

    # 加入文件路径
    text += "训练使用的文件（内容可能在后续训练中修改，文件名仅供参考）\n\n"
    for key in file_info:
        text += key + " : " + str(file_info[key]) + "\n"
    text = separate_word_end(text)

    # 加入训练参数
    text += "训练使用的超参数\n\n"

    for key in param_train:
        text += key + " : " + str(param_train[key]) + "\n"

    text = separate_word_end(text)

    # 加入函数
    text += text_func

    # 日志文件名
    log_time = datetime.now()
    formatted_log_time = log_time.strftime("%Y-%m-%d %H-%M-%S")
    demo_file_name = os.path.basename(file_info["demo_file_path"])[:-3]
    log_file_name = demo_file_name + "【" + formatted_log_time + "】"

    log_file_dic_path = log_file_all + "/" + log_file_name + "/"
    if not os.path.exists(log_file_dic_path):
        os.mkdir(log_file_dic_path)

    log_file_path = log_file_dic_path + log_file_name + ".txt"

    # 写入
    with open(log_file_path, 'w') as file:
        file.write(text)

    # 存图片
    for ii in range(len(fig_all)):
        fig = fig_all[ii]
        fig_name = fig_name_all[ii]

        fig_path = log_file_dic_path + "/" + fig_name + ".png"
        fig.savefig(fig_path)


def log_func(env, runner, other_func=None):
    text = ""

    text += "训练函数\n\n"
    if other_func is not None and len(other_func) > 0:
        for func in other_func:
            func_code = inspect.getsource(func)
            text += func_code
            text = separate_word_end(text)

    # 加入runner函数
    # 获取 env 对象中所有方法和属性的名字
    all_attrs = dir(runner)

    # 过滤出是方法的名字
    runner_func_all = [attr for attr in all_attrs if inspect.ismethod(getattr(runner, attr))]

    text += "训练函数\n\n"
    for runner_func_name in runner_func_all:
        func = getattr(runner, runner_func_name, None)
        func_code = inspect.getsource(func)
        text += func_code
        text = separate_word_end(text)

    # 加入奖励函数
    # 获取父类的所有方法名（不包括继承自 object 的方法）
    parent_methods = {attr for cls in inspect.getmro(SingleArm7dObsAvoid) for attr in dir(cls) if
                      inspect.ismethod(getattr(cls, attr, None)) and attr not in dir(object())}

    # 获取 env 对象中所有方法和属性的名字
    all_attrs = dir(env)

    # 过滤出是方法的名字
    methods = [attr for attr in all_attrs if inspect.ismethod(getattr(env, attr))]

    # 从 env 的方法中排除父类的方法
    env_func_all = [method for method in methods if method not in parent_methods]

    text += "奖励函数\n\n"
    for env_func_name in env_func_all:
        func = getattr(env, env_func_name, None)
        func_code = inspect.getsource(func)
        text += func_code
        text = separate_word_end(text)

    return text
