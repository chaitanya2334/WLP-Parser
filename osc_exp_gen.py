import os
import config as cfg

def template(name, ch_lvl, f_lvl, g, time):
    s = "#PBS -N " + name + "\n"
    s += "#PBS -l walltime=" + time + "\n"
    s += "#PBS -l nodes=1:ppn=28:gpus=1\n"
    s += "source ~/.init_workspace_owens\n"
    s += "cd Documents/action-sequence-labeler\n"
    s += "python -m main --lm_gamma " + str(g) + " --char_level " + ch_lvl + " --feature_level " + f_lvl + " " + name

    return s


def write_file(filename, s):
    with open(filename, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    time = "20:00:00"

    gammas = [x / 10 for x in range(10)]
    ch_lvls = ["None", "Input", "Attention"]
    f_lvls = ["None", "v1"]
    title = "LSTM_SHUFFLE"
    for g in gammas:
        for ch in ch_lvls:
            for f in f_lvls:
                name = title + "_G_" + str(g) + "_CH_" + ch + "_F_" + f
                s = template(name, ch, f, g, time)
                file_path = os.path.join(cfg.SCRIPT_DIR, name + ".job")
                write_file(file_path, s)
