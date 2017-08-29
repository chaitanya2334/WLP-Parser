import argparse
import os


def template(name, t_per, ch_lvl, pos, dep_label, dep_word, g, word_emb, time):
    s = "#PBS -N " + name + "\n"
    s += "#PBS -l walltime=" + time + "\n"
    s += "#PBS -l nodes=1:ppn=28:gpus=1\n"
    s += "source ~/.init_workspace_owens\n"
    s += "cd Documents/action-sequence-labeler\n"

    s += "python -m main " "--train_per " + str(t_per) + " --train_word_emb " + word_emb + \
         " --lm_gamma " + str(g) + " --char_level " + ch_lvl + " --pos " + pos + " --dep_label " + dep_label + \
         " --dep_word " + dep_word + " " + name

    return s


def write_file(filename, s):
    with open(filename, 'w') as f:
        f.write(s)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate OSC jobs.')

    parser.add_argument('title',
                        help="The subscript attached to every job's file name")

    parser.add_argument('--time', '-t', metavar='t', required=True,
                        help='Wall time')

    parser.add_argument('--job_dir', '-j', metavar='DIR', required=True,
                        help='Path to job directory where all the generated jobs will be saved')

    args = parser.parse_args()
    time = args.time

    script_dir = args.job_dir

    gammas = [x / 10 for x in range(10)]
    ch_lvls = ["None", "Input", "Attention"]
    f_lvls = ["Yes", "No"]
    title = args.title
    for per in [100]:
        for i in range(1):
            for we in ["pre_and_post", "random"]:
                for g in gammas:
                    for ch in ch_lvls:
                        for f_pos in f_lvls:
                            for f_dep_label in f_lvls:
                                for f_dep_word in f_lvls:

                                    if f_pos == "No" and f_dep_label == "No" and f_dep_word == "No":
                                        continue

                                    name = title + '_T_SIZE_' + str(per) + '_R_' + str(i) + "_WE_" + str(we) + \
                                           "_G_" + str(g) + "_CH_" + ch + \
                                           "_POS_" + f_pos + "_DEP_L_" + f_dep_label + "_DEP_W_" + f_dep_word

                                    s = template(name, per, ch, f_pos, f_dep_label, f_dep_word, g, we, time)
                                    file_path = os.path.join(script_dir, name + ".job")
                                    write_file(file_path, s)
