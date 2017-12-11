import csv
import time
import collections
import numpy as np
import sys

from tabulate import tabulate

import conlleval
import os
import config as cfg


class Evaluator(object):
    def __init__(self, name, main_label_ids, main_label_name, skip_label=None, label2id=None, conll_eval=False):
        # assuming that labels are either B, I, or O only.
        # main_label_ids are ids that will be converted to 1. ( that is the ids for B and I )
        if label2id is None:
            label2id = {'B': 0, 'I': 1, 'O': 2}
        self.name = name
        self.main_label_ids = main_label_ids
        self.label2id = label2id
        self.conll_eval = conll_eval

        self.cost_sum = 0.0
        self.correct_sum = 0.0
        self.main_predicted_count = 0
        self.main_total_count = 0
        self.main_correct_count = 0
        self.token_count = 0
        self.start_time = time.time()
        self.total_samples = 0
        self.results = None
        self.skip_tags = [label2id['<s>'], label2id['</s>']]

        if skip_label:
            self.skip_tags += [label2id[label] for label in skip_label]


        if self.label2id is not None:
            self.id2label = collections.OrderedDict()

            # TODO hard coded id2label generation.
            for label in self.label2id:
                self.id2label[self.label2id[label]] = label

        self.conll_format = []

    def append_data(self, cost, predicted_labels, word_ids, label_ids):
        self.total_samples += 1
        self.cost_sum += cost
        self.token_count += len(label_ids)
        self.correct_sum += np.equal(np.array(predicted_labels), np.array(label_ids)).sum()
        self.main_predicted_count += sum([predicted_label in self.main_label_ids
                                          for predicted_label in predicted_labels])

        self.main_total_count += sum([label_id in self.main_label_ids for label_id in label_ids])

        self.main_correct_count += sum([pred_label in self.main_label_ids and true_label in self.main_label_ids
                                        for pred_label, true_label in zip(predicted_labels, label_ids)])

        for i in range(len(word_ids)):
            if not label_ids[i] in self.skip_tags:
                try:
                    self.conll_format.append(
                        str(word_ids[i]) + "\t" + str(self.id2label[label_ids[i]]) + "\t" + str(
                            self.id2label[predicted_labels[i]]))
                except KeyError:
                    print("Unexpected label id in predictions.")

        self.conll_format.append("")

    def macro_fscore(self, metrics_by_type):
        p_total = 0
        r_total = 0

        for key, metric in metrics_by_type.items():
            p_total += metric.prec
            r_total += metric.rec

        p_avg = p_total / len(metrics_by_type)
        r_avg = r_total / len(metrics_by_type)
        if p_avg + r_avg == 0:
            f_avg = 0
        else:
            f_avg = (2 * p_avg * r_avg) / (p_avg + r_avg)

        return conlleval.Metrics(0, 0, 0, p_avg, r_avg, f_avg)

    def gen_summary_results(self):
        p = (float(self.main_correct_count) / float(self.main_predicted_count)) if (
            self.main_predicted_count > 0) else 0.0
        r = (float(self.main_correct_count) / float(self.main_total_count)) if (self.main_total_count > 0) else 0.0
        f = (2.0 * p * r / (p + r)) if (p + r > 0.0) else 0.0
        f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (p + r > 0.0) else 0.0

        results = collections.OrderedDict()
        results[self.name + "_dataset_count"] = self.total_samples
        results[self.name + "_cost_avg"] = self.cost_sum / float(self.token_count)
        results[self.name + "_cost_sum"] = self.cost_sum
        results[self.name + "_main_predicted_count"] = self.main_predicted_count
        results[self.name + "_main_total_count"] = self.main_total_count
        results[self.name + "_main_correct_count"] = self.main_correct_count
        results[self.name + "_p"] = p
        results[self.name + "_r"] = r
        results[self.name + "_f"] = f
        results[self.name + "_f05"] = f05
        results[self.name + "_accuracy"] = self.correct_sum / float(self.token_count)
        results[self.name + "_token_count"] = self.token_count
        results[self.name + "_time"] = float(time.time()) - float(self.start_time)

        if self.label2id is not None and self.conll_eval is True:
            conll_counts = conlleval.evaluate(self.conll_format)
            self.conll_metrics_overall, self.conll_metrics_by_type = conlleval.metrics(conll_counts)
            self.macro_metrics = self.macro_fscore(self.conll_metrics_by_type)
            results[self.name + "_conll_accuracy"] = float(conll_counts.correct_tags) / float(
                conll_counts.token_counter)
            results[self.name + "_conll_p"] = self.conll_metrics_overall.prec
            results[self.name + "_conll_r"] = self.conll_metrics_overall.rec
            results[self.name + "_conll_f"] = self.conll_metrics_overall.fscore
            results[self.name + "_macro_p"] = self.macro_metrics.prec
            results[self.name + "_macro_r"] = self.macro_metrics.rec
            results[self.name + "_macro_f"] = self.macro_metrics.fscore
            results['label_table'] = [
                [label, metric.prec, metric.rec, metric.fscore, conll_counts.t_found_correct[label]]
                for label, metric in sorted(self.conll_metrics_by_type.items())]

        self.results = results
        return self.results

    def classification_report(self):
        self.gen_summary_results()
        self.print_results()

    def print_results(self):
        for key in self.results:
            if key == 'label_table':
                print(tabulate([('Label', 'Precision', 'Recall', 'Fscore', 'Support')]+self.results[key],
                               headers="firstrow",
                               tablefmt='psql'))
            else:
                print(key + ": " + str(self.results[key]))

    def verify_results(self):
        if np.isnan(self.results[self.name + "_cost_sum"]) or np.isinf(self.results[self.name + "_cost_sum"]):
            sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
            exit()

    def config_desc(self):

        s = "Configuration:\n"
        s += "Learning Rate: {0}\n".format(cfg.LEARNING_RATE)
        s += "Gamma (will be non zero if LM is being used): {0}\n".format(cfg.LM_GAMMA)
        s += "Character Level embedding type: {0}\n".format(cfg.CHAR_LEVEL)
        return s

    def write_results(self, filename, text, overwrite):
        if overwrite:
            spec = 'w'
        else:
            spec = 'a'

        with open(filename, spec, encoding='utf-8') as f:
            f.write(text + "\n")
            f.write(self.config_desc())
            for key in self.results:
                if key == 'label_table':
                    f.write(tabulate(self.results[key], headers=['Label', 'Precision', 'Recall', 'Fscore'],
                                     tablefmt='psql'))
                else:
                    f.write(key + ": " + str(self.results[key]) + "\n")

    def write_csv_results(self, csv_filepath, title, overwrite=True):
        # if file doesnt exist or is empty or you want to overwrite the file, write the headers.
        if not os.path.isfile(csv_filepath) or os.stat(csv_filepath).st_size == 0 or overwrite:
            with open(csv_filepath, 'w') as csvfile:
                fieldnames = ['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filepath, 'a') as csvfile:
            fieldnames = ['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Method': title,
                             'Accuracy': self.results[self.name + "_accuracy"],
                             'Precision': self.results[self.name + "_conll_p"],
                             'Recall': self.results[self.name + "_conll_r"],
                             'F1-Score': self.results[self.name + '_conll_f']})
