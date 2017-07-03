import time
import collections
import numpy as np
import sys

import conlleval


class Evaluator(object):
    def __init__(self, name, main_label_ids, main_label_name, label2id=None, conll_eval=False):
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

        if self.label2id is not None:
            self.id2label = collections.OrderedDict()

            # TODO hard coded id2label generation.
            for label in self.label2id:
                if label == 'B' or label == 'I':
                    full_label = label + '-' + main_label_name
                else:
                    full_label = 'O'
                self.id2label[self.label2id[label]] = full_label

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

            try:
                self.conll_format.append(
                    str(word_ids[i]) + "\t" + str(self.id2label[label_ids[i]]) + "\t" + str(
                        self.id2label[predicted_labels[i]]))
            except KeyError:
                print("Unexpected label id in predictions.")
            self.conll_format.append("")

    def gen_results(self):
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
            print(self.conll_format)
            conll_counts = conlleval.evaluate(self.conll_format)
            conll_metrics_overall, conll_metrics_by_type = conlleval.metrics(conll_counts)
            results[self.name + "_conll_accuracy"] = float(conll_counts.correct_tags) / float(
                conll_counts.token_counter)
            results[self.name + "_conll_p"] = conll_metrics_overall.prec
            results[self.name + "_conll_r"] = conll_metrics_overall.rec
            results[self.name + "_conll_f"] = conll_metrics_overall.fscore

        self.results = results
        return self.results

    def print_results(self):
        for key in self.results:
            print(key + ": " + str(self.results[key]))

    def verify_results(self):
        if np.isnan(self.results[self.name + "_cost_sum"]) or np.isinf(self.results[self.name + "_cost_sum"]):
            sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
            exit()

    def write_results(self, filename, text, spec='a'):
        with open(filename, spec, encoding='utf-8') as f:
            f.write(text + "\n")
            for key in self.results:
                f.write(key + ": " + str(self.results[key]) + "\n")
