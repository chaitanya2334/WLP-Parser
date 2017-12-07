import config as cfg
import os
import csv

from main import dataset_prep


def write_csv(csv_filepath, protocols, overwrite=True):
    # if file doesnt exist or is empty or you want to overwrite the file, write the headers.
    fieldnames = ['protocol_id', 'protocol_name']

    if not os.path.isfile(csv_filepath) or os.stat(csv_filepath).st_size == 0 or overwrite:
        with open(csv_filepath, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_filepath, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for protocol in protocols:
            writer.writerow({'protocol_id': protocol.protocol_name.split("_")[1],
                            'protocol_name': " ".join(protocol.heading)})


if __name__ == '__main__':
    dataset = dataset_prep(loadfile=cfg.DB)
    p_sorted = sorted(dataset.protocols, key=lambda p: int(p.protocol_name.split("_")[1]))
    write_csv("protocol_metadata.csv", p_sorted)
