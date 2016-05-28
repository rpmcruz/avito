# -*- coding: utf-8 -*-


class MyCorpus:
    def __init__(self, csv_filename, column, rows=None):
        self.filename = csv_filename
        self.column = column
        self.rows = rows  # rows must be sorted and unique

    def __iter__(self):
        with open(self.filename, 'rb') as f:
            import csv
            reader = csv.reader(f, delimiter=',', quotechar='"')
            r = 0
            for row, text in enumerate(reader):
                if row == 0:
                    continue  # skip header
                if self.rows is not None:
                    if r == len(self.rows):
                        break
                    if row-1 < self.rows[r]:
                        continue  # incremental search
                    assert row-1 == self.rows[r]
                    r += 1
                document = text[self.column]
                yield unicode(document, 'utf-8')
