# -*- coding: utf-8 -*-

import csv


class MyCorpus:
    def __init__(self, csv_filename, column, rows=None):
        self.filename = csv_filename
        self.column = column
        self.rows = rows  # rows must be sorted and unique

    def __iter__(self):
        with open(self.filename, 'rb') as f:
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


class MyCSVReader:
    # This is a wrapper around CSVReader which allows reading individual rows
    # within the file.
    # NOTE: there is a difference between rows and lines; each CSV row
    # corresponds to one CSV item and so may encompass several lines.

    def __init__(self, csv_filename):
        self.file = open(csv_filename, 'rb')

        # csv.reader only provides reliable line numbers; because it reads
        # several characters at the time, we cannot rely on self.file.tell()
        # so, we first get entry lines, and only then convert to file tells

        # row -> line
        lines = []
        reader = csv.reader(self.file, delimiter=',', quotechar='"')
        line = 0
        for row, text in enumerate(reader):
            if row > 0:  # skip header
                lines.append(line)
            line = reader.line_num

        # line -> tell
        self.tells = [0] * (len(lines)+1)
        self.file.seek(0)
        self.file.readline()  # skip header
        line = 1
        for i in xrange(len(lines)):
            while line < lines[i]:
                self.file.readline()
                line += 1
            self.tells[i] = self.file.tell()

    def __del__(self):
        self.file.close()

    def get_row(self, column, row):
        self.file.seek(self.tells[row])
        self.file.seek(self.tells[row])
        reader = csv.reader(self.file, delimiter=',', quotechar='"')
        text = next(reader)[column]
        return unicode(text, 'utf-8')


# testing
if __name__ == '__main__':
    corpus = MyCSVReader('../../data/ItemInfo_train.csv')
    print 'first 10 items...'
    for i in xrange(10):
        print '\t', corpus.get_row(2, i)
    print 'now backwards...'
    for i in xrange(9, -1, -1):
        print '\t', corpus.get_row(2, i)
