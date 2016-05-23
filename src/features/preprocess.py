# -*- coding: utf-8 -*-

## CSV FILE -> DOCUMENTS

import csv


class Documents:
    def __init__(self, csv_filename, column, rows=None):
        self.filename = csv_filename
        self.column = column
        self.rows = rows  # rows must be sorted and unique

    def __iter__(self):
        with open(self.filename, 'r') as f:
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
                yield document.lower()

## DOCUMENT -> WORDS


def split_only_alphanumeric(document):
    # FIXME: this does not work for Cyrillic
    #from nltk.tokenize import RegexpTokenizer
    #tokenizer = RegexpTokenizer(r'\w+')
    #return tokenizer.tokenize(document)
    return document.split()

## FILTER WORDS

import enchant

with open('utils/colors.txt') as f:
    _colors = f.readlines()
_ru_dict = enchant.Dict('ru')


def filter_russian_except_colors(words):
    ret = []
    for word in words:
        if word and (not _ru_dict.check(word) or word in _colors):
            ret.append(word)
    return ret
