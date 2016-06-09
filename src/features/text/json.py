# -*- coding: utf-8 -*-

# Compara JSON values.

import numpy as np
import itertools

'''
http://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
'''


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


class MyJSON:
    def compare(self, text1, text2):
        if len(text1) and len(text2):
            # one possible optimization would be to use a regular expression
            # rather than the fully featured json parsing
            import simplejson as json
            words1 = json.loads(text1).values()
            words2 = json.loads(text2).values()

            words1 = set(word for word in words1 if hasNumbers(word))
            words2 = set(word for word in words2 if hasNumbers(word))

            common = words1 & words2
            den = min(len(words1), len(words2))
            if den > 0:
                return len(common) / float(den)
        return 0

    def transform(self, myreader, rows):
        ret = np.zeros(len(rows))
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)

        for i, (row1, row2) in enumerate(itertools.izip(
                rows[ix[:(len(ix)/2)]], rows[ix[(len(ix)/2):]])):
            text1 = myreader.get_row(5, row1)
            text2 = myreader.get_row(5, row2)
            ret[i] = self.compare(text1, text2)
        return ret
