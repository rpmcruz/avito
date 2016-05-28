# -*- coding: utf-8 -*-

startTime_for_tictoc = None


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc(msg=None):
    import time
    if startTime_for_tictoc:
        if msg:
            print '%s: %ds' % (msg, time.time() - startTime_for_tictoc)
        else:
            print '%ds' % (time.time() - startTime_for_tictoc)
    else:
        print "Toc: start time not set"
    tic()
