# -*- coding: utf-8 -*-

_start = None


def tic():
    import time
    global _start
    _start = time.time()


def toc(msg=None):
    import time
    global _start
    now = time.time()
    if _start:
        dt = now - _start
        if dt > 60:
            t = '%dm%02ds' % (dt / 60, dt % 60)
        else:
            t = '%2ds' % dt
        if msg:
            print '%-20s %s' % (msg, t)
        else:
            print t
    else:
        print "Toc: start time not set"
    _start = now
