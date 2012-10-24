"""
build the tests for gmix_em
"""

from fabricate import *
import sys, os
import glob

CC='gcc'

# -lrt is only needed for the timing stuff
LINKFLAGS=['-lm']

#CFLAGS=['-std=gnu99','-Wall','-Werror','-O2','-mfpmath=sse']
CFLAGS=['-std=gnu99','-Wall','-Werror','-O2']

test_sources = ['test','shape']
programs = [{'name':'test', 'sources':test_sources}]

def build():
    compile()
    link()

def compile():
    for prog in programs:
        for source in prog['sources']:
            run(CC, '-c', '-o',source+'.o', CFLAGS, source+'.c')

def link():
    for prog in programs:
        objects = [s+'.o' for s in prog['sources']]
        run(CC,'-o', prog['name'], objects,LINKFLAGS)

def clean():
    autoclean()


main()
