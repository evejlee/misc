"""
Build the admom test program
"""
from fabricate import *
import sys, os
import optparse

import glob


parser = optparse.OptionParser()
# make an options list, also send to fabricate
optlist=[optparse.Option('--noopt',action="store_true",help="turn off compiler optimizations"),
         optparse.Option('-d','--debug',action="store_true",help="turn on debugging (assert)"),
         optparse.Option('--tests',action="store_true",help="build the test programs)")]
         
parser.add_options(optlist)

options,args = parser.parse_args()

CC='gcc'

LINKFLAGS=['-lm']

CFLAGS=['-std=gnu99','-Wall','-Werror']
if not options.noopt:
    CFLAGS += ['-O2']
if not options.debug:
    CFLAGS += ['-DNDEBUG']

programs=[]

if options.tests:
    trand_src=['test/test-mca-rand','mca']
    trand_prog = {'name':'test/test-mca-rand','sources':trand_src}

    tconst_src=['test/test-mca-const','mca']
    tconst_prog = {'name':'test/test-mca-const','sources':tconst_src}

    programs=[trand_prog, tconst_prog]

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
        #run(CC,LINKFLAGS,'-o', prog['name'], objects)
        run(CC,'-o',prog['name'],objects,LINKFLAGS)

def clean():
    autoclean()

# send options so it won't crash on us
main(extra_options=optlist)

