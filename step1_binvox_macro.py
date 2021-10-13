# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:43:20 2020

@author: NSMLAB
"""

import glob
from subprocess import check_output


def runbinvox(stl):
    #cmd=check_output("cd", shell=True) #return the path, where there are the objects
#    print()
#    check_output("binvox.exe -rotx -rotx -rotx -d 26 " + stl, shell=True)
    check_output("binvox.exe -d 26 " + stl, shell=True)

    if __name__ == "__main__":
        for f in glob.glob("./*.stl"):
            print(f)
            runbinvox(f)
            # binvox file saves in the forder where stl file is
        pass
