"""

This script creates a txt file with num2-num1 number of lines.

The name of output file: provided by user

The default name of the input file: all_params.txt

Generated from examples provided by supervisor

"""

import os

import sys

import random

from random import randint

from random import sample

import numpy as np

import pylab as pl

def trim_data(percent, name2, filename2, filename3, filename4, num2= 1024, filename1="./data/all_params.txt"):

       maxwid = 50

       lines = np.loadtxt(filename1)

       deleted_entries = [] 
 
       clean = []

       train = []

       while True:
    
            # choose a random position and flagging width:
            pos = np.random.randint(0,len(lines))

            wid = np.random.randint(1,maxwid+1)

            # check end of bandpass:
            if (pos+wid)>num2:

               wid = (pos+wid)-num2

            print("flagging ",wid," channels starting with channel ",pos)
    
            # apply flagging

            selected_lines = lines[pos:pos+wid]

            selected_lines[:,2:4]= 0.0 

            for i in selected_lines:
                 
               deleted_entries.append(i)

            # check flagged percentage:       

            ndata = 1024 - len(deleted_entries)
           
            print (len(deleted_entries))

            perc = 1 -(float(ndata)/float(num2))
             
            print (perc) 
    
            # if flagged percentage exceeds specification: exit loop
            if perc>float(percent):

                print("Percentage flagged: ",perc)

                break

       np.savetxt(filename3, deleted_entries, delimiter=' ')   # X is an array

       for line in lines:

         clean.append(line)

       np.savetxt(filename4, clean, delimiter=' ')   # X is an array

       for line in lines:

         if (line[2]!= 0.0):

            train.append(line)
        
       np.savetxt(filename2, train, delimiter=' ')   # X is an array
       
       return perc

def create_removed(file2, file3, file1="./data/all_params.txt"):

       with open(file1) as infile:

           f1 = infile.readlines()

       with open(file2) as infile:

           f2 = infile.readlines()

       only_in_f1 = [i for i in f1 if i not in f2] 

       only_in_f2 = [i for i in f2 if i not in f1]

       with open(file3, 'w') as outfile:

           if only_in_f1:

             for line in only_in_f1:

               outfile.write(line)

           if only_in_f2:

             for line in only_in_f2:

               outfile.write(line)

       return file3

 
percent = input('Please enter the percentage of entries to remove: ')

name2 = input('Please enter the name of the directory for the output files: ')

os.makedirs(name2)

filename2 = "./" + name2 + "/training.txt"

filename3 = "./" + name2 + "/deleted.txt"

filename4 = "./" + name2 + "/cleaned.txt"

print ('A file with {} percent of the original data has been successfully generated'.format(trim_data(percent, name2, filename2, filename3, filename4, num2= 1024, filename1="./data/all_params.txt")))

file2 = "./" + name2 +"/training.txt"

file3 = "./" + name2 +"/removed.txt"

print ("A file {} has been successfully created".format(create_removed(file2, file3, file1="./data/all_params.txt")))

















