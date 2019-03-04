"""

This script creates a txt file with a specified percentage.

The name of directory: provided by user

The default name of the input file: all_params.txt


"""

import os

import sys

import random


def trim_data(percent, name2, filename2, filename3, num2= 1024, filename1="./data/all_params.txt"):

       num1 = int(round((num2*percent/100), 0)) 

       with open(filename1) as file:

           lines = file.read().splitlines()

       random_lines = random.sample(lines, num1) 

       with open(filename2, "w") as output_file1:  

            output_file1.writelines(line + '\n' for line in lines if line not in random_lines)

            output_file1.close()  

       with open(filename3, "w") as output_file2:  

            output_file2.writelines(line + '\n' for line in lines if line in random_lines)

            output_file2.close()


       num_of_lines = num2 - num1

       remainder = 100 - percent

       return remainder


 
percent = input('Please enter the percentage of entries to remove: ')

name2 = input('Please enter the name of the directory for the output files: ')

os.makedirs(name2)

filename2 = './' + name2 + '/' + "training" + '.txt'

filename3 = './' + name2 + '/' + "removed" + '.txt'

print 'A file {} with {} percent of the original data has been successfully generated'.format(filename2, trim_data(percent, name2, filename2, filename3, num2= 1024, filename1="./data/all_params.txt"))




