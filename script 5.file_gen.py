"""

This script creates a txt file with num2-num1 number of lines.

The name of output file: provided by user

The default name of the input file: all_params.txt

all_params.txt file should be in the same directory as the script

Script in Python 2

"""

import os

import sys

import random



def trim_data(num1, filename2, num2= 1024, filename1="./data/all_params.txt"):



    with open(filename1) as file:

       lines = file.read().splitlines()

    random_lines = random.sample(lines, num1) 

    with open(filename2, "w") as output_file:  

        output_file.writelines(line + '\n' for line in lines if line not in random_lines)

        output_file.close()  

    num_of_lines = num2 - num1

    return num_of_lines

 
num1 = input('Please enter the number of lines to remove: ')

name2 = input('Please enter the name of the output file (in quotes): ')

os.makedirs(name2)

filename2 = './' + name2 + '/' +name2 + '.txt'

print 'A file {} with {} rows has been successfully generated'.format(filename2, trim_data(num1, filename2, num2= 1024, filename1="./data/all_params.txt"))





