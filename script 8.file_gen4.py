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

from random import randint

from random import sample


def trim_data(percent, name2, filename2, filename3, num2= 1024, filename1="./data/all_params.txt"):

       num1 = int(round((num2*percent/100), 0)) 

       with open(filename1) as file:

           lines = file.read().splitlines()

       #random_lines = random.sample(lines, num1) 
       n = randint(2,5)

       addends = []

       picks = range(1, num1)

       while sum(addends) != num1:

            addends = random.sample(picks, n-1)

            if sum(addends) > num1-1:

                continue

            addends.append(num1 - sum(addends))

       random_lines = []

       for i in addends:

          #j = randint(1, 1024-max(addends))
          random_line_num = random.randrange(0, len(lines)-max(addends))

          count = 0

          while count < i:

              random_lines.append(lines[random_line_num])

              #print j  

              random_line_num+=1

              count +=1

          i+=1       



       with open(filename2, "w") as output_file1:  

            output_file1.writelines(line + '\n' for line in lines if line not in random_lines)

            output_file1.close()  

       with open(filename3, "w") as output_file2:  

            output_file2.writelines(line + '\n' for line in lines if line in random_lines)

            output_file2.close()


       num_of_lines = num2 - num1

       remainder = 100 - percent

       print n

       print addends

       #print random_lines

       return remainder


 
percent = input('Please enter the percentage of entries to remove: ')

name2 = input('Please enter the name of the directory for the output files: ')

os.makedirs(name2)

filename2 = './' + name2 + '/' + "training" + '.txt'

filename3 = './' + name2 + '/' + "removed" + '.txt'

print 'A file {} with {} percent of the original data has been successfully generated'.format(filename2, trim_data(percent, name2, filename2, filename3, num2= 1024, filename1="./data/all_params.txt"))







