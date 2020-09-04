import sys

digit_string = sys.argv[1]

sum = 0

for dig in digit_string:
    sum += int(dig)
    
print(sum)