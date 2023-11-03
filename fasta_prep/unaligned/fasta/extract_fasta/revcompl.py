import sys

LINE_WIDTH = 80

seq = []

compl_dict = {'A':'T','C':'G','T':'A','G':'C',
              'a':'t','c':'g','t':'a','g':'c'}

for line in sys.stdin:
    seq.extend(list(line[:-1]))

for idx,c in enumerate(seq[::-1]):
    sys.stdout.write(compl_dict.get(c,c))
    if (idx+1)%LINE_WIDTH==0:
        sys.stdout.write('\n')

if not (idx+1)%LINE_WIDTH==0:
    sys.stdout.write('\n')
