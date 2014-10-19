import sys, os
sys.path.append('PPP-dataset/')
import dataset

en_dict = dataset.make_dictionary('PPP-dataset/embeddings-scaled.EMBEDDING_SIZE=25.txt')

while True:
    sentence = input()
    fs = dataset.FormatSentence(sentence, en_dict)
    file = open('input.txt', 'w')
    file.write(fs.data_set_input())
    file.close()
    os.system('cd ML; th forward.lua')
    result = open('output.txt', 'r')

    a, b, c = [], [], []
    i = 0
    for line in result:
        if line == '1\n':
            a.append(fs.words[i])
        elif line == '2\n':
            b.append(fs.words[i])
        elif line == '3\n':
            c.append(fs.words[i])
        i += 1

    print(' '.join(a) + ' | ' + ' '.join(b) + ' | ' + ' '.join(c))
