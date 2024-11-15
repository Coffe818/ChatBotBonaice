import nltk
import numpy as np
#ingles
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()
#Español
from nltk.stem.snowball  import SnowballStemmer
stemmer = SnowballStemmer('spanish')


def tokenize(sentence):
    return nltk.word_tokenize(sentence)
''''
ejemplo de uso
    print(tokenize("A cuento esta el bonice?"))
    ['A', 'cuento', 'esta', 'el', 'bonice', '?']
'''

def stem(word):
    return stemmer.stem(word.lower())
'''
Ejemplo de uso
    palabras = ["jugador", "jugdores", "jugadoras", "jugar", "juega", "jugaron", "jugaran", "jugaria", "jugarian", "jugarian"]
    stemmed_palabras = [stem(p) for p in palabras]
    print (stemmed_palabras)
    ['jugador', 'jugdor', 'jugador', 'jug', 'jueg', 'jug', 'jug', 'jugari', 'jugari', 'jugari']
'''


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
    return bag



'''
Ejemplo de uso
    sentene= ["hola", "como", "estas"]
    words = ["hola", "Que","tal","como", "estas", "bien"]
    bag= [1,0,0,1,1,0]
'''

