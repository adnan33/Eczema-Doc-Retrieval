from spellchecker import SpellChecker
spell = SpellChecker()
def preprocess_input(data):
    words=data
    words=words.lower().split()
    for i,word in enumerate(words):
       words[i]=spell.correction(word)
    data=" ".join(words)+"?"
    if(data.lower().find("hand eczema")!=-1):
        return data
    elif(data.lower().find("atopic eczema")!=-1):
        data=data.lower().replace("atopic eczema","atopic dermatitis")
        return data
    elif(data.lower().find("eczema")!=-1):
        data=data.lower().replace("eczema","atopic dermatitis")
        data=data.lower().replace("atopic atopic","atopic")
        return data
    elif(data.lower().find("eczema")==-1 and data.lower().find("atopic dermatitis")):
        return data+" atopic dermatitis"
a=preprocess_input('are blisters contagious?')
print(a)