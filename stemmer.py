# -*- coding: utf-8 -*-
"""
Created on Sun May 31 22:54:33 2020

@author: hasbe
"""


from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from typing import List

def remove_elements_from_list(the_list, elements):
   return [element for element in the_list if element not in elements]

startJVM(
    getDefaultJVMPath(),
    '-ea',
    f'-Djava.class.path=E:/CSE/Thesis/Thesis/Code/zemberek-full.jar',
    convertStrings=False
)

TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')

TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
Token: JClass = JClass('zemberek.tokenization.Token')

morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
        Token.Type.Punctuation,
        Token.Type.NewLine,
        Token.Type.SpaceTab,
        Token.Type.URL
    ).build()

corpus_dir = "E:/CSE/Thesis/Thesis/Code/corpus/"
sourceFile = open(corpus_dir + 'posts12-mixed-all-removed.txt', "r", encoding="utf-8")
destFile = open(corpus_dir + 'stemmed-12-alternative.txt', "w", encoding="utf-8")


for line in sourceFile:
    line = line.strip()
    tokenizedLine = []
    for i, token in enumerate(tokenizer.tokenizeToStrings(JString(line))):
        tokenizedLine.append(str(token))
    tokenizedLine = " ".join(tokenizedLine)
    if tokenizedLine == "" or tokenizedLine == " ":
        continue
    analysis: java.util.ArrayList = (
    morphology.analyzeAndDisambiguate(line).bestAnalysis()
    )
    pos: List[str] = []
    for i, analysis in enumerate(analysis, start=1):
        pos.append(
            f'{str(analysis.getLemmas()[-1])}'
            )
    pos = remove_elements_from_list(pos, ["UNK", "lt", "gt"])
    analyzedLine = " ".join(pos)
    # print(f'\n Kelime Kökleri: {analyzedLine}')
    
    destFile.write(analyzedLine + "\n")
    
shutdownJVM()
