import re
from dependency_parsing import nlp
def complete_sentence(sentence):
    doc = nlp(sentence)
    new_sentence = ''
    cnt = 0
    pre = None
    for token in doc:
        if token.pos_ == 'VERB' or (token.pos_== 'NOUN' and token.dep_ == 'dobj'):
            if token.text.endswith('ing'):
                # print(token.text+" "+token.o)
                cnt+=1
                if cnt > 1:
                    new_sentence += 'and '
                else:
                    if pre != None and pre.pos_ != 'AUX':
                        # print(pre.text)
                        new_sentence += "is "+ token.text + ' '
                    else:
                        new_sentence += token.text + ' '
                continue
        new_sentence += token.text + ' ';
        pre = token;
    new_sentence = re.sub(r'a\s[A-Za-z]+(\s[A-Za-z]+)?\sof ','',new_sentence);
    return new_sentence



def complete_noun(word):
    result = word.lemma_
    for child in word.children:
        if child.dep_ == 'compound':
            result = ' '.join([child.text,result])
    return result


def prep_expand(token):
    prep_text = ''
    oftoken = token;
    findof = False;
    for child in token.children:
        if child.text == 'of' :
            prep_text+= 'of'
            findof = True
            oftoken = child
    if not findof:
        return prep_text

    for child in oftoken.children:
        if child.dep_ == "pobj":
            for subchild in child.children:
                if subchild.dep_ in ('amod', 'det','compound'):
                    prep_text += ' '+subchild.text
            prep_text += ' '+child.text
    return prep_text



def extract_tuples_from_fragment(sentence):
    sentence = complete_sentence(sentence)
    sentence = complete_sentence(sentence)
    # print(sentence)
    doc = nlp(sentence)

    tuples = []
    current_subject = None
    # Iterate over the tokens in the parsed sentence
    for token in doc:
        # Identify noun chunks (subjects)
        if token.dep_ in ('nsubj', 'nsubjpass'):
            if token.tag_ == 'NNS':
                current_subject = [(complete_noun(token),'plural')]
            else:
                current_subject = [(complete_noun(token),'singular')]

        if token.dep_ == 'conj':
            if(token.head.dep_ in ('nsubj', 'nsubjpass')):
                if token.tag_ == 'NNS':
                    current_subject.append((complete_noun(token),'plural'))
                else:
                    current_subject.append((complete_noun(token),'singular'))

        if token.dep_ == 'ROOT' and token.pos_ == 'NOUN':
            if token.tag_ == 'NNS':
                tuples.append((complete_noun(token),'','plural'))
            else:
                tuples.append((complete_noun(token),'','singular'))
        # Identify verbs (actions)
        if token.pos_ == 'VERB' and token.dep_ in ('ROOT','conj'):
            if token.dep_ == 'conj'and token.head.dep_ != 'ROOT':
                continue;
            action = token.lemma_

            # Find objects related to the verb
            obj = None
            for child in token.children:
                if child.dep_ in ('dobj', 'pobj'):
                    obj = child.text
                    pre = '';
                    # Include any descriptive words (like determiners and adjectives)
                    for subchild in child.children:
                        if subchild.dep_ in ('amod', 'det','compound'):
                            pre += subchild.text+' '
                    obj = pre + obj
                elif child.dep_ == 'prep':
                    prep_text = child.text
                    pre = '';
                    prep_obj = ''
                    for subchild in child.children:
                        if subchild.dep_ == 'pobj':
                            prep_obj = subchild.text + ' ' + prep_expand(subchild)
                            for ad in subchild.children:
                                if ad.dep_ in ('amod', 'det','compound'):
                                    pre += ad.text+' '
                    prep_obj = prep_text+' '+ pre +prep_obj
                    # # Handle prepositions (e.g., "with a dog")
                    # prep_obj = ' '.join([child.text] + [c.text for c in child.children])
                    obj = f"{obj} {prep_obj}" if obj else prep_obj
                    break
            if current_subject and action:
                for subj in current_subject:
                    if obj:
                        tuples.append((subj[0],' '.join([action, obj]),subj[1]))

                    else:
                        tuples.append((subj[0],action,subj[1]))
    return tuples

def extract_nouns(sentence):
    tuples = []
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == 'NOUN' and token.dep_ != 'compound':
            if token.tag_ == 'NNS':
                tuples.append((complete_noun(token),'plural'))
            else:
                tuples.append((complete_noun(token),'singular'))
    return tuples

def merge(object_list,action_list):
    result = []
    for obj in object_list:
        if obj[0] not in [t[0] for t in action_list]:
            result.append((obj[0],'',obj[1]))
    result.extend(action_list)
    return result