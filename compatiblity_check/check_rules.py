from nli import nli_inference

def obj_eq(obj1, obj2):
    sentence1 = 'There is a ' + obj1
    sentence2 = 'There is a ' + obj2
    result = nli_inference(sentence1,sentence2)
    if result == 'contradiction':
        return False
    else:
        return True

def action_eq(action1, action2):
    if action1=='' or action2=='':
        return True
    sentence1 = 'I am' + action1
    sentence2 = 'I am' + action2
    result = nli_inference(sentence1,sentence2)
    if result == 'contradiction':
        # print(sentence1,sentence2)
        return False
    else:
        return True

def action_in(action, action_list):
    result = False
    for act in action_list:
        if obj_eq(action[0],act[0]) and action_eq(action[1],act[1]) and action[2]==act[2]:
            result = True
            break
    return result

def check_whole(sentence_source,sentence_split_1,sentence_split_2):
    return nli_inference(sentence_source,sentence_split_1+'.'+sentence_split_2) != 'contradiction'