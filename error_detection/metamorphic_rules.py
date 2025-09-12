from compatiblity_check.check_rules import obj_eq, action_in

def mr_1(action_source,action_split_left,action_split_right):
    valid = True
    for action in action_source:
        if action[2] == 'plural':
            condition1 = action_in((action[0],action[1],'singular'),action_split_left)
            condition2 = action_in((action[0],action[1],'singular'),action_split_right)
            condition3 = action_in((action[0],action[1],'plural'),action_split_left)
            condition4 = action_in((action[0],action[1],'plural'),action_split_right)
            result = (condition1 and condition2) or condition3 or condition4
            if not result:
                valid = False
    return valid

def mr_2(action_source,action_split_left,action_split_right):
    valid = True
    for action in action_source:
        if not (obj_eq(action[0],'person') or obj_eq(action[0],'animal')):
            continue
        if action[2] == 'singular':
            condition1 = action_in((action[0],action[1],'singular'), action_split_left)
            condition2 = action_in((action[0],action[1],'singular'), action_split_right)
            result = condition1 or condition2
            if not result:
                valid = False
    return valid

