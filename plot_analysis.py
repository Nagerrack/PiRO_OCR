from collections import deque
import numpy as np

def get_val_param(val):
    bonus = 0.08
    if val == 3:
        bonus += 0.03
    if val ==1:
        bonus += 0.02
    if val==8:
        bonus*=2
    return bonus/6

def get_val_value_param(val, max, step, length):
    bonus = 0
    if max == 10:
        bonus -= 0.2
        if length-step < 6:
            bonus-=0.15
    if max==3:
        if val>0.55:
            bonus+=min(0.2, val-0.55)/1.5

    if max == 7:
        if length - step < 6:
            bonus -= 5
    if max==1:
        if length-step<6:
            bonus -=  5
        if length-step<8:
            bonus -=  min(1.5, 0.15 * np.max([0,(10-(length-step))]))

        elif step>10:
            bonus += 0.14
    if max==5:
        if val>0.5:
            bonus += 0.08
        if val>0.8:
            bonus+=0.13
    if val > 0.8:
        bonus += (val-0.8)*1.5
    if val>0.9:
        bonus += 0.15
    if val>0.95:
        bonus += 0.25
    if val<0.8:
        bonus -= (0.8-val)*2
    return bonus




def analyze(vals, maxes):
    get_prob = -0.5
    step = 0
    length = len(vals)
    window = deque(maxlen=5)
    max = 10
    last_drawn = 10
    val = 1.0
    steps_last_drawn =0
    drawn = []
    decisions = []
    while(len(vals)>0):
        last_val = val
        val = vals.pop(0)
        last_max = max
        max = maxes[step]
        gradient= val-last_val
        if len(vals) > 1:
            window.append(maxes[step+1])
        current_bonus =0
        if last_max == max:
            current_bonus +=0.075
        else:
            current_bonus-=0

        val_value_param = get_val_value_param(val, max, step,length)

        step += 1
        steps_last_drawn += 1

        increase_steps = 0
        if max==last_drawn:
            increase_steps+=4
        if steps_last_drawn <6+increase_steps:
            basic = 0.07
        else:
            basic = 0.09
        if steps_last_drawn < 6+increase_steps/2:

            get_prob += basic
            decisions.append('b')
        else:
            if last_drawn==max:
                current_bonus-=0.05
            if last_drawn == 8 and max==3:
                current_bonus-=  np.max([ (0.8-steps_last_drawn*0.06), 0])
            current_bonus += (abs(gradient) ** (1.2) if gradient > 0 else -abs(gradient)**2)
            current_bonus += get_val_param(
                max) * window.count(max)**2.5
            current_bonus += val_value_param
            reach = min(4, (length - 6) - step)
            if reach >2:
                uniq, counts = np.unique(maxes[step:step+reach], return_counts=True)

                coll = sorted(zip(uniq, counts), key=lambda x:x[1], reverse=True)
                if coll[0][0] != max and coll[0][0] != 10 and coll[0][1] > window.count(max):
                    #current_bonus -= 0.05
                    mult = 1
                    if max ==1:
                        mult = 0.3
                    current_bonus -= (coll[0][1]-window.count(max)) * 0.25 * mult
                    if any(vals[0:reach]) > val:
                        current_bonus -= (0.2 + np.max(vals[0:reach]) - val) * 1.4* mult
                        # print(step)
                        # print((0.2 + np.max(vals[0:reach]) - val) * 1.4)


                    grad1 = vals[0] - val
                    grad2 = vals[1] - vals[0]

                    if grad1 < -0.4 or grad2<-0.4 or grad1+grad2 < -0.5:
                        current_bonus+=0.35* mult

            get_prob += basic + current_bonus





            if get_prob > 1.0:
                if max != 10:
                    drawn.append(max)
                    get_prob=0.0
                    steps_last_drawn =0
                    last_drawn = max
                    decisions.append('r')
                else:
                    get_prob -= 0.05
                    decisions.append('b')
                    get_prob -= current_bonus
            else:
                decisions.append('b')
                get_prob -= current_bonus




    # print(drawn)
    return drawn, decisions