import numpy as np

def hourangle(angle):
    if type(angle) == str or type(angle) == np.string_:
        angle = [float(i) for i in angle.split(" ")]
    
    if len(angle) == 3:
        hours = angle[0] + angle[1]/60 + angle[2]/3600
    elif len(angle) == 2:
        hours = angle[0] + angle[1]/60
    return hours*360/24

def degree(angle):
    if type(angle) == str or type(angle) == np.string_:
        angle = [float(i) for i in angle.split(" ")]
    
    sign = angle[0]/abs(angle[0])
    if len(angle) == 3:
        return sign * (abs(angle[0]) + angle[1]/60 + angle[2]/3600)
    elif len(angle) == 2:
        return sign * (abs(angle[0]) + angle[1]/60)
    
def score(string, name):

    if len(name) == 0 or len(string) == 0:
        return -1

    score = 0.
    nm = list(name)
    for s in string:
        if s in nm:
            score += 1
            nm.remove(s)
        else:
            score -= 1
            pass
    
    return (score - len(nm)/2)/len(name)