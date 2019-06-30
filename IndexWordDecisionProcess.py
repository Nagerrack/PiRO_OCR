def decisionProcess(data, params):
    return gate1(data, params)


def gate1(data, params):
    if data[1] > params[0]:
        return gate4(data, params, 0)
    else:
        return gate4(data, params, 1)


def gate4(data, params, acceptance):
    if data[2] - params[1] < -(0.3 + 0.12 * acceptance) * params[1]:
        return gate6(data, params, acceptance)
    else:
        return gate7(data, params, acceptance)


def gate6(data, params, acceptance):
    if abs(data[3] - params[1]) < 0.3 * params[1]:
        return 'merge'
    else:
        return gateExtra(data, params, acceptance)


def gateExtra(data, params, acceptance):
    if data[0] > (2 + acceptance) * params[2]:
        return 'remove'
    else:
        return 'merge'


def gate7(data, params, acceptance):
    if data[0] > (2 + acceptance) * params[2]:
        return gate8(data, params, acceptance)
    else:
        return 'accept'


def gate8(data, params, acceptance):
    if abs(data[2] - params[1]) > (0.19 + 0.11 * acceptance) * params[1]:
        return gate6(data, params, acceptance)
    else:
        return 'accept'
