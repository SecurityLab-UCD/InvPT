a, b = (3, 5)
inner_flag = True
if a > 2:
    print('ture')
    if not inner_flag:
        print('inner false')
    else:
        print('inner true')
elif a > 4 or b > 0:
    print('a bigger than 4')
    if not inner_flag:
        print('inner false')
    else:
        print('inner true')
elif not (b > 4 and a < 4):
    print('last else statement')
    print('last else statement2')
    if not inner_flag:
        print('inner false')
    else:
        print('inner true')
else:
    print('b is biger than 4')