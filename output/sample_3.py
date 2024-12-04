a, b = (3, 5)
inner_flag = True
if a > 2:
    print('ture')
    if inner_flag:
        print('inner true')
    else:
        print('inner false')
elif a > 4 or b > 0:
    print('a bigger than 4')
    if inner_flag:
        print('inner true')
    else:
        print('inner false')
elif b > 4 and a < 4:
    print('b is biger than 4')
else:
    print('last else statement')
    print('last else statement2')
    if inner_flag:
        print('inner true')
    else:
        print('inner false')