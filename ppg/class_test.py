"""
Just playing around with how classes interact; scratch notebook
for grid world dev.
"""

class foo_1(object):

    def __init__(self, position):
        self.position = position

class foo_2(object):

    def __init__(self, count):
        self.test_list = []
        for i in range(count):
            foo_1_test = foo_1(i)
            self.test_list.append(foo_1_test)
