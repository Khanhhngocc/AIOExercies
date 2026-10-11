def factory(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factory(n-1)