# Memoization

import time

def fib(n):
    if n==1 or n==0:
        return 1
    else:
        return fib(n-1)+fib(n-2)
    


print(fib(40))

print("time ",time.time())

# above code is normally used but it is highly inefficient coz its time complexity is in Exponential so it takes more time as the number increases

# almost effiecient code :
def fib(n,d):
    if n in d:
        return d[n]
    
    else:
        d[n] = fib(n-1,d)+fib(n-2,d)
        return d[n]
    

d = {0:1,1:1}
print(fib(40,d))