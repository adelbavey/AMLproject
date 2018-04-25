from multiprocessing import Pool

import time

def F(n):
    if n == 0:
    	return 0
    elif n == 1:
    	return 1
    else:
    	return F(n-1)+F(n-2)

n = 35

start = time.time()
print([F(i) for i in [n,n,n,n,n,n,n,n]])
end = time.time()
print("normal time: ",end - start)



pool = Pool()
start = time.time()
print(pool.map(F, [n,n,n,n,n,n,n,n]))
#print([pool.apply(F, (i,)) for i in [n,n,n,n,n,n,n,n,n,n]])
end = time.time()
pool.close()
pool.join()

print("multi time: " ,end - start)
