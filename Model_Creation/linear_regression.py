import numpy

params = [0,0,0]
samples = [[1,1,1],[1,2,2],[1,3,3],[1,4,4],[1,5,5]]
y = [2,4,6,8,10]

def hypothesis(params, samples):
    return numpy.dot(samples,params)

print(hypothesis(params, samples))