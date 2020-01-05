import chumpy as ch

x = ch.zeros(10)
y = ch.zeros(10)

# Beale's function
e1 = 1.5 - x + x*y
e2 = 2.25 - x  + x*(y**2)
e3 = 2.625 - x + x*(y**3)

objective = {'e1': e1, 'e2': e2, 'e3': e3}
ch.minimize(objective, x0=[x,y], method='dogleg')
print(x) # should be all 3.0
print(y) # should be all 0.5
