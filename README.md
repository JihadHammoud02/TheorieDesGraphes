# Network X and numpy
## Usage for Markov chain

### 
```python
import numpy as np
from numpy.linalg import matrix_power
from numpy import linalg as LA

np.set_printoptions(precision=4,suppress=True)

# Create matrix
P=np.array([[1,2,3]
           ,[4,5,6],
            [7,8,9]])

P10=matrix_power(P,10) # P to the power 10

print(P10[2][0]) # 3eme ligne premiere colonne (7)

# produit entre pi et matrice 
# pour trouver etat de convergence

pi00=[1 0 0]
print(P10 @ pi00)

# Method of eigenvectors to find state of convergence

val,vec=LA.eig(np.matrix.transpose(P))

print(val)
print(vec)
# We choose the vector of column where val=1 
#(let's say in the column j val=1)
pstat=vec[:,j]
pStat=pstat/np.sum(pstat)

```

### Linear programming dependencies
### Method 1:

```python
import numpy as np
from numpy.linalg import matrix_power
from scipy.optimize import linprog

```

#### declare decision variables
###### Decision variables: x1 , x2

###### f(x1,x2)=4x1+12x2
```python
x=[-4,-8] # use the objective function, python minimizes coefficient so we place - to minimize it
```
### Formula: A.x (<,=,>) B 

#### Constraints:
###### g1(x1,x2)= x1-1000<0


###### g2(x1,x2)=x2-500<0

###### g3(x1,x2)=x1+2x2-1750<0

###### g4(x1,x2)=-x1<0
###### g5(x1,x2)=-x2<0

```python
A=[[1,0],[0,1],[1,2],[-1,0],[0,-1]]
x=[4,8]
B=[1000,500,1750,0,0]
res = linprog(x, A_ub = A, b_ub = b)
print(res)
```
```bash
 message: Optimization terminated successfully. (HiGHS Status 7: Optimal)
        success: True
         status: 0
            fun: -7000.0
              x: [ 7.500e+02  5.000e+02]
            nit: 0
          lower:  residual: [ 7.500e+02  5.000e+02]
                 marginals: [ 0.000e+00  0.000e+00]
          upper:  residual: [       inf        inf]
                 marginals: [ 0.000e+00  0.000e+00]
          eqlin:  residual: []
                 marginals: []
        ineqlin:  residual: [ 2.500e+02  0.000e+00  0.000e+00  7.500e+02
                              5.000e+02]
                 marginals: [-0.000e+00 -0.000e+00 -4.000e+00 -0.000e+00
                             -0.000e+00]
```


###### you can find the most optimal x1 and x2 in x

### Method 2:
###### We can use bound to limit the value of x1 and x2 in a domain. In that case x1 < X and x2< Y can be removed from constraints they will be added as boundaries to the function ```linprog```

###### g4 => x1>0 and g5 => x2>0 and g1=> x1<1000 g2=> x2<500

```python
A=[[1,2]]
b=[1750]
bounds[(0,1000),(0,500)]
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print(res)
```

