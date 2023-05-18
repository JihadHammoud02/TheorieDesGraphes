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
