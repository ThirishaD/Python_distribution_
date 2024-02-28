#DISTRIBUTIONS
#mean of the random values => peak of the curve
#standard deviation => how the peak of the curve deviated
#mean
#standard deviation
#random values
#myu - sd< myu + sd = 65%
#myu-2sd <x<myu+2sd=95%
#lam => number of occurences the event occurs

#uniform
#logistic
#from the graph or given dataset plot => we can makw decisions
#accurate values

# logistic => can make easy decisions than normal
#poisson => reverse -> exponential distribution
#chi-square => assumption(hypothesis) contradiction

#pareto
#80/20 => 80 for training purposes,20 percent testing
#uses shape of arguments,max number of times occured

#binomial distribution
import  seaborn as sb
import matplotlib.pyplot as plt
from numpy import random



#normal distribution
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
x=random.normal(loc=10,scale=5,size=500)
sb.distplot(x,hist=False)
plt.show()

#poisson
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.poisson(lam=3,size=1000)
sb.distplot(a,kde=False)
plt.show()

#uniform
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.uniform(size=(3,3))
sb.distplot(a,hist=False)
plt.show()

#logistic
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.logistic(loc=1,scale=2,size=1000)
sb.distplot(a,hist=False)
plt.show()

#multinomial
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.multinomial(n=6,pvals=[1/6,1/6,1/6,1/6,1/6,1/6],size=10)
sb.distplot(a)
plt.show()

#multi without graph
from numpy import random
a=random.multinomial(n=6,pvals=[1/6,1/6,1/6,1/6,1/6,1/6],size=10)
print(a)

#exponential
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.exponential(size=1000)
sb.distplot(a,hist=False)
plt.show()

#chi-square
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.chisquare(df=1,size=1000)
sb.distplot(a,hist=False)
plt.show()

#pareto
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
a=random.pareto(a=2,size=1000)
sb.distplot(a,kde=False)
plt.show()

#pandas

import pandas as pd
ds=[90,95,89,65,88]
df=pd.Series(ds)
print(df)
print(df[2])


import pandas as pd
ds=[90,95,89,65,88]
df=pd.Series(ds,index=['T','E','M','S','SS'])
print(df['M'])
import pandas as pd
ds={'T':90,'E':95,'M':89,'S':65,'SS':88}
df=pd.Series(ds)
print(df)

import pandas as pd
ds={'T':90,'E':95,'M':89,'S':65,'SS':88}
df=pd.Series(ds,index=['M','S'])
print(df)

import pandas as pd
ds={'names':['Tushar','Saran','Madhan','Sabari'],'marks':[95,89,93,98]}
df=pd.DataFrame(ds)
print(df)
print(df.loc[1])
print(df.loc[[1,2]])


import pandas as pd
ds={'Tam':[90,86,98,100,87],'Eng':[95,89,93,98,79],'Mat':[89,95,79,90,100]}
df=pd.DataFrame(ds,index=['Test1','Test2','Test3','Test4','Test5'])
print(df)


import pandas as pd
ds={'Tam':[90,86,98,100,87],'Eng':[95,89,93,98,79],'Mat':[89,95,79,90,100]}
df=pd.DataFrame(ds,index=['Test1','Test2','Test3','Test4','Test5'])
print(df.loc['Test2'])

import pandas as pd
df=pd.read_csv('D:\PY\customers1-100.csv')
print(df)

#points marking graph
import matplotlib.pyplot as plt
import numpy as np
#It takes x points as [0,1,2,3]
y=np.array([2,3,8,10])
plt.plot(y,'D-g')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
#It takes x points as [0,1,2,3]
y=np.array([2,3,8,10])
plt.plot(y,'D--r')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
#It takes x points as [0,1,2,3]
y=np.array([266,31,23,14])
plt.plot(y,'o-.y')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
y=np.array([2,3,8,10])
plt.plot(y,'o-.g',ms=10,mec='b',mfc='r') # mec => marker edge color ms=> marker size mfc=> marker fill color
plt.show()

import matplotlib.pyplot as plt
import numpy as np
y=np.array([2,3,8,10])
plt.plot(y,'o-.g',ms=10000,mec='#ff00ff',mfc='#ffff00')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
y=np.array([2,3,8,10])
plt.plot(y,linestyle='dashed')
plt.show()
print()

import matplotlib.pyplot as plt
import numpy as np
y1=np.array([2,3,8,10])
y2=np.array([1,4,9,12])
plt.plot(y1,linestyle='-',c='b')
plt.plot(y2,linestyle='-',c='r')
plt.show()
print()

import matplotlib.pyplot as plt
import numpy as np
x1=np.array([1,2,3,4,5])
x2=np.array([1,2,3,4,5])
y1=np.array([14,18,8,10,9])
y2=np.array([10,8,12,12,15])
plt.plot(x1,y1,x2,y2)
plt.xlabel('overs')
plt.ylabel('runs')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x1=np.array([1,2,3,4,5])
x2=np.array([1,2,3,4,5])
y1=np.array([14,18,8,10,9])
y2=np.array([10,8,12,12,15])
plt.plot(x1,y1,x2,y2)
plt.xlabel('overs')
plt.ylabel('runs')
plt.show()





