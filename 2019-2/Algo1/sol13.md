# Homework Assignment 13
### Algorithms 1 Summer 2019
### ID: 011862141
#### Aug 27th, 2019
---


## Question 1
First let's recap the mapping of a flow in $G$ to a flow in $G'$:
- For edges $(u,v) \in E \subset E'$: The flow $f(u,v)$ and the capacity $c((u,v))$ are the same as in $G$.
- For edges $(u_{in},u_{out})$: 
    - $f(u_{in},u_{out})=|f|(u)$
    - $c((u_{in},u_{out}))=b(u)$

Claim: This is a valid flow in $G'$.
Proof: We go over the three rules of a valid flow
##### Symmetry
Symmetry is maintained for $(u,v) \in E$ by construction.
Symmetry is not contradicted for $(u_{in},u_{out})$ since a node is either of type "in" or "out". So symmetry is fulfilled by setting $f(u_{out},u_{in}) =- f(u_{in},u_{out})$.

##### Conservation
Define a set $A$ of all "in"-nodes and a set $B$ of all "out" nodes in $G'$. $A \cup B \cup \{s,t\}= V',\ A \cap B=\empty$. 
for $u_{in{}} \in A$ there exists a unique $u_{out} \in B$ such that $(u_{in},u_{out})\in E'$.
We note the following **Lemma**: In a graph $G=(V,E)$ with flow function $f$:
$excess\_flow(v)=\sum_{u \in V}f(u,v)=\sum_{\{u|(u,v) \in V\}}f(u,v)-\sum_{\{u|(v,u) \in V\}}f(v,u)$.
In words, this lemma states that the excess flow (which must be zero for a flow to be valid), is the difference between the total _incoming_ flow and the total _outgoing_ flows for that node. 

**Proof**:
$\sum_{u \in V}f(u,v)=\sum_{\{u|(u,v) \in V\}}f(u,v)+\sum_{\{u|(v,u) \in V, (u,v) \notin V\}}f(u,v) + \sum_{\{u|(u,v)\notin V,(v,u) \notin V\}}f(u,v)$ breakdown of the sum into 3 disjoint groups according to wheter v touches u, u touches v or there is no edge between them.
$=\sum_{\{u|(u,v) \in V\}}f(u,v)+\sum_{\{u|(v,u) \in V\}}f(u,v)$ since a pair of nodes has nonzero flow only if there is an edge connecting them.
$=\sum_{\{u|(u,v) \in V\}}f(u,v)-\sum_{\{u|(v,u) \in V\}}f(v,u)$ by symmetry.




We now apply this lemma in turn to $v_{in}\in A$ and  $v_{out} \in B$ in $G'$:

for $v_{in} \in A:$ there is a single outgoing edge so 
$\sum\limits_{v\in V' }f(u,v_{in})= \sum\limits_{\{u|(u,v)\in E\}}f(u,v_{in}) - f(v_{in},v_{out})$ 
Similarly for $v_{out} \in B:\ \sum\limits_{v\in V' }f(u,v_{out})= f(v_{in},v_{out}) - \sum\limits_{\{u|(v_{out},u)\in E\}}f(v_{out},u)$ 

For conservation to hold on $v_{in}$: $f(v_{in},v_{out})=\sum\limits_{\{u|(u,v)\in E\}}f(u,v_{in})$.

For conservation to hold on $v_{out}$: $f(v_{in},v_{out})=\sum\limits_{\{u|(v_{out},u)\in E\}}f(v_{out},u)$ 

So for the law of conservation to hold for all nodes in $A \cup B$ we need, for every $(v_{in},v_{out})$ pair in $V'$:
 $\sum\limits_{\{u|(u,v)\in E\}}f(u,v_{in})= f(v_{in},v_{out}) = \sum\limits_{\{u|(v_{out},u)\in E\}}f(v_{out},u)$.
We claim the quantities on the left and right are equal by construction, and we use the equation above to _define_ the flow  $f(v_{in},v_{out})$ in $G'$.
Proof: 
Conservation on a node $v$ in $G$:
$\sum\limits_{\{u|(u,v)\in E\}}f(u,v)-\sum\limits_{\{u|(v,u)\in E\}}f(v,u)=0$ This is the breakdown we saw in the Lemma.
By construction, in $G'$ this relationship "maps" to:
$\sum\limits_{\{u|(u,v_{in})\in E\}}f(u,v_{in})-\sum\limits_{\{u|(v_{out},u)\in E\}}f(v_{out},u)=0$, moving the second expression to the righthand side proves the claim.

#### Capacity
For edges $(u,v) \in E \subset E'$ both flow and capicity are maintained so capacity rule is held.
For edges $(v_{in},v_{out})$ since in $G$, $|f_G|(v) \le b(v)$ we have $f_{G'}(v_{in},v_{out}) \equiv |f_G|(v) \le b(v)=c_{G'}(v_{in},v_{out})$.&square;


With the validity of the construction proven, it is straightforward to show that $|f_G|=\sum_{v\in V}f(s,v)=|f_{G'}|$

<div dir='rtl'>

## שאלה 2א
#### כיוון &Rightarrow;
בהנתן גרף עם k מסלולים זרים, נזרים בכל מסלול $t \leadsto s$ זרימה של 1. מכיוון שהקיבולת של כל הקשתות היא 1, הקשתות דרכן עוברת הזרימה רוויות. ע"מ לחפש מסלול שיפור, נסיר את הקשתות דרכן עוברת זרימה. מכיוון שהזרמנו על כל אחד מהמסלולים הזרים, אחרי הסרת הקשתות עליהן הזרמנו, לא ישאר מסלול נוסף בגרף, ולכן לא נוכל למצוא מסלול שיפור. ע"פ למה 8, עובדה זו מוכיחה כי זוהי זרימה מקסימלית.
אם יש כמה קבוצות של מסלולים זרים, נחזור על השיטה לכל קבוצה של מסלולים זרים בגרף. לפי ההנחה הזרימה המקסימלית שתתקבל היא k כיוון שזהו מספר המסלולים הזרים הגדול ביותר.

#### כיוון &Leftarrow;
נתונה זרימה k בגרף קיבולת יחידה. נגדיר חתך S={s}, T=V\s.
לפי הגדרת הזרימה |f| בחתך זה יש זרימה k. מכיוון שהקיבולת של קשת היא 1, מכאן שיש לפחות k קשתות שונות עליהן יש זרימה היוצאת מs. כעת נבצע "צעד עדכון חתך" המוגדר כצירוף ל-S של כל הקודקודים בצד T המחוברים בקשת לקודקוד בצד S. לפי למה 6, הזרימה בחתך היא k ולכן יש לפחות k קשתות עם זרימה בחתך זה. מתוך שימור זרימה, קשתות אלו מחוברות בקודקוד לקשתות בהן יש זרימה בחתך הקודם, כך שלכל קשת בחתך הקודם יש לפחות קשת אחת שמחוברת אליה בחתך החדש.
ניתן להמשיך להעביר קודקודים לצד S בחתך ובכל פעם אנו נמצא לפחות קשת אחת שמחוברת לכל אחת מk הקשתות בחתך הקודם. מסופיות הגרף, אם נמשיך כך נצרף את כל הקשתות על מסלול זרימה כלשהו מ s ל t. כאשר ל שרשורי הקשתות שנוצרו בתהליך הם k מסלולים זרים $t \leadsto s$.

</div>

## Question 2b.
This graph has at most $\frac{|E|}{\ell}$ edge-disjoin paths from _s_ to _t_. As shown in 2a above, the flow in a unit-capacity graph is up to the number of disjoint paths so $k \le \frac{|E|}{\ell} \lt \frac{|E|}{\ell-1}$

## Question 2c.

Claim: $\forall V \in V \setminus \{s,t\}:\ |f(v)| \le 1$
Proof: by definition, $|f(v)| \le \sum \limits_{(u,v)\in E}c((u,v))$ and $|f(v)| \le \sum \limits_{(v,u)\in E}c((v,u))$.
Since $\forall (u,v):\ c((u,v))=1$ and at least one of the sums has a single element, then $|f(v)| \le 1$.


So by feeding the same network to the algorithm _A_ which finds the maximum flow in a network with restriction 1 on the nodes, the algorithm will find the correct solution for G.

<div dir='rtl'>

## שאלה 2 ד
- מספר השכנים של s הוא <p dir='ltr'> $n(s) \ge k$  </p> כי דרך כל קודקוד שמחובר לs ניתן להזרים לכל היותר 1. השכנים של s הם במרחק של 1 מ s.
- נגדיר חתך <p dir='ltr'> $S=\{s\} \cup n(s), T=V\setminus S$  </p> הזרימה דרך חתך זה היא k כלומר היא מגיעה ללפחות k קודקודים שונים בצד T  של החתך (והנמצאים במרחק מקסימלי 2 מ k). בשלב 2 נעביר גם אותם לצד S בחתך.
- אחרי <p dir='ltr'> $\frac{|V|-2}{k}$ </p> שלבים לכל היותר כל הקודקודים למעט t יהיו בצד S. מצד שני כדי שתהיה קשת ל t נדרש שהמרחק s ל t יהיה <p dir='ltr'> $\delta(s,t) \le \frac{|V|-2}{k}+1$ </p> ומכאן <p dir='ltr'> $\ell -1 \le \frac{|V|-2}{k}$
או
$k \le \frac{|V|-2}{\ell-1}$

</div>

## Question 3a - increase in capacity of an edge
We call this edge $e$. 

If $|f|(e) \lt c(e)$ before the change, then $e$ is an edge in the residual graph $G_f=(V_f,E_f)$. By Theorem 8, $G_f$ has no path $s\leadsto t$ and so if we just change the capacity of $e$, there is no change in $E_f$. By Theorem 8, the current flow is the maximum flow.

If $|f|(e) = c(e)$ before the change, then we apply the change and construct the new $G_f$, $G_f'$. We search for a path $s \leadsto t$ in $G_f'$ using DFS and report the first path found. The capacity of the path must be greater than 0, and since all capcities are in whole numbers, then (by the 3rd statement of Lemma 8), the previous flow was a whole number and the augmenting path has a capacity of at least 1. So we can update the flow of a single path, causing edge $e$ to become saturated again and eliminating it from $G_f'$. Since the new graph has the same edges of $G_f$ or fewer, there is no augmenting path and By Theorem 8, this is a max flow.

#### Running time
- Changing the capacity $O(1)$
- Constructing $G_f$ requires $O(1)$ operations for each node and edge (for edge, comparing its |f| and c) so total time is $O(|V|+|E|)$
- Running DFS is $O(|V|+|E|)$.
- Updating the flow of a single path is $O(|V|+|E|)$.
- The total is $O(|V|+|E|)$.

## Question 3b - decrease in capacity of an edge
If $|f|(e) \le c(e)-1$ before the change, then the flow is still valid after the change. The residual graph $G_f$ had no path $s\leadsto t$. There are no additional edges that were not in $G_f$ and so the current flow is still the maximum.

If $|f|(e) \gt c(e)-1$, then the current flow becomes invalid due to the change. We run DFS in $G$, and report the first path we find $s\leadsto t$ that goes through $e$. We subtract 1 from the flow $f$ of each edge on the path. The new flow maintains the laws of symmetry, conservation, and maximum capacity, as can be seen directly from their definitions. The new flow is valid for the new capacity on $e$.

#### Running time
- Running DFS is $O(|V|+|E|)$.
- Updating the flow of a single path is $O(|V|+|E|)$.
- The total is $O(|V|+|E|)$.



## Question 4
- Calculate $G^{SCC}$, the graph of the strongly connected components of graph $G$, runs in $O(|V|+|E|)$.
- Run topological sort on $G^{SCC}$ in $O(|V|+|E|)$.
- Add nodes without edges until the size of the graph is a power of 2. At most, this adds |V|-1 edges and takes $O(|V|)$


To calculate $G^\ast$, the TC of $G$, from $G^{SCC^*}$, add an edge in $E^*$ for every two edges in the same SCC, and an edge $(u,v)$ between nodes in different SCC if in $E^{SCC^*}$ there is an edge between their corresponding SCCs.
