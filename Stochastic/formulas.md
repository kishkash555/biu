
### concetration
* Chebyshev $a\ge0\Rightarrow\mathbb{P}(|x-\mathbb{E}x|\ge a) \le \frac{\mathrm{Var}(x)}{x}$
* Markov $x\ge0 \land a\ge0\Rightarrow \mathbb{P}(x \ge a) \le \frac{\mathbb{E}x}{a}$
* Cauchy-Schwarz $\mathbb{E}(X\cdot Y) \le \sqrt{\mathbb{E}x^2 \cdot \mathbb{E}y^2}$
* Jensen $\Phi\textrm{ convex} \Rightarrow \mathbb{E}\Phi(x) \ge \Phi(\mathbb{E}x)$
* Chernoff $ a \gt 0 \land t \in \mathbb{R} \Rightarrow \mathbb{P}(x \ge a) \le \frac{\mathbb{E}e^{tx}}{e^{ta}}$
* Hoeffding $\{X_i\}: \mathbb{E}X_i=0, |X_i|\le 1, X_i \textrm{ independent}\\ \Rightarrow \mathbb{P}(\sum X_i \ge a) \le e^{a^2/2N}$

Conditions    |  Statement | $ $
:------------:|:----------:|-----
$x\ge0 \land a\ge0$ | $\mathbb{P}(x \ge a) \le \frac{\mathbb{E}x}{a}$ | Markov
 $a\ge0$      | $\mathbb{P}(\|x-\mathbb{E}x\|\ge a) \le \frac{\mathrm{Var}(x)}{a^2}$ | Chebyshev
 $\Phi\textrm{ convex}$ | $\mathbb{E}\Phi(x) \ge \Phi(\mathbb{E}x)$ | Jensen
$a \gt 0 \land t \in \mathbb{R}$ | $\mathbb{P}(x \ge a) \le e^{-ta}\mathbb{E}e^{tx}$ | Chernoff
 $\{X_i\} $ independent: | 
 $\|X_i\|\le 1 \land \mathbb{E}X_i=0$ | $\mathbb{P}(\sum X_i \ge a) \le e^{a^2/2N}$ | Hoeffding
 |$\{X_i\}$ i.i.d| $\frac{1}{n}\sum(X_i) \xrightarrow[]{a.s.} \mathbb{E}X_1$ | SLLN
 |$\{X_i\}$ i.i.d| $\frac{\sum(X_i -\mu)}{\sqrt{n}\sigma} \xrightarrow{d} \mathcal{N}(0,1)$ | CLT

### Convergence
$X_n \xrightarrow{a.s.}X: \mathbb{P}(\{\omega:X_n(\omega)\rightarrow X(\omega)\})=1$
$X_n \xrightarrow{\mathbb{P}}X: \forall \epsilon \gt 0 \textrm{, }\mathbb{P}(|X_n-X|\gt \epsilon) \rightarrow 0$
$X_n \xrightarrow{\mathbb{P}}X: \forall \epsilon \gt 0, \delta \gt 0 \exist N \textrm{ s.t. }\forall n \gt N,\mathbb{P}(|X_n-X|\gt \epsilon) \lt \delta$
$X_n \xrightarrow{d}X: \forall a \textrm{ point of continuity},\lim\limits_{n\rightarrow \infty} F_{x_n}(a) = F_x(a)$

### Distributions
* Geometric support: $k \in {1,2, ...}$ pdf: $(1-p)^{k-1}p$ cdf: $1-(1-p)^k$
* Exponential support: $x \in [0,\infty)$ pdf: $\lambda e^{-\lambda x}$ cdf: $1-e^{-\lambda x}$
* Binomial support: $ k \in \mathbb{N}$ pdf: ${n \choose k} p^k(1-p)^{n-k}$ cdf: $I_{1-p}(n-k, 1+k)$


$ $|Geometric | Exponential | Binomial | Normal
----|--------|---------|-----|---- | ----
sup | $k \in {1,2, ...}$ |$x \in [0,\infty)$         | $ k \in \mathbb{N}$           | $x \in \mathbb{R}$
pmf | $(1-p)^{k-1}p$     | $\lambda e^{-\lambda x}$ |  ${n \choose k} p^k(1-p)^{n-k} $ | $\exp(-{\frac{(x-\mu)^2}{2\sigma^2}})/\sqrt{2\pi\sigma^2}$
cdf| $1-(1-p)^k$ | $1-e^{-\lambda x}$ | $I_{1-p}(n-k, 1+k)$
Mean | $1/p$ | $\lambda^{-1}$ | $np$ | $\mu$ | 
Var | $(1-p)/p^2$ | $\lambda^{-2}$ | $np(1-p)$ | $\sigma^2$ |
##### Generating functions
* $\mathcal{M}_x(t) = \mathbb{E}(e^{tx})$
* $\mathcal{M}^{(k)}_x(0) = \mathbb{E}(x^{k})$   
* $\mathcal{M}_{x+y}(t) = \mathcal{M}_x(t) \mathcal{M}_y(t)$
* $\phi_x(t) = \mathbb{E}(e^{itx})$

###### A.E., I.O, Fatou
$ A_i\textrm{ a.e.} \triangleq \bigcup\limits_{m=1}^\infty\bigcap\limits_{k=m}^\infty A_k$
$ A_i\textrm{ i.o.} \triangleq \bigcap\limits_{m=1}^\infty\bigcup\limits_{k=m}^\infty A_k$

 $\mathbb{P}(\{A_i\textrm{ a.e.}\}) \le \lim\limits_{n\rightarrow\infty} \inf \mathbb{P}(A_n)$ 

 $\mathbb{P}(\{A_i\textrm{ i.o.}\}) \ge \lim\limits_{n\rightarrow\infty} \sup \mathbb{P}(A_n)$ 
##### Borel - Cantelli
* $\sum{\mathbb{P}(A_i)} \lt \infty \Rightarrow \mathbb{P}(\{A_i \textrm{ i.o.}\}) = 0 $
* $A_i$ independent, $\sum{\mathbb{P}(A_i)} = \infty \Rightarrow \mathbb{P}(\{A_i\textrm{ i.o.}\}) = 1 $ 

##### Filtration
$F_i \subseteq F_j \uparrow i\lt j $
$F\downarrow: F_\infty \triangleq \bigcap\limits_{j=1}^\infty F_j$
$ A \in F_\infty \Rightarrow \mathbb{P}(A) \in \{0,1\} $

$\sum\limits_{n=1}^{\infty}{\frac{1}{n^\alpha}}< \infty \Leftrightarrow \alpha \ge 1 + \epsilon$

$\mathbb{E}(XY) = \mathbb{E}(X)\mathbb{E}(Y)$

##### Markov
$\pi(x) = 1/ \mathbb{E}_x(T_x) $
$period(a)= \gcd\{n \ge 1: P^n(a,a)>0\}$
$x \textrm{ transient} \Leftrightarrow \pi(x)=0$
$\sum\limits_{i \in S} | (\mu P^k - \pi)(i)| \le c \alpha^k$
$\mathbb{P}_\mu(x_k = a) \xrightarrow{k \rightarrow \infty} \pi(a)$


##### Doob's optional stopping theorem
* $\tau \lt c$ a.s.
* $\mathbb{E}(\tau) \lt \infty$, $\mathbb{E}[|X_{t+1}-X_t| | F_k] \le c$, 
* $\exist c\  \forall t \in  \mathbb{N}: \ |X_{t\cap\tau}| \le c$
          $\Downarrow$
$ \mathbb{E}(X_\tau) =\mathbb{E}(X_0)$


 שאלה 8 (המשלושים בגרף):  התחלה בסדר גמור.
אז עכשיו צריך לחשב את התוחלת ואת השונות של Q (מספר המשולשים).
כמו שציינת, זה מתפרק לסכום של Y_i Y_j Y_k, אבל שים לב שאפשר להריץ את הסכום על פני כל ה-i,j,k השונים בכלל (ולא חייבים להצטמצם לקבוצה המקרית של משולשים). זאת כי אם i,j,k אינם מהווים משולש בגרף המקרי, הרי שאחד מהמשתנים המקריים Y_i, Y_j, Y_k  מתאפס ולכן המכפלה שלהם מתאפסת. כלומר, אני טוענת ש-

$Q = \sum_{i,j,k\in V different} Y_i Y_j Y_k$
ולכן מלינאריות התוחלת:
$E(Q) = \binom{n}{3} E(Y_1 Y_2 Y_3)$
כלומר, התוחלת של Q היא המקדם הבינומי n מעל 3 כפול ההסתברות של שלוש צלעות מסויימות להיות קיימות. ובמילים אחרות:
$E(Q) = \binom{n}{3} p^3$


$\mathbb{E}(X_{k+1}) = X_k(1-\frac{1}{a+b-k}) + \frac{a}{a+b-k}$