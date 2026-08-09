# Mathematical Foundations of Machine Learning: Scalars, Vectors, Matrices, and Eigenvalues

## Executive Summary
- This module covers the **key mathematical concepts** used in machine learning: **scalars**, **vectors**, **matrices**, and **eigenvalues/eigenvectors**. Each concept is defined and illustrated with clear LaTeX formulas and intuitive examples.  
- We explain how **vectors** and **matrices** represent data and transformations.  Feature vectors (arrays of numeric attributes) are introduced, with citation. We summarize **vector operations** (addition, scaling, dot product) and **matrix operations** (addition, multiplication, transpose) in tables.  
- We compute **determinants** step by step for example matrices, and derive **eigenvalues/eigenvectors** via the characteristic equation $\det(A-\lambda I)=0$. All derivations are shown in LaTeX with intermediate steps.  
- Visual aids include a diagram of a linear transformation and a flowchart linking the concepts. At the end we include practice problems with full solutions.  

---

## Scalars and Vectors

A **scalar** is a single number (a one-dimensional quantity).  In ML, scalars often represent individual measurements (e.g. age = 25).  We denote scalars with regular font (e.g. $a=3.14$).  As one source notes, “we call a number a scalar”. 

A **vector** is an ordered list of numbers. For example, a 3-dimensional vector can be written 
```latex
\[
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix},\quad \mathbf{x}\in\mathbb{R}^3,
\]
``` 
meaning it has three entries $x_1,x_2,x_3$.  More generally, an $n$-dimensional vector is $\mathbf{x}\in\mathbb{R}^n$ with components $x_1,\dots,x_n$.  In machine learning, a **feature vector** collects the numeric attributes of one data point. For instance, a student’s record with (Age, Salary, Experience) = (25, 50000, 2) can be represented as 
```latex
\[
\mathbf{x} = \begin{bmatrix} 25 \\ 50000 \\ 2 \end{bmatrix}.
\]
``` 
This $3\times1$ column is an example of a feature vector in $\mathbb{R}^3$.

- **Dimension:** The *dimension* of a vector is the number of entries it contains.  A vector in $\mathbb{R}^n$ has dimension $n$.  For $\mathbf{x}\in\mathbb{R}^3$ above, $\dim(\mathbf{x})=3$.  

## Vector Operations

We can combine and manipulate vectors using familiar algebraic operations:

- **Vector addition:** If $\mathbf{a}=[a_1,a_2,\dots,a_n]^T$ and $\mathbf{b}=[b_1,b_2,\dots,b_n]^T$, then their sum is 
  ```latex
  \[
  \mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1+b_1 \\ a_2+b_2 \\ \vdots \\ a_n+b_n \end{bmatrix}.
  \]
  ``` 
  For example, 
  ```latex
  \[
  \begin{bmatrix}2\\3\end{bmatrix}
  +
  \begin{bmatrix}4\\5\end{bmatrix}
  =
  \begin{bmatrix}6\\8\end{bmatrix}.
  \]
  ```
  (This matches the definition in.) Each entry adds component-wise.

- **Scalar multiplication:** Multiplying a vector by a scalar $c$ scales each component: 
  ```latex
  \[
  c\,\mathbf{a} = c 
  \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}
  = 
  \begin{bmatrix} c a_1 \\ c a_2 \\ \vdots \\ c a_n \end{bmatrix}.
  \]
  ``` 
  For instance, if $\mathbf{a}=[2,4]^T$ and $c=3$, then
  ```latex
  \[
  3\,\begin{bmatrix}2\\4\end{bmatrix} 
  = 
  \begin{bmatrix}6\\12\end{bmatrix}.
  \]
  ```
  Scalar multiplication changes the *magnitude* but not the “direction” of the vector (it stretches or shrinks it).

- **Dot product (scalar product):** The dot product of two vectors $\mathbf{a}=[a_1,\dots,a_n]^T$ and $\mathbf{b}=[b_1,\dots,b_n]^T$ is defined by 
  ```latex
  \[
  \mathbf{a}\cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n,
  \]
  ``` 
  which yields a scalar.  Geometrically, it measures the degree of alignment between $\mathbf{a}$ and $\mathbf{b}$.  For example,
  ```latex
  \[
  \begin{bmatrix}2\\3\end{bmatrix} \cdot \begin{bmatrix}5\\1\end{bmatrix} 
  = (2)(5) + (3)(1) = 10 + 3 = 13.
  \]
  ``` 

**Table 1. Vector operations (in $\mathbb{R}^n$).**

| Operation             | Formula                                         | Example                                   |
|-----------------------|-------------------------------------------------|-------------------------------------------|
| Vector addition       | $\mathbf{a}+\mathbf{b} = [\,a_i+b_i\,]_i$       | $\begin{bmatrix}2\\3\end{bmatrix}+\begin{bmatrix}4\\5\end{bmatrix}=\begin{bmatrix}6\\8\end{bmatrix}$ |
| Scalar multiplication | $c\,\mathbf{a} = [\,c\,a_i\,]_i$                | $3\begin{bmatrix}2\\4\end{bmatrix}=\begin{bmatrix}6\\12\end{bmatrix}$              |
| Dot product           | $\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i$    | $\begin{bmatrix}2\\3\end{bmatrix}\cdot\begin{bmatrix}5\\1\end{bmatrix}=13$         |

Each of these operations is fundamental.  For instance, adding feature vectors corresponds to combining the effects of two data points, and the dot product appears in many algorithms (e.g. linear regression uses $\mathbf{x}^T\mathbf{w}$).

## Matrices

A **matrix** is a rectangular array of numbers (or other elements) arranged in rows and columns. We denote an $m\times n$ matrix $\mathbf{A}$ with $m$ rows and $n$ columns. For example, 
```latex
\[
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\]
``` 
is a $2\times 3$ matrix (two rows, three columns). The general notation is $A_{ij}$ for the entry in row $i$, column $j$.

- **Rows and columns:** An $m\times n$ matrix has $m$ rows and $n$ columns.  We often use matrices to represent datasets: each **row** can be an observation (sample) and each **column** a feature.  For example, if we record (Math, Python, ML) scores for four students, we can arrange them as:  
  ```latex
  \[
  \mathbf{X} = 
  \begin{bmatrix}
  80 & 75 & 85\\
  60 & 70 & 65\\
  90 & 95 & 92\\
  70 & 80 & 75
  \end{bmatrix},
  \quad \mathbf{X}\in\mathbb{R}^{4\times 3}.
  \]
  ``` 
  Here $X$ has 4 rows (students) and 3 columns (features), so $4\times3$. 

- **Matrix types:**  Special kinds of matrices include: single-row (row) or single-column (column) matrices, square matrices (same number of rows and columns), the zero matrix (all entries zero), and the identity matrix. The identity matrix $I_n$ of size $n$ is a square matrix with 1’s on the diagonal and 0’s elsewhere.  For example, 
  ```latex
  \[
  I_2 = \begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}.
  \]
  ``` 
  Multiplying any $2\times2$ matrix by $I_2$ leaves it unchanged. 

**Table 2. Common types of matrices.**

| Type             | Size           | Example                                     |
|------------------|----------------|---------------------------------------------|
| Row matrix       | $1\times n$    | $\begin{bmatrix}1 & 2 & 3\end{bmatrix}$    |
| Column matrix    | $n\times 1$    | $\begin{bmatrix}1\\2\\3\end{bmatrix}$       |
| Square matrix    | $n\times n$    | $\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix}$ (2×2) |
| Zero matrix      | all zeros      | $\begin{bmatrix}0 & 0\\ 0 & 0\end{bmatrix}$ |
| Identity matrix  | diagonal 1’s   | $\begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}$ (2×2) |

Each entry of a matrix can be accessed by its row and column indices, e.g. $A_{23}$ is row 2, col 3.

### Matrix Operations

- **Matrix addition:** Two matrices $\mathbf{A}$ and $\mathbf{B}$ of the same dimensions ($m\times n$) add component-wise: $(A+B)_{ij}=A_{ij}+B_{ij}$. For example, if 
  ```latex
  A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad B=\begin{bmatrix}5&6\\7&8\end{bmatrix},
  ```
  then 
  ```latex
  A+B=\begin{bmatrix}6&8\\10&12\end{bmatrix}.
  ```

- **Transpose:** The transpose of $A$ (denoted $A^T$) swaps rows and columns: $(A^T)_{ij}=A_{ji}$. E.g. for 
  ```latex
  A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
  ```
  we have 
  ```latex
  A^T=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}.
  ```

- **Matrix–vector multiplication:** If $\mathbf{A}$ is $m\times n$ and $\mathbf{x}$ is an $n\times1$ column vector, then $\mathbf{A}\mathbf{x}$ is an $m\times1$ vector.  For instance, let 
  ```latex
  \[
  A=\begin{bmatrix}2&3\\4&5\end{bmatrix},\quad x=\begin{bmatrix}10\\20\end{bmatrix}.
  \]
  ``` 
  Then 
  ```latex
  \[
  A x 
  =\begin{bmatrix}2&3\\4&5\end{bmatrix}\begin{bmatrix}10\\20\end{bmatrix}
  =\begin{bmatrix}2\cdot10 + 3\cdot20 \\ 4\cdot10 + 5\cdot20\end{bmatrix}
  =\begin{bmatrix}80 \\ 140\end{bmatrix}.
  \]
  ``` 
  This expresses a linear transformation of the vector $x$.  

   *Figure: A linear transformation represented by matrix $\mathbf{A}$ acting on a vector $\mathbf{x}$.  Some special vectors (eigenvectors) keep their direction under the transformation.* 

- **Matrix multiplication:**  To multiply $A$ ($m\times n$) by $B$ ($n\times p$), the inner dimensions $n$ must match, and the result is $m\times p$.  The element $(AB)_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$. For example: 
  ```latex
  A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad B=\begin{bmatrix}5&6\\7&8\end{bmatrix}.
  ```
  Then 
  ```latex
  AB = \begin{bmatrix}
     1\cdot5 + 2\cdot7 & 1\cdot6 + 2\cdot8 \\[6pt]
     3\cdot5 + 4\cdot7 & 3\cdot6 + 4\cdot8
  \end{bmatrix}
  = \begin{bmatrix}19&22\\43&50\end{bmatrix}.
  ```
  In ML, matrix multiplication is used to combine features and weights, transform data, and more.

## Determinants

The **determinant** is a scalar value associated with a square matrix that reflects certain properties (e.g. invertibility).  For a $2\times2$ matrix 
$\begin{pmatrix}a&b\\c&d\end{pmatrix}$, the determinant is $ad - bc$.  We compute determinants with expansion formulas:

1. **Example:** $A_1 = \begin{bmatrix}3&4\\2&5\end{bmatrix}$.  Its determinant is 
   ```latex
   $$
   \det(A_1) = 
   \begin{vmatrix}3 & 4 \\ 2 & 5\end{vmatrix}
   = (3)(5) - (4)(2)
   = 15 - 8
   = 7.
   $$
   ``` 
   So $\det(A_1)=7$.  Because this is nonzero, $A_1$ is invertible.

2. **Example:** $A_2 = \begin{bmatrix}1&2&3\\0&4&5\\1&0&6\end{bmatrix}$.  We compute its $3\times3$ determinant by cofactor expansion along the first row:  
   ```latex
   $$
   \det(A_2) 
   = 1\cdot\det\begin{pmatrix}4&5\\0&6\end{pmatrix}
     - 2\cdot\det\begin{pmatrix}0&5\\1&6\end{pmatrix}
     + 3\cdot\det\begin{pmatrix}0&4\\1&0\end{pmatrix}.
   $$
   ``` 
   Compute each $2\times2$ minor:  
   - $\det\begin{pmatrix}4&5\\0&6\end{pmatrix} = 4\cdot6 - 5\cdot0 = 24$.  
   - $\det\begin{pmatrix}0&5\\1&6\end{pmatrix} = 0\cdot6 - 5\cdot1 = -5$.  
   - $\det\begin{pmatrix}0&4\\1&0\end{pmatrix} = 0\cdot0 - 4\cdot1 = -4$.  
   Plugging in:  
   ```latex
   $$
   \det(A_2) = 1(24) - 2(-5) + 3(-4)
             = 24 + 10 - 12
             = 22.
   $$
   ``` 
   Thus $\det(A_2)=22$.  

The determinant of a square matrix is especially important: as noted in linear algebra, “a square matrix is invertible if and only if it has a nonzero determinant” and *its eigenvalues are the roots of* $\det(\lambda I - A) = 0${}.  We will use this in the next section.

## Eigenvalues and Eigenvectors

An **eigenvector** of a square matrix $A$ is a special nonzero vector whose direction remains unchanged when $A$ is applied.  Mathematically, if $A\mathbf{v} = \lambda \mathbf{v}$ for some scalar $\lambda$ and nonzero vector $\mathbf{v}$, then $\mathbf{v}$ is an eigenvector and $\lambda$ is the corresponding **eigenvalue**.  Geometrically, $A$ *stretches* (or compresses) the eigenvector by factor $\lambda$ but does not rotate it. 

To find eigenvalues, we solve the **characteristic equation** $\det(A - \lambda I) = 0$.  The solutions $\lambda$ are the eigenvalues.  We illustrate with an example:

**Example:** Let 
```latex
\[
E = \begin{bmatrix}3 & 2\\ 3 & 8\end{bmatrix}.
\]
``` 
Find its eigenvalues and eigenvectors.  

1. **Characteristic polynomial:** Compute $\det(E - \lambda I)$:  
   ```latex
   $$
   \det\bigl(E - \lambda I\bigr)
   = \begin{vmatrix}3-\lambda & 2 \\ 3 & 8-\lambda\end{vmatrix}
   = (3-\lambda)(8-\lambda) - (2)(3).
   $$
   ``` 
   Expanding: $(3-\lambda)(8-\lambda) = 24 - 11\lambda + \lambda^2$, then subtract $6$:  
   ```latex
   $$
   = \lambda^2 - 11\lambda + 24 - 6 
   = \lambda^2 - 11\lambda + 18.
   $$
   ``` 
   So the characteristic equation is 
   ```latex
   $$
   \lambda^2 - 11\lambda + 18 = 0.
   $$
   ``` 
   Solve for $\lambda$: the roots are $\lambda = 2$ and $\lambda = 9$.  These are the eigenvalues.

2. **Eigenvectors:** For each eigenvalue, solve $(E - \lambda I)\mathbf{v} = \mathbf{0}$.  
   - For $\lambda=2$: 
     ```latex
     $$
     E - 2I = \begin{bmatrix}1 & 2\\3 & 6\end{bmatrix}.
     $$
     ``` 
     The system $(E - 2I)\,[x,y]^T=\mathbf{0}$ gives:
     \[
       \begin{cases}
         x + 2y = 0,\\
         3x + 6y = 0.
       \end{cases}
     \]
     Both equations are equivalent to $x + 2y=0$, so $x=-2y$.  A nonzero solution is 
     \[
       \mathbf{v}_{\lambda=2} = \begin{bmatrix}-2\\1\end{bmatrix}.
     \]
     (We often normalize to unit length: $\tfrac{1}{\sqrt{5}}[-2,1]^T$, but any scalar multiple is an eigenvector.)  
   - For $\lambda=9$: 
     ```latex
     $$
     E - 9I = \begin{bmatrix}-6 & 2\\3 & -1\end{bmatrix}.
     $$
     ``` 
     The equations $-6x+2y=0$ and $3x - y=0$ both reduce to $y=3x$.  A nonzero solution is 
     \[
       \mathbf{v}_{\lambda=9} = \begin{bmatrix}1\\3\end{bmatrix},
     \]
     normalized as $\tfrac{1}{\sqrt{10}}[1,3]^T$.

Thus, $E$ has eigenvalues $\lambda_1=2$ and $\lambda_2=9$ with corresponding eigenvectors $[-2,1]^T$ and $[1,3]^T$ (up to normalization).  

In summary, finding eigenvalues involves solving $\det(A-\lambda I)=0$ and then solving the linear system $(A-\lambda I)\mathbf{v}=0$.  These eigenvalues/eigenvectors characterize the transformation $A$ (they form the eigen-decomposition $A=V\Lambda V^{-1}$ when all eigenvectors are used). Such decompositions are fundamental in PCA, linear regression, and many ML techniques.

---

## Concept Map

```mermaid
flowchart LR
    Vectors --> Matrices
    Matrices --> Operations
    Operations --> Eigenvalues
```

*(Flowchart: How vectors and matrices lead to operations and eigenvalues.)*

## Practice Problems

**Problem 1:** Given $\mathbf{u}=[2,\, -1]^T$ and $\mathbf{v}=[-3,\,4]^T$ in $\mathbb{R}^2$, compute (a) $\mathbf{u}+\mathbf{v}$, (b) $3\,\mathbf{u}$, and (c) the dot product $\mathbf{u}\cdot\mathbf{v}$.  
**Solution:**  
(a) Add component-wise: 
```latex
\[
\mathbf{u}+\mathbf{v} = 
\begin{bmatrix}2\\-1\end{bmatrix}
+ \begin{bmatrix}-3\\4\end{bmatrix}
= \begin{bmatrix}2+(-3)\\ -1+4\end{bmatrix}
= \begin{bmatrix}-1\\3\end{bmatrix}.
\]
```  
(b) Scale $\mathbf{u}$ by 3: 
```latex
\[
3\,\mathbf{u} = 3\begin{bmatrix}2\\-1\end{bmatrix}
= \begin{bmatrix}6\\-3\end{bmatrix}.
\]
```  
(c) Dot product: 
```latex
\[
\mathbf{u}\cdot\mathbf{v} = (2)(-3) + (-1)(4) = -6 - 4 = -10.
\]
```  

**Problem 2:** For the matrix $A = \begin{bmatrix}4 & 1\\ 2 & 3\end{bmatrix}$, compute (a) its determinant and (b) its eigenvalues.  
**Solution:**  
(a) $\displaystyle \det(A) = (4)(3) - (1)(2) = 12 - 2 = 10$.  (Since $\det(A)\neq0$, $A$ is invertible.)  
(b) Solve $\det(A-\lambda I)=0$:  
```latex
\[
\det\begin{pmatrix}4-\lambda & 1\\ 2 & 3-\lambda\end{pmatrix}
= (4-\lambda)(3-\lambda) - (2)(1)
= (\lambda^2 - 7\lambda + 12) - 2
= \lambda^2 - 7\lambda + 10.
\]
``` 
Set $\lambda^2 - 7\lambda + 10 = 0$.  Factoring gives $(\lambda-5)(\lambda-2)=0$, so $\lambda=5$ or $\lambda=2$.  These are the two eigenvalues of $A$.  (Optional: one can then solve $(A-5I)\mathbf{v}=0$ or $(A-2I)\mathbf{v}=0$ to find corresponding eigenvectors.)

