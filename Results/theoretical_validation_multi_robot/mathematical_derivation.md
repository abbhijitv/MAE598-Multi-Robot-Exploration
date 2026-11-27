# Interacting-robot baseline: equilibrium and invariant neighborhood

## Scope and relation to the existing analysis

This is a separate two-robot example of the continuous baseline model from Step 8D and the two properties derived in `Research_Notes/Step_15_Theoretical_Analysis.md`. It does not replace the one-robot example or any exploration or Robotarium result. The original 3/5/8-robot exploration runs use discrete updates, caps and collision safeguards; their coverage results do not certify a smooth invariant neighborhood. The existing multi-robot force-gradient test verifies a model identity, rather than equilibrium existence or stability.

Two robots are used because reflection symmetry reduces equilibrium existence to one scalar root and gives all four collective Hessian eigenvalues explicitly. The certificate still includes genuine robot-to-robot coupling and arbitrary small perturbations in all four position and velocity coordinates. It is not restricted to symmetric trajectories. A three-robot certificate is not claimed; it would require a separate equilibrium and collective bound.

A candidate with two robots in the original 3 × 3 square was checked first. The symmetric diagonal equilibrium near (1.03796383,1.03796383), (1.96203617,1.96203617) had collective Hessian eigenvalues approximately -0.42905508, 0.74856413, 1.36318787, 4.33304762. The horizontal candidate also had a negative eigenvalue, approximately -1.23567446. Neither is a strict local minimum. Narrowing this separate theoretical workspace to 3 × 2 supplies enough transverse wall curvature; no force, stabilizer or controller change is introduced.

## 1. Configuration and potential

The workspace is [0,3] × [0,2], with two radius-0.2 robots, no circular obstacles and the existing 0.5-unit wall-sampling procedure. There are 20 distinct wall samples. Use robot gain k_r=2, wall gain k_o=0.35, epsilon=10^-6, interaction ranges R_r=4 and R_o=2, mass m=1, and damping b=1.5. These are an explicit separate configuration, not changes to `Config()` defaults. The seed is unused because equilibrium and perturbations are specified deterministically.

For an active source at distance r, define

$$
\phi_{k,R}(r)=k\int_r^R\frac{s}{s^3+\epsilon}\,ds,\qquad
\phi_{k,R}(r)=0\quad(r\geq R).
$$

The collective potential counts the robot pair once:

$$
U(X)=\phi_{k_r,R_r}(\|x_1-x_2\|)+
\sum_{i=1}^2\sum_o\phi_{k_o,R_o}(\|x_i-o\|).
$$

For positive source distances strictly away from the cutoffs, this is smooth and its negative gradient is exactly `baseline_forces`. The ODE is

$$
\dot X=v,\qquad m\dot v=-\nabla U(X)-bv.
$$

There are no velocity or acceleration caps, position projections, collision resets, discovery targets or random inputs. Finite epsilon does not supply an infinite collision barrier. Collision freedom below follows from the invariant neighborhood, not from regularization alone.

## 2. Existence of an interacting equilibrium

Put x_1=(a,1), x_2=(3-a,1), with zero velocities. Reflection in y=1 cancels each robot's vertical force exactly. Reflection in x=1.5 makes their horizontal forces opposite. Therefore equilibrium is equivalent to f(a)=F_1x=0:

$$
f(a)=\sum_{o\text{ active}}
\frac{k_o(a-o_x)}{((a-o_x)^2+(1-o_y)^2)^{3/2}+\epsilon}
-\frac{k_r(3-2a)}{(3-2a)^3+\epsilon}.
$$

`equilibrium_case()` starts with the rational bracket [0.770124,0.770126] and performs 20 sign-certified bisections. The final bracket is approximately

$$
0.7701250686740875\leq a^*\leq0.7701250686759949.
$$

The certificate also stores both endpoints as exact rational strings; displayed decimal coordinates are approximations. Rational interval evaluation gives f(left) > 6.95168 × 10^-12 and f(right) < -1.37285 × 10^-12. No cutoff crosses this bracket, so continuity and the intermediate value theorem establish an exact root. The derivative is negative throughout the bracket by the positive relative-horizontal Hessian eigenvalue below, so the scalar root is unique there.

The reported numerical representative is

- x_1 ≈ (0.7701250686750412,1),
- x_2 ≈ (2.229874931324959,1).

Its residual collective force is approximately 3.95 × 10^-12. The existence proof uses the sign bracket, not that small residual. The collective error between the bracket midpoint and the exact equilibrium is at most the bracket width, less than 1.908 × 10^-12 units, apart from floating-point representation. Initial-state energy bounds use the exact binary floating-point initial coordinates enclosed against the rational equilibrium bracket.

Both robots interact: their equilibrium separation is approximately 1.45974986265 < 4. Each has 15 active wall samples. Reflection preserves the entire sampled wall set, including the active set.

## 3. Collective positive-definite Hessian

For z=x-o and r=||z||, one active radial term contributes

$$
B(z)=k\left[\frac{3rzz^T}{(r^3+\epsilon)^2}
-\frac{I}{r^3+\epsilon}\right].
$$

Let A=diag(A_x,A_y) be the summed wall Hessian at robot 1. Its mixed entry vanishes by y-reflection; robot 2 has the same A by x-reflection. For the horizontal robot pair, B=diag(B_r,B_t), where

$$
B_r=\frac{k_r(2r^3-\epsilon)}{(r^3+\epsilon)^2},\qquad
B_t=-\frac{k_r}{r^3+\epsilon},\qquad r=3-2a^*.
$$

The full Hessian, including off-diagonal pair coupling, is

$$
H=\begin{bmatrix}A+B&-B\\-B&A+B\end{bmatrix}.
$$

Common and relative horizontal/vertical displacements give its four exact eigenvalue formulas:

| Mode | Eigenvalue | Enclosure, rounded here for display |
| --- | --- | --- |
| Common horizontal | A_x | 1.79255207321 to 1.79255207333 |
| Common vertical | A_y | 2.13626549515 to 2.13626549520 |
| Relative horizontal | A_x+2B_r | 4.36445343710 to 4.36445343729 |
| Relative vertical | A_y+2B_t | 0.85031419305 to 0.85031419312 |

The JSON contains outward-rounded enclosures rather than this abbreviated table. All four lower endpoints are strictly positive. In particular, transverse pair curvature is negative and is retained; the wall curvature dominates it. The root derivative satisfies f'(a)=-(A_x+2B_r)<0. This verifies a collective minimum, rather than testing each robot's diagonal Hessian block independently.

## 4. Rigorous arithmetic and Hessian neighborhood bounds

`Interval` uses standard-library `Fraction` endpoints. Addition, subtraction, multiplication, division and squaring use exact rational arithmetic. Square roots are enclosed by integer square roots on a 10^24 grid: if q is rational and s=10^24, the computed integer n satisfies n^2 <= q s^2 < (n+1)^2. The interval [n/s,(n+1)/s] therefore encloses sqrt(q). Lower and upper input endpoints are treated separately. JSON interval endpoints are expanded with `nextafter` when converted to floating point. Gains, epsilon and geometry are interpreted as the documented exact decimal model parameters; the ODE solver evaluates that model in floating point.

Take the collective Euclidean position radius rho=0.001 in R^4. A single robot's displacement is at most rho; relative robot displacement is at most sqrt(2) rho. Interval source-distance bounds give:

- Wall cutoff margin at equilibrium > 0.0018840679205 > rho.
- Robot-pair cutoff margin throughout the ball > 2.53883592378.
- Body-wall clearance throughout the ball > 0.56912506867.
- Body-pair surface clearance throughout the ball > 1.05833564908.

Thus the entire closed ball is collision-free, has unchanged wall active sets and keeps its single robot pair active. All relevant source distances are positive, so it lies strictly in one smooth interaction region.

For one radial Hessian, write B=k[C(r)zz^T-D(r)I], with C=3r/(r^3+epsilon)^2 and D=1/(r^3+epsilon). Since r^3+epsilon >= r^3,

$$
|C'|\leq21/r^6,\qquad 2|C|r\leq6/r^4,\qquad |D'|\leq3/r^4,
\qquad \|DB(z)\|\leq30k/r^4.
$$

For a wall term, the position extraction map has norm 1. For a robot pair, the difference map J=[I,-I] has norm sqrt(2), and its collective Hessian is J^T B J. Thus the pair's collective Hessian Lipschitz bound includes ||J||^3=2sqrt(2); omitting this factor would be incorrect.

Let d_o^- be the interval lower distance from (a*,1) to an active wall sample, and r^- the lower pair distance. Sum both robots' wall terms and the pair once:

$$
L\leq2\sum_{o\text{ active}}\frac{30k_o}{(d_o^- -\rho)^4}
+\frac{2\sqrt{2}\,30k_r}{(r^- -\sqrt{2}\rho)^4}<276.
$$

The code rounds this bound upward to L=276. Weyl's eigenvalue perturbation bound then yields throughout the entire ball

$$
0.574 I\preceq\nabla^2U(X)\preceq4.641 I.
$$

Indeed the smallest equilibrium eigenvalue exceeds 0.85031419305, so subtracting L rho=0.276 leaves more than 0.574. The largest is below 4.36445343729; adding 0.276 leaves less than 4.641. This bound holds between integration samples as well as at them.

## 5. Energy dissipation, invariant sublevel and local convergence

For E=U+(m/2)||v||^2, the chain rule and the actual continuous dynamics give

$$
\dot E=\nabla U\cdot v+mv\cdot\dot v=-b\|v\|^2\leq0.
$$

This proves mechanical energy dissipation for interacting robots, with the pair counted once in U. The plot is numerical supporting evidence, not the proof.

Let W=U(X)-U(X*)+(m/2)||v||^2. Taylor's integral formula and the certified Hessian bounds imply

$$
\frac{0.574}{2}\|X-X^*\|^2+\frac m2\|v\|^2
\leq W\leq
\frac{4.641}{2}\|X-X^*\|^2+\frac m2\|v\|^2.
$$

Choose c=1.435 × 10^-7 and

$$
K_c=\{(X,v):\|X-X^*\|\leq0.001,\ W(X,v)\leq c\}.
$$

At the position boundary, W >= (0.574/2)(0.001)^2 = 2.87 × 10^-7 > c. This compact sublevel lies strictly inside the smooth collision-free ball. Since W is nonincreasing, a first-exit argument proves positive invariance; compactness and local smoothness extend solutions for all future time. Strong convexity implies (X-X*) dot grad U(X) >= 0.574 ||X-X*||^2, so there is no other critical point in the ball. The largest invariant subset with W_dot=0 is therefore the single equilibrium (X*,0). LaSalle's invariance principle gives convergence to that equilibrium, and the positive-definite energy bounds give local asymptotic stability. These establish both properties for the collective interacting system.

The initial position offsets are [[0.0001,-0.00008],[-0.00007,0.00009]], and velocities are [[0.00002,-0.00001],[-0.00001,0.00002]]. They have nonzero projections onto all four common/relative coordinate modes and do not preserve equilibrium symmetry. Exact rational interval evaluation of the represented initial coordinates bounds W(0) below 6.873 × 10^-8 < c. This proves initial membership in K_c without relying on numerical potential quadrature.

## 6. Numerical experiment and verification

The unchanged `baseline_forces` is integrated by DOP853 for 20 seconds, rtol=10^-11, atol=10^-13, maximum internal step 0.02 s, output interval 0.01 s (2,001 samples). `potential_energy` reuses the exact regularized-potential definition evaluated by quadrature. No clipped/discrete simulation update is called.

| Quantity | Recorded value |
| --- | ---: |
| Initial relative mechanical energy | 3.86319723 × 10^-8 |
| Final relative energy | -8.88 × 10^-16, cancellation roundoff |
| Largest recorded energy increment | 3.55 × 10^-15, roundoff scale |
| Maximum trapezoidal energy-balance residual | 1.21044 × 10^-12 |
| Initial collective position error | 1.71464282 × 10^-4 units |
| Maximum collective position error | 1.72465125 × 10^-4 units |
| Final collective position error | 7.65362 × 10^-11 units |
| Final maximum robot speed | 7.32117 × 10^-11 units/s |
| Maximum robot speed | 1.23112112 × 10^-4 units/s |
| Maximum robot acceleration | 4.34654817 × 10^-4 units/s² |
| Minimum sampled robot-center separation | 1.45957929085 units |
| Minimum sampled body-wall clearance | 0.57009630064 units |

All sampled states are finite and collision-free, all interaction sets remain unchanged and the full trajectory stays inside the certified ball. The theorem certifies the exact ODE; DOP853 is an ordinary high-accuracy floating-point calculation, not an interval-enclosed numerical solution. The sampled checks do not themselves prove between-sample behavior. The tiny negative final W and positive final increments are preserved as numerical roundoff, not clipped to zero. Position error need not decrease monotonically.

The four 300-DPI figures show relative energy, individual robot speeds, collective equilibrium error and robot recovery trajectories. The trajectory figure and video include a full workspace view and separate local displacement panels because the certified perturbations are too small to see at workspace scale. The displacement panels retain actual coordinate units and are labeled explicitly. The MP4 is H.264/yuv420p, 960 × 960, 20 fps, 401 frames, duration 20.05 s, including both endpoints of the 20-second simulation. Thin colored trails use linewidth 1.1 and alpha 0.55. FFprobe checks its stream; FFmpeg decodes the full video without errors.

Reproduce into fresh destinations from the repository root:

```sh
python Code/multi_robot_theoretical_analysis.py --output Results/theoretical_validation_multi_robot_repeat --video-output Videos/theoretical_validation_multi_robot_repeat
python -m unittest discover -s Code -p 'test_*.py'
```

Runtime dependencies are unchanged. FFmpeg, FFprobe and the libx264 encoder are required for the video. Existing nonempty output folders are refused. The script, new regression tests, parameter record, certificate and CSV provide reproducibility without importing local research notes.

## 7. Limits and sources

This is a local interacting baseline result for the specified 3 × 2 workspace, not a theorem for arbitrary teams, the original five-robot final state, cutoff switching, the clipped Euler simulator, Robotarium unicycle/barrier dynamics or discovery target changes. It provides no global optimum, global collision-avoidance or complete-coverage guarantee.

The force and damped mass model follow the project's Step 8D selection and Step 15 conservative-potential derivation, motivated by Howard, Matarić and Sukhatme (2002). The gradient/LaSalle framework is consistent with the theoretical guidance from Schwager, Rus and Slotine (2011), but their controller and theorems are not transferred verbatim. Discovery ideas from Liu and Lyons (2015) do not enter this fixed conservative example. The symmetry reduction, exact rational root enclosure, collective Hessian and explicit invariant bounds above are project-specific calculations. Existing research notes and report files remain unchanged.
