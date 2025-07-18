# arXiv:1808.01293

**Paper ID:** c3cec39d725d9ec4636b1280c1e53913

**PDF Path:** apl/Superconductors/arXiv:1808.01293.pdf

**Processing Status:** complete

**Captions Added:** 0

**Generated:** 2025-06-24T13:44:27.233243

---

# Vortex dynamics and losses due to pinning: Dissipation from trapped magnetic flux in resonant superconducting radio-frequency cavities

Danilo B. Liarte,<sup>1</sup> Daniel Hall,<sup>2</sup> Peter N. Koufalis,<sup>2</sup> Akira

Miyazaki,3, 4 Alen Senanian,<sup>1</sup> Matthias Liepe,<sup>2</sup> and James P. Sethna<sup>1</sup>

<sup>1</sup>Laboratory of Atomic and Solid State Physics, Cornell University, Ithaca, NY, USA

<sup>2</sup>Cornell Laboratory for Accelerator-Based Sciences and Education, Cornell University, Ithaca, NY, USA

<sup>3</sup>CERN, Geneva, Switzerland

<sup>4</sup>University of Manchester, Manchester, UK

(Dated: September 5, 2024)

We use a model of vortex dynamics and collective weak pinning theory to study the residual dissipation due to trapped magnetic flux in a dirty superconductor. Using simple estimates, approximate analytical calculations, and numerical simulations, we make predictions and comparisons with experiments performed in CERN and Cornell on resonant superconducting radio-frequency NbCu, doped-Nb and Nb3Sn cavities. We invoke hysteretic losses originating in a rugged pinning potential landscape to explain the linear behavior of the sensitivity of the residual resistance to trapped magnetic flux as a function of the amplitude of the radio-frequency field. Our calculations also predict and describe the crossover from hysteretic-dominated to viscous-dominated regimes of dissipation. We propose simple formulas describing power losses and crossover behavior, which can be used to guide the tuning of material parameters to optimize cavity performance.

#### I. INTRODUCTION

Vortex matter is the "smoking gun" of type II superconductors [1–4], typically appearing in the form of a lattice of quantized magnetic flux lines in equilibrium superconductor states at intermediate ranges of applied magnetic fields and low temperatures. Compared to clean Meissner states, the vortex state is not a good superconductor state; transverse transport currents (j ⊥ H, with H representing the vortex magnetic field) acting on the vortex flux line via Lorentz forces can dissipate power. To restore dissipation-free current flows and control the dissipation of high-temperature superconductors, it has become common practice to employ impurity doping to pin the vortices and restrain their motion. A dirty superconductor is often a good superconductor. Incidentally, highpower resonant superconducting radio-frequency (SRF) cavities for particle accelerators operate in the metastable Meissner state [5–7], i.e. at magnetic fields above the lower critical field and below the superheating field [8], which might mislead one to conclude that vortex motion have negligible, if any impact on cavity power dissipation. Here pinning by impurities plays a double role. On the one hand, defects can trap vortex flux lines (originating in the earth magnetic fields, thermo-electric currents, etc) that should have been expelled from the superconductor during the cavity cool-down. On the other hand, pinning can restrain the motion of the trapped vortices and restore the desired dissipation-free current flow property of the Meissner state. In typical SRF applications, oscillating magnetic fields parallel to the superconductor interface can move isolated flux lines near the surface, and produce non-negligible contributions to the cavity surface resistance.

In this paper, we use a model of vortex dynamics and collective weak pinning theory [4] to study the dissipation of an isolated superconducting vortex line in a Gaussian random disordered potential (due to weak pinning on defects), subject to a time-dependent forcing near the surface (due to the alternating magnetic fields Brf parallel to the inner surface of the SRF cavity). We will compare our results to three experimental measurements, for doped Nb, Nb3Sn, and NbCu cavities.<sup>1</sup>

Superconductors subject to oscillating fields dissipate power on their surface due to thermal excitation of quasiparticles, even if there is no vortex matter. We write the surface resistance of a superconductor as [5]

$$R\_S = \frac{2}{H\_{rf}}P,\tag{1}$$

where P is the power per unit area dissipated in the superconductor wall and Hrf is the amplitude of the rf applied magnetic field<sup>2</sup> . The surface resistance decomposes into temperature-dependent and temperatureindependent parts, R<sup>S</sup> = RBCS+R0, with the former and latter named BCS and residual resistance, respectively. The BCS part is usually described by BCS theory<sup>3</sup> .

The residual part is caused by several factors. Here we focus our attention to the case where R<sup>0</sup> is caused primarily by trapped magnetic flux. Indeed, recent measurements in current cavity designs show that the temperature independent residual resistance R<sup>0</sup> can be a large

<sup>1</sup> For those not in the accelerator community, NbCu cavities are niobium films a few microns thick sputtered onto copper substrates – not a compound.

<sup>2</sup> Henceforth, we restrict our attention to ac rf fields.

<sup>3</sup> The decomposition into temperature-dependent and independent parts is phenomenological, and the linear response of BCS theory (Mattis-Bardeen) does not necessarily describe some experimental results (see [9]).

fraction of the total dissipation (from about 20% for Nb to 50% for Nb3Sn) at operating temperatures [10, 11] and that it is roughly linear in the density of trapped flux [11]. The fact that R<sup>0</sup> is negligible at small trapped flux strongly suggests that it is due to vortices; the linearity suggests that the vortices are not interacting strongly with one another – motivating our study of the dissipation due to a single flux line. Measurements of trappedflux residual resistance are routinely employed by the SRF community to quantify power losses and cavity quality factors. Typical experiments show a characteristic bell shape dependence of R<sup>0</sup> as a function of the electronic mean free path [12, 13], though early Nb films display a still intriguing "U"-shaped dependence [14].

Previous theoretical calculations of dissipation [13, 15] have ignored the effects of collective weak pinning on vortex motion, and have derived a value for the residual resistance R<sup>0</sup> that is independent of the amplitude Brf of the cavity rf field. The recent cavities show a residual resistance that is roughly linear in the rf field Brf (and hence a dissipation that is cubic in the rf field) [9, 16–18]. Also, our calculations show that the total dissipation, ignoring pinning, predicts not only a constant R0, but one that is much higher than the measured dissipation at low fields<sup>4</sup> . Since the energy dissipated by a moving vortex<sup>5</sup> is given by the Lorentz force times the distance moved at the surface, some kind of pinning must be included to restrict the amplitude of motion. This motivates our consideration of collective weak pinning. We shall find that collective weak pinning does indeed predict a linear dependence of R<sup>0</sup> on Brf. Our estimates suggest that weak pinning due to point impurities (dopants) is a factor of 6-20 too small to explain the low losses observed, and will discuss the possible role of extended defects (dislocations, grain boundaries) and other possible reasons for the remaining discrepancy.

It is surprising that the dynamical behavior of an individual vortex is less well-known and understood than that of many interacting vortices<sup>6</sup> [19]. To study dynamics, we consider an idealized model where the vortex line is an elastic one-dimensional string whose conformation is fully described by a displacement field u = u(z) from a reference configuration, where z is the Cartesian coordinate associated with the distance from the superconductor surface, and we assume u(z) = 0 ∀z in the reference configuration (see Fig 1a). The displacement

field satisfies the equation of motion,

$$f\_V + f\_E + f\_L + f\_P = 0,\tag{2}$$

where f<sup>I</sup> denotes a force per length, and the subscripts V, E, L, and P are associated with viscous, elastic, Lorentz, and pinning forces, respectively<sup>7</sup> . Gurevich and Ciovati studied the ac dynamics of individual vortex lines strongly, irreversibly pinned at fixed distances from the interface, and made contact with thermal measurements of hot spots in Nb cavities [15]. They assume f<sup>P</sup> = 0, and implement strong pinning by fixing one end of the vortex line so that u(`<sup>P</sup> ) = 0 for a pinning center at z = `<sup>P</sup> . More recently, Checchin et al. extended the Gittleman-Rosenblum model [20] to study weakly, but also irreversibly pinned vortices using the harmonic approximation for the pinning potential and neglecting the vortex line tension f<sup>E</sup> [13]. Working with cuprates (YBCO), Auslaender et al. used collective weak pinning theory to study low-frequency dynamic properties of individual vortex lines that were imaged and manipulated by magnetic force microscopy [19].

![](_page_1_Figure_10.jpeg)

FIG. 1. (a) Illustration of an elastic vortex line subject to a Lorentz force and random pinning forces. (b) Sketch of our theoretical predictions for the sensitivity of the residual resistance to trapped flux as a function of the amplitude of the rf magnetic field. (c) Dimensionless crossover field as a function of the depinning current and inverse frequency.

Figure 1a depicts the collective weak pinning scenario in which we are interested. The red and blue line represents the vortex, with the inner red and outer blue tubes corresponding to the vortex core and the region of non-zero magnetic inductance, respectively. Small grey

<sup>4</sup> The previous theories fix the vortex position at a certain depth in the material (corresponding to a single, inescapable pinning point), which can be used to reduce the dissipation, but cannot introduce a dependence of the dissipation on the strength of the rf field.

<sup>5</sup> This ignores the contribution of quasiparticle excitations inside the vortex core, which contribute a small constant term to R0.

<sup>6</sup> Although individual vortex lines are easier to study, there are just a few situations, such as trapped flux dissipation in SRF cavities, in which they play important roles.

<sup>7</sup> In this paper, we neglect inertial and Magnus forces, which have sub-dominant contributions.

spheres represent point-like impurities. The arrows near H and f<sup>L</sup> define the directions of the rf magnetic field and the Lorentz force, respectively. We also show the depth coordinate z and the displacement field u(z), from a reference configuration (dashed line).

The near-depinning behavior of d-dimensional manifolds moving in d 0 -dimensional disordered environments is a long-standing problem in the field of non-equilibrium statistical mechanics that is connected to diverse physical situations, from crackling noise [21] to raindrops on windshields to superconducting vortices and plasticity [22, 23]. In typical vortex pinning models, pinning forces originate in the overlap of the normal conducting regions associated with the vortex core and the impurity defect. Pinning forces associated with atomic impurities are very weak. Collectively, they add up randomly, so that the average force over a length L vanishes. Only fluctuations in either force or impurity density can pin a vortex line. If the external Lorentz force is small, the vortex line can trade elastic energy and find an optimal stationary configuration in the disordered potential landscape. Right above the depinning force, the vortex line moves; velocity and velocity autocorrelations display universal power laws and scaling behavior associated with emergent scale invariance. As the Lorentz force increases further away from depinning, the dynamical behavior crosses over from quenched to dynamic disorder, reminiscent of the quenched to thermal KPZ crossover [24, 25], and the vortex line starts moving through unexplored regions of the potential landscape.

Thus, for collective weak pinning disorder, the vortex line will not move macroscopically until an external force per unit length becomes greater than the depinning threshold fp. The vortex line depinning transition is thought to be continuous — the force per unit length resisting the motion of a slowly moving vortex will approach f<sup>p</sup> as the velocity goes to zero (unlike, say, the textbook behavior of static vs. sliding friction). Here we shall simulate this depinning explicitly, and also provide a mean-field model, incorporating the depinning threshold f<sup>p</sup> but ignoring the critical fluctuations, avalanches, and scaling characteristic of continuous dynamical phase transitions.

Figures 1b and c summarize our main results. In b, we show a sketch of the behavior of the sensitivity of the residual resistance to trapped flux as a function of the amplitude of the rf field. We ignore the regime of very small applied magnetic field also known as the Campbell regime [26], in which the vortex displacements are much smaller than the characteristic pinning length, the vortex line remains trapped, and the low-dissipation Campbell response probes the pinning wells [27]. The sensitivity (black curve) crosses over at Brf = B<sup>X</sup> (dashed-green line) from a linear behavior (red line, with P ∼ Brf 3 ) at low fields to a plateau (blue line, with P ∼ Brf 2 ) at high fields. Our analysis describes the hysteretic losses dominating the linear behavior that is observed in the experiments, and the crossover to a viscous-dominated

regime. In c, we show our calculations for the crossover field B<sup>X</sup> (in units of the thermodynamic critical field Bc) as a function of the depinning current j<sup>d</sup> (in units of the depairing current [1] jo) and the inverse frequency fX/f, where f<sup>X</sup> is a function of superconductor parameters (see Eq. (28)). We find that B<sup>X</sup> ∼ j<sup>d</sup> f −1/2 . The blue, green and red lines correspond to the rescaled frequencies of the Nb3Sn, doped-Nb and NbCu cavities, respectively.

The rest of the paper is organized as follows. Section II discusses the vortex equations of motion, and our solutions for mean-field and local-potential models based on collective weak pinning theory. In Section III, we apply our theoretical analysis to new experimental results for CERN 100MHz NbCu and Cornell 1.3GHz doped-Nb and Nb3Sn cavities, and discuss possible mechanisms to justify the high depinning fields that are necessary to explain the experiments, and the remaining discrepancy between theory and measurements. We summarize our results and make some final remarks in Section IV. In Appendix A, we present some sanity checks that corroborate the results presented in Sections II and III. In Appendix B, we derive the correction factor that we used in Section III to make contact between our calculations and the experimental measurements.

## II. VORTEX MOTION AND DISSIPATION

# A. Equations of motion

We consider the dynamics of one vortex line in a superconductor that occupies the half-infinite space (z > 0). In its reference configuration, the vortex is a straight line normal to the the superconductor surface (i.e. the z = 0 plane). The vortex configuration at time t is completely determined by the displacement field u = u(z;t), which in this case is a scalar function of z. Let us write down explicit expressions for some of the terms appearing in Eq. (2):

$$f\_V = \eta \frac{du}{dt}, \quad f\_E = \epsilon\_\ell \frac{d^2 u}{dz^2},$$

$$f\_L = \frac{\phi\_0}{\lambda} \frac{H\_{rf}}{} \, e^{-z/\lambda} \sin(2\pi f \, t), \tag{3}$$

where η is the viscosity, ` is the vortex line tension, Hrf and ω are the amplitude and frequency of the magnetic field, λ is the superconductor penetration depth, and f is the rf frequency. The line tension can be written as [4, 28]

$$
\epsilon\_{\ell} = \epsilon\_0 \, c(\kappa), \tag{4}
$$

with

$$\begin{aligned} \epsilon\_0 &= \phi\_0^2 / (4\pi\mu\_0\lambda^2), & (5) \\ c\left(\kappa\right) &\approx \ln\kappa + 0.5 + \exp(-0.4 - 0.8\ln\kappa - 0.1(\ln\kappa)^2), & \end{aligned}$$

and φ<sup>0</sup> and µ<sup>0</sup> denoting the fluxoid quantum and the permeability of free space, respectively. The viscosity is given by the Bardeen-Stephen formula [29]: η = φ0 2 /(2πξ2ρn), where ξ is the superconductor coherence length, and ρ<sup>n</sup> is the resistivity of the normal phase. Defining dimensionless quantities ˜u = u/λ, ˜z = z/λ and t˜= f t, we can combine Eqs. (2) and (3) to write

$$\frac{d\tilde{u}}{d\tilde{t}} = C\left(\frac{B\_{rf}}{\sqrt{2}B\_c}e^{-\tilde{z}}\sin(2\pi\tilde{t}) + \frac{c\left(\kappa\right)}{2\kappa}\frac{d^2\tilde{u}}{d\tilde{z}^2} + \frac{\xi f\_P}{2\epsilon\_0}\right),\tag{6}$$

where C = ρn/(µ0λ ξfκ<sup>2</sup> ), Brf denotes the amplitude of the rf magnetic inductance, and B<sup>c</sup> is the thermodynamic critical field.

In collective weak pinning theory [4, 30], the accumulated pinning force over a length L is given by the squareroot fluctuation form,

$$F\_P(L) \approx \sqrt{{F\_i}^2 \, n\_{\mathcal{D}} \, \xi^{2-\mathcal{D}} L},\tag{7}$$

where F<sup>i</sup> denotes a typical individual pinning force, D is the spatial dimension of the defects (0, 1 and 2 for point-like, line and surface defects, respectively), and n0, n<sup>1</sup> and n<sup>2</sup> are the number of defects per unit volume, area, and length, respectively<sup>8</sup> . Note that standard collective weak pinning theory assumes point-like defects (D = 0 in our notation). For higher-dimensional defects (D > 0), we consider a scenario where the line or surface defects are randomly placed and randomly oriented, as illustrated in Fig. 2. The normal-conducting core of the vortex line is attracted to the defect region and can exhibit pinning and depinning behavior similar to that of point-like impurities. Using the superconductor condensation energy, we estimate F<sup>i</sup> for point-like impurities and extended defects such as dislocations and grain boundaries (see Appendix A). Note that pinning by extended defects can be substantially stronger than pinning by point-like defects. At lengths larger than the depinning length Lc, defined as the length in which the pinning energy balances the elastic energy, a vortex can deform and trade elastic energy to find a favorable configuration in the disordered potential landscape (cutting off the square-root dependence of the pinning force). In the standard theory, the vortex line breaks up into a chain of segments of length Lc, each individually competing with the Lorentz force. We propose and discuss approximate formulas for the collective pinning force in Sections II B and II C.

The power dissipated by a single oscillating vortex is given by

$$\begin{split} P\_1 &= f \int\_0^{1/f} dt \int\_0^{\infty} dz \, f\_L \cdot \frac{du}{dt} \\ &= \frac{f \lambda \, \phi\_0 B\_{rf}}{\mu\_0} \int\_0^1 d\tilde{t} \sin 2\pi \tilde{t} \int\_0^{\infty} d\tilde{z} \, e^{-\tilde{z}} \frac{d\tilde{u}}{d\tilde{t}}. \end{split} \tag{8}$$

![](_page_3_Figure_10.jpeg)

FIG. 2. Illustration of the collective weak pinning scenario for general D-dimensional defects. The gray line at the top represents the superconductor-vacuum interface. The blue lines on the left, center and right represent vortex lines. Small circles on the right correspond to point-like impurities with D = 0 (as in Fig. 1a). Red lines at the center correspond to line defects with D = 1, such as dislocation lines. Grey polygons on the left correspond to surface defects with D = 2, such as grain boundaries; the red dashed lines illustrate the regions where the vortex is pinned by the defects.

The net flux trapped in an area s breaks up into N vortices of fluxoid quanta φ0, Btraps = Nφ0, so that, using Eq. (1) to calculate the residual resistance, we find:

$$\begin{split} \frac{R\_0}{B\_{\text{trap}}} &= \frac{2\mu\_0 {}^2 P\_1}{\phi\_0 B\_{rf}} \\ &= \frac{2f\lambda\,\mu\_0}{B\_{rf}} \int\_0^1 d\tilde{t} \sin 2\pi \tilde{t} \int\_0^\infty d\tilde{z} \, e^{-\tilde{z}} \frac{d\tilde{u}}{d\tilde{t}}. \end{split} \tag{9}$$

# B. Mean-field model

In this section, we consider a mean-field version of the pinning force using the collective weak pinning theory. We assume that the absolute value of the pinning force is the depinning force, i.e. the Lorentz force due to a transverse uniform current accumulated over the depinning length Lc, and that its sign is chosen so that it opposes the sum of the Lorentz and the elastic forces,

$$f\_P = -\text{sgn}(f\_L + f\_E)\,\phi\_0 \, j\_d,\tag{10}$$

where sgn denotes the sign function and j<sup>d</sup> is the depinning current. Equation (10) is a key assumption on our mean-field model, and partly follows from the force balance equation (2). If the frequency is small, we can ignore the viscous dissipation force in (2), which leads to a pinning force that opposes the sum of the elastic and Lorentz forces, thus justifying the sign function. The constant appearing in Eq. (10) also follows from the force balance equation (2), and collective weak pinning theory. If the motion is quasi-static, at each time the vortex line accommodates itself in the rugged potential landscape to minimize its free energy, deforming over lengths of order the depinning length Lc. As previously mentioned,

<sup>8</sup> Note that n<sup>D</sup> ξ <sup>2</sup>−D corresponds to the number of individual forces per unit length.

we can break up the vortex line into smaller segments of size L<sup>c</sup> and assume that the pinning force balances the Lorentz force for each segment. The segments will not move away from their low-energy configuration until the Lorentz force overcomes the pinning force; so we assume the pinning force is given by the Lorentz force (φ0j) at the "critical" depinning current j = jd, which is a convenient and experimentally measurable quantity)<sup>9</sup> . Note that f<sup>P</sup> is a piecewise function, with each sub-domain being determined by the sign of f<sup>L</sup> + fE, instead of the expected depinning length Lc. This simplifying assumption allows us to gain insight from approximate analytical solutions, and is motivated by the fact that we consider ranges of large magnetic fields, far above depinning, so that we expect the realistic model to display fairly smooth solutions. We show in Section II C that our numerical simulations of the local potential model corroborate this assumption.

First we consider the low frequency behavior, where the vortex motion is slow, and we can neglect the viscous dissipation term.<sup>10</sup> This approximation is valid for the range of parameters in which η vmax/|f<sup>P</sup> | 1, where vmax is the maximum velocity of the vortex displacement field at the boundary. We revisit this condition later on this section, when we self-consistently define the crossover from cubic to quadratic dissipation. We also make a point-force approximation, by replacing the exponential decay of the Lorentz force by a delta function: exp(−z˜) ≈ δ(˜z). This approximation is adequate when the amplitude of motion in the z direction (az) is sufficiently large compared to the penetration depth λ. Note that the existence of a delta function at the boundary fixes the slope of the displacement at z = 0 for each time, violating, in general, the realistic constraint of zero normal current at the superconductor surface (du/d ˜ z˜ = 0 at ˜z = 0).<sup>11</sup> Now equation (6) can be written as

$$\frac{d^2\bar{u}}{d\bar{z}^2} = \pm \alpha - \beta \, \sin(2\,\pi\bar{t})\delta(\bar{z}),\tag{11}$$

where the ± depends on the value of the sgn function in Eq. 10, and where α and β are given by

$$\alpha = \frac{\lambda |f\_P|}{\epsilon\_\ell}, \quad \beta = \frac{\sqrt{2}}{c(\kappa)} \frac{B\_{rf}}{B\_c}. \tag{12}$$

The solution of Eq. (11) is a parabola:

$$
\tilde{u}(\tilde{z}) = a + b\,\tilde{z} \pm \frac{\alpha}{2}\tilde{z}^2. \tag{13}
$$

where a and b are constants determined by the boundary conditions. Integration of Eq. (11) over a small interval near the surface leads to

$$\left. \frac{d\tilde{u}}{d\tilde{z}} \right|\_{\tilde{z}=0^{+}} = b = -\beta \sin(2\pi \tilde{t}),\tag{14}$$

and,

$$
\tilde{u}(\tilde{z}) = a - \beta \sin(2\pi \tilde{t}) \,\tilde{z} \pm \frac{\alpha}{2} \tilde{z}^2. \tag{15}
$$

Equation (15) is only valid at sufficiently small z; the vortex line remains pinned in the superconductor deep interior. We find a by imposing that the vortex moving section continuously and smoothly merges with the pinned section at a distance ˜z ∗ that we determine. Let ˜u<sup>&</sup>lt; and ˜u<sup>&</sup>gt; be the solutions near and away from the superconductor surface, respectively. The complete solution is given by

$$
\tilde{u} = \begin{cases}
\tilde{u}\_{<}, & \text{for } \tilde{z} < \tilde{z}^\*, \\
\tilde{u}\_{>}, & \text{otherwise},
\end{cases} \tag{16}
$$

where a and ˜z <sup>∗</sup> are determined by the equations:

$$
\tilde{u}\_{<}(\tilde{z}^\*) = \tilde{u}\_{>}(\tilde{z}^\*), \quad \frac{d\tilde{u}\_{<}}{d\tilde{z}}(\tilde{z}^\*) = \frac{d\tilde{u}\_{>}}{d\tilde{z}}(\tilde{z}^\*). \tag{17}
$$

Let us study the solutions for t˜ ∈ [0, 1/4], assuming u˜(˜z;t˜ = 0) = 0. We use the subscript 0 to denote solutions in this interval. Using Eqs. (15) and (17), we find,

$$\tilde{u}\_0(\tilde{z};\tilde{t}) = \begin{cases} \frac{\alpha}{2} \left( \tilde{z} - \frac{\beta}{\alpha} \sin(2\pi\tilde{t}) \right)^2, & \text{for } \tilde{z} < \frac{\beta}{\alpha} \sin(2\pi\tilde{t}),\\ 0, & \text{otherwise.} \end{cases} \tag{18}$$

The blue line in Fig. 3 corresponds to ˜u<sup>0</sup> as a function of ˜z for t˜ = 1/4 and α = β = 1. As t˜ increases from 1/4, the elastic and pinning forces exchange signs near the surface, the tip of vortex line reverses motion and starts "unzipping" from the the blue curve. The complete solution has ˜u<sup>&</sup>gt; = ˜u0(˜z; 1/4) and ˜u<sup>&</sup>lt; given by (15) with the negative sign (red curves in Fig 3), and with a and z˜ ∗ satisfying Eq. (17). For t˜∈ [1/4, 3/4], we find,

$$\tilde{u}(\bar{z};\bar{t}) = \begin{cases} a(\bar{t}) - \beta \sin(2\pi\bar{t})\bar{z} - \frac{\alpha}{2}\,\bar{z}^2, & \text{for } \bar{z} < \bar{z}^\*(\bar{t}),\\ \frac{\alpha}{2}\left(\bar{z} - \frac{\beta}{\alpha}\right)^2, & \text{otherwise.} \end{cases} \tag{19}$$

<sup>9</sup> The idea of a "critical" force also appears in a critical state model [3, 31], such as the Bean model [32], but in a different context. However, unlike the Bean model, our model ignores the interactions between vortices and incorporates the structure of the vortex line. The Bean model involves many interacting vortices pinned on dirt; our model is a single vortex pinned (collectively) on many dirt particles.

<sup>10</sup> One must note that the low-frequency limit approaches the depinning transition, where disorder-induced fluctuations become important and the mean-field model is not quantitatively correct. It is, however, analytically solvable and a useful illustration and starting point for understanding high-frequencies (Section III) and interpreting the local potential simulations incorporating disorder (Section II C).

<sup>11</sup> In Appendix A, we deform our analytical solution over a length λ near the boundary to satisfy the constraint at the surface. For large enough fields (in particular, for most of the range of fields considered in Fig. 5),the change in vortex length is very small compared to the amplitude of motion in the y direction, suggesting that the error resulting from this approximation is small.

with

$$a(\hat{t}) = \frac{\beta^2}{8\alpha} \left( 1 + \cos 4\pi \hat{t} + 4 \sin 2\pi \hat{t} \right), \qquad (20)$$

$$
\tilde{z}^\*(\tilde{t}) = \frac{\beta}{2\alpha} (1 - \sin 2\pi \tilde{t}).\tag{21}
$$

Note that the amplitude of motion at the surface is proportional to ˜u(0, 1/4) ∝ β <sup>2</sup> ∝ Brf 2 , so that the dissipation energy is proportional to f<sup>L</sup> × Brf <sup>2</sup> ∝ Brf 3 , in agreement with the experiments. This leads to the important conclusion that the cubic dissipation is intimately connected to the quadratic solutions for the vortex motion, which is an ultimate consequence of the existence of a pinning force α. One caveat: The cubic dissipation might become quadratic when the boundary condition in the deep interior of the superconductor is changed. For instance, a simple way of controlling the total dissipation consists in employing restrictive inescapable pinning potentials (such as the ones considered in references [13] and [15]) for the vortex line at a distance ˜z<sup>p</sup> so that u˜(˜zp) ≈ 0. Our simple calculations show that if ˜z<sup>p</sup> is sufficiently small (for a given field), the dissipation is proportional to Brf 2 ; the cubic behavior disappears. In Section III C we discuss how the combination of strong and collective weak pinning might help explain the discrepancy between theory and experiments.

Figure 3 shows solutions of ˜u as a function of ˜z, for α = β = 1, and t˜ = 1/4 (blue), 5/12 and 7/12 (dashed red), and 3/4 (solid red). The purple dots correspond to the points where the two parabolas merge. The subsequent solution in the interval [3/4, 5/4], is a reflection of the solutions in [1/4, 3/4], i.e. ˜u(t˜) = −u˜(t˜− 1/2), for t˜ ∈ [3/4, 5/4].

![](_page_5_Figure_5.jpeg)

FIG. 3. Mean-field solution of the vortex displacement field u˜ as a function of depth coordinate ˜z for t˜= 1/4 (blue), 5/12 and 7/12 (dashed red) and 3/4 (solid red).

We use Eq. (8) to write down the power dissipated by one vortex,

$$P\_1 = \frac{8\pi}{3\,\mu\_0^2} \frac{f\,\lambda^2}{j\_d\,c(\kappa)} B\_{\rm rf}^{\,3},\tag{22}$$

and Eq. (9) to calculate the sensitivity of the residual

resistance to trapped flux,

$$\frac{R\_0}{B\_{\text{trap}}} = AB\_{rf},\tag{23}$$

where

$$A = \frac{16\pi}{3\,\phi\_0} \frac{f\,\lambda^2}{c(\kappa)j\_d}.\tag{24}$$

The sensitivity linear increase with the rf field is qualitatively consistent with experimental measurements for 100MHz NbCu and 1.3GHz doped-Nb and Nb3Sn cavities (see Section III). For better quantitative agreement with the experimental results, we include a correction to account for the alignment of the vortices throughout the cavity surface and field depletion at the cavity poles (see Appendix B). Note that the measured residual resistance approaches a finite value as Brf → 0, whereas our pinning model predicts R<sup>0</sup> → 0 at the same limit (Eq. (23)). We ignore other sources of residual resistance that are not associated with vortex motion, and that can explain this offset. For example, a plausible model of static residual resistance considers the normal conducting resistance originating in the core of the vortex line [5]. To make a direct comparison with experiments, we subtract off the offset of the measured sensitivity (red circles) in Fig. 5. In these modern cavities, the linear term we attribute to vortex motion dominates R<sup>0</sup> under operating conditions; the offset is equal to 7.7, 4.6 and 0.03 nΩ/µT in Fig. 5, for doped Nb, Nb3Sn and NbCu, respectively.

The hysteretic losses that are responsible for the linear slope of the sensitivity become less important at high rf field amplitudes, so that we expect a crossover to a highfields regime where viscous dissipation is the dominant loss mechanism. To quantify this crossover, we use the solution given by Eq. (19) to self-consistently calculate η vmax, which we compare with the pinning force. Here vmax is the maximum of du/dt at z = 0 over one period of oscillation. We define the crossover field B<sup>X</sup> from the equation:

$$\left.\frac{\eta \, v\_{\text{max}}}{f\_P}\right|\_{B\_{rf}=B\_X} = 1,\tag{25}$$

yielding,

$$B\_X = \sqrt{\frac{2}{3\sqrt{3}\pi} \frac{\mu\_0 \rho\_n c \left(\kappa\right) j\_d^2}{\kappa^2 f}}. \tag{26}$$

Equation (26) can also be written in the dimensionless form

$$\frac{B\_X}{B\_c} = \sqrt{\frac{f\_X}{f}} \frac{j\_d}{j\_o},\tag{27}$$

where,

$$f\_X \equiv \frac{16}{81\sqrt{3}\pi} \frac{c\left(\kappa\right)}{\kappa^2} \frac{\rho\_n}{\mu\_0 \lambda^2},\tag{28}$$

and the depairing current is given by

$$j\_o = \frac{4}{3\sqrt{6}} \frac{B\_c}{\mu\_0 \lambda},\tag{29}$$

according to Ginzburg-Landau (GL) theory. We have already briefly discussed Figure 1c, showing the crossover field BX/B<sup>c</sup> as a function of the depinning current jd/j<sup>o</sup> and the inverse frequency fX/f, with blue, green and red lines corresponding to the Nb3Sn, doped-Nb and NbCu cavity rescaled frequencies, respectively. Table I show our calculated values for B<sup>X</sup> using the simulation parameters. Note that low-κ, low-frequency SRF cavities at high depinning currents have high BX.

The viscous dissipation term is important and cannot be neglected at either high frequency or high field amplitudes. Finding closed forms for the piecewise solutions of the full equation of motion is beyond the scope of this paper. We then opted for discretizing the vortex line using Python arrays, and using SciPy odeint package to numerically integrate Eq. (6). We give more details of these simulations in Section III.

#### C. Local-potential model

![](_page_6_Figure_6.jpeg)

FIG. 4. (a) Illustration of a disordered potential landscape in the collective weak pinning scenario. (b) Discretized vortex line (blue dots and segments), subject to local random forces (black arrows) originating in the disordered potential landscape shown in (a).

Here we consider a model where the vortex line is subject to local pinning forces originating in a Gaussian random potential of zero mean, and adequately scaled variance. In our numerical approach we first define a grid of spacing ˜a < a in the y-z plane, where a is the spacing of the z-coordinate of the vortex array (we choose a˜ ∼ ξ, which is the smallest length a superconductor can resolve). We then assign i.i.d. normal random variables to each point of the grid, and use a spline interpolant to implement the unscaled potential U˜ for arbitrary y and z. Figure 4a depicts a square grid (blue-dashed lines) and the corresponding interpolated potential. The individual force per length acting on a segment i of the discretized vortex is then given by f<sup>i</sup> = (U0/a) ∂U /∂y ˜ , where the constant U<sup>0</sup> is chosen to match the depinning force determined by collective weak pinning theory. The force accumulated over the depinning length L<sup>c</sup> can be written as

$$F\_P(L\_c) = \sqrt{\left\langle \left(\sum\_{i=i\_0}^{i\_0+L\_c/a} f\_i \, a\right)^2 \right\rangle},\tag{30}$$

where i<sup>0</sup> is an arbitrary initial site of the vortex array (see Fig. 4b), and h·i denotes an average over i0. We expect local individual forces be uncorrelated over distances <sup>&</sup>gt;<sup>∼</sup> 2 ˜a, so that, after some algebra,

$$F\_P(L\_c) \approx U\_0 \sqrt{a \, L\_c} \, \sigma\_f,\tag{31}$$

where <sup>σ</sup>f˜ is the variance of ˜f<sup>i</sup> <sup>≡</sup> <sup>f</sup>i/U0. Equations (31) and (7), result in

$$U\_0 = \frac{\phi\_0 \, j\_d}{\sigma\_{\tilde{f}}} \sqrt{\frac{L\_c}{a}}. \tag{32}$$

We use collective weak pinning theory to express L<sup>c</sup> as a function of jd. Let γ ≡ F<sup>i</sup> <sup>2</sup> n<sup>D</sup> ξ <sup>2</sup>−D in Eq. (7), so that f<sup>P</sup> = p γ/L is the pinning force per length L. The pinning energy per length is then given by ξf<sup>P</sup> = p γ ξ2/L. To find the depinning length, we minimize the total energy per length with respect to L for a small displacement (of order ξ) of the vortex line in the absence of the Lorentz force, i.e.,

$$\frac{d}{dL}\left[\frac{\epsilon\_{\ell}}{2}\left(\frac{\xi}{L}\right)^{2} - \sqrt{\frac{\gamma \xi^{2}}{L}}\right]\_{L=L\_{c}} = 0,\tag{33}$$

resulting in

$$L\_c^{\,3} = \frac{4\,\epsilon\_\ell^{\,2}\,\xi^2}{\gamma} = \frac{4\,\epsilon\_\ell^{\,2}\,\xi^{\mathcal{D}}}{n\_{\mathcal{D}}F\_i^{\,2}}.\tag{34}$$

Now we make f<sup>P</sup> equal the Lorentz force due to a transverse uniform current jd,

$$f\_P = \sqrt{\frac{\gamma}{L\_c}} = \phi\_0 \, j\_d,\tag{35}$$

to eliminate γ in (34), finding,

$$\left|L\_c\right|^2 = \frac{2\,\epsilon\_\ell\,\xi}{\phi\_0\,j\_d} = \frac{\sqrt{2}\,B\_c\,c\left(\kappa\right)/(\lambda\,\mu\_0)}{j\_d}\,\xi^2.\tag{36}$$

Equation (36) is usually written in the approximate form [1]: Lc/ξ ≈ p jo/jd, where j<sup>o</sup> is the depairing current calculated using GL theory (see Eq. (29)). Collective weak pinning is valid when L<sup>c</sup> ξ, or j<sup>d</sup> jo. We present our simulation results for doped Nb, Nb3Sn and NbCu along with the experimental results of Section III.

## III. EXPERIMENTS AND SIMULATIONS

In this section, we discuss our numerical simulations, and make contact with experimental measurements performed in CERN and Cornell. In Section III A we discuss the experimental setup for doped Nb, Nb3Sn and NbCu cavities. In Section III B we give additional details of the simulations, and present experimental, analytical and simulation results for the sensitivity to trapped flux of the residual resistance. In Section III C we discuss plausible mechanisms that might explain the discrepancy between theory and experiments.

## A. Experimental setup

### 1. Doped Nb

Niobium cavities impurity doped with nitrogen in a high-temperature furnace show a characteristic fielddependent decrease in the BCS surface resistance that is frequently referred to as "anti-Q-slope" [33]. In the last few years, significant effort has gone into the study of the science of impurity doped niobium (see for example [11, 34, 35]), and nitrogen doped 1.3 GHz SRF cavities have now found their first use in the LCLS-II accelerator [36]. The residual resistance of nitrogen doped niobium cavities due to trapped flux has been shown to strongly depend on the electronic mean free path of the niobium in the rf penetration layer, with a characteristic bell shape dependence of R<sup>0</sup> [12, 13]. Recent results indicate that the anti-Q slope is not unique to nitrogen doping, but can also be found in higher frequency (multi-GHz) SRF cavities without doping [35, 37], as well as in 1.3 GHz cavities with high concentrations of oxygen and carbon dissolved in the surface [38]. As part of our studies on the field dependence of the trapped flux residual resistance, we measured trapped flux losses in a 1.3 GHz cavity that had been heat treated at 160C for 48 hr in an Ar/CO<sup>2</sup> gas mixture (99.99999% purity Ar gas mixed with 10 ppm CO2) immediately following an 800C vacuum anneal and prior to rf performance testing. Secondary ion mass spectroscopy (SIMS) analysis of a witness sample revealed very high concentrations of C and O—especially within the first few 100 nm [38].

The performance of the impurity doped cavity and its sensitivity to trapped magnetic flux was measured in a standard SRF vertical test setup, with a uniform (±10%), ambient DC magnetic field applied along the direction of the cavity axis by a Helmholtz-coil during cool-down (refer to [12] for details on this setup). Using standard cavity rf measurement techniques, the quality factor of the cavity was measured as function of rf field amplitude, temperature, and trapped magnetic field, from which the average additional surface resistance caused by the trapped flux was estimated as a function of the strength of the rf field. The results showed a clear linear dependence on the rf field, i.e. a cubic field dependence of the

trapped vortex losses.

# 2. Nb3Sn

The A15 superconductor Nb3Sn is a particularly promising material for next-generation, highperformance SRF cavities [39]. Cornell University has a leading Nb3Sn SRF research program that aims at exploring the full potential of this material [40]. Nb3Sn coatings of a few microns thickness are produced on Nb substrate cavities via a tin vapor diffusion process [41, 42]. Optimization of this process has resulted in the first Nb3Sn SRF accelerator cavities ever to clearly outperform traditional solid-niobium cavities in cryogenic efficiency at usable accelerating fields. Cornell's 1.3GHz Nb3Sn cavities are now routinely reaching quality factors at 4.2K in the 1 to 2 × 10<sup>10</sup> range [43], more than one order of magnitude above those reachable with niobium at that temperature and rf frequency. Due to the bi-metal structure of these cavities, very small spatial thermal gradients are essential during cool-down to minimize thermoelectrically induced magnetic fields, that could be trapped and cause significant losses in rf fields. However, because of the small thermal gradients during cool-down, expulsion of residual ambient magnetic fields is poor, therefore still resulting in some trapped magnetic flux. Understanding the sensitivity of the residual resistance to trapped flux is therefore of particular importance for Nb3Sn cavities.

We used the same experimental procedure discussed in Section III A 1 for doped Nb. The results also showed a clear linear dependence on the rf field.

# 3. NbCu

Quarter-wave resonators of Nb films have been developed for the post-acceleration of heavy ions at CERN (HIE-ISOLDE project [44]). The Nb film of a few microns is deposited on a Cu cavity by the DC-bias sputtering technique. The resonant frequency is 101.28 MHz and operation temperature is 4.5 K. As the cavity is made of the thin film, its crystal structure contains fine grains and dislocations inside [45]. Flux expulsion during cooling down is typically poor because of a lot of possible pinning centers, uniformity of temperature caused by the Cu substrate, and QWR geometry. Hence, an ambient field can be fully trapped by the cavity. The bi-metal structure also gives rise to a possible thermoelectrically induced magnetic field during the cooling down, to be trapped by the pining centers [46].

The performance of the cavity was evaluated by the standard rf measurement and magnetometry of representative samples [9]. In the rf measurement, the quality factor of the cavity was obtained by field-decay and coupling information. From the quality factor along with a geometrical factor evaluated by rf simulation, rf surface
resistance averaged over the cavity surface was estimated as a function of the strength of rf fields. At 2.4 K, where the effect of quasi-particles are negligible i.e. no BCS resistance, the surface resistance turned out to be linearly dependent on the rf fields [47]. This behavior was previously reported in references [14, 48, 49]. The magnetometry revealed the depinning current of the Nb film showing such surface resistance [17]. This de-pinning current is larger than the literature value of clean bulk Nb, but still well below the surface current caused by the rf fields.

## B. Simulations

We model the vortex line as a discrete one-dimensional Python array of size L and spacing a. We use a = 38nm, a = 13nm and a = 40nm in the doped Nb, Nb3Sn and NbCu simulations, respectively, and L = 128 for all simulation data presented in this paper. Table I summarizes the material parameters used in the simulations. For each simulation, we start with a straight line vortex, u˜(˜z; 0) = 0, ∀z, and and find the solution at a later time t by implementing the equations as an ordinary differential equation (ODE). For the mean-field model, we integrate Eq. (2) with the pinning force given by Eq. (10) for three cycles (i.e. three periods of oscillation of the applied magnetic field) to relax the vortex, and then run the simulation for one additional cycle to calculate the resistance from an average of the dissipated power. For the local potential model, we integrate Eq. (2) for the elastic vortex line moving in the random potential as described in Section II C, and use three cycles to relax the vortex and three cycles to measure the dissipation; we repeat this protocol for ten random initial configurations of the disordered potential, and calculate the average12. Increasing the number of cycles in the relaxation and measurement processes does not lead to significant changes.

Figure 5 shows a plot of the sensitivity of the residual resistance to trapped magnetic flux as a function of the amplitude of the rf field for doped Nb (a), Nb3Sn (b) and NbCu (c). Red circles correspond to experimental measurements, multiplied by the correction factor G <sup>−</sup><sup>1</sup> ≈ 2 (for the Cornell doped-Nb and Nb3Sn cavities) and G <sup>−</sup><sup>1</sup> ≈ 3 (for the CERN NbCu cavity <sup>13</sup>), to account for vortex misalignment near the cavity equator and field depletion near the cavity poles (see Appendix B). We have also subtractted off the offset (lim<sup>B</sup>rf→<sup>0</sup> R0/Btrapped) of the measured sensitivity, which is presumably due to

loss mechanisms not involving macroscopic vortex motion. Blue and orange circles correspond to our numerical simulations using the mean-field and the localpotential models, respectively (the dashed lines emphasize the low-field linear behavior). The black line corresponds to our approximated analytical solution given by Eqs. (23) and (24). Note that our calculations correctly capture the low-field linear behavior observed in the experiments. As expected, our calculations for the crossover field BX, shown in Table I, are consistent with the simulation results for the mean-field model (but consistently smaller than the crossover field of the more realistic local-potential model.) Also, note that we could fit the experimental data if we use larger depinning currents (by a factor from about six for Nb3Sn to twenty for doped Nb) in our calculations. The discrepancy between theory and experiments is larger for doped Nb in part due to the low-frequency design (100MHz) of the NbCu cavity, and the small coherence length of Nb3Sn. The remaining discrepancy could be ascribed to a number of factors, which we discuss in Section III C. A word of caution: the fact that the analytical curve (black line) is close to the localpotential solution (orange circles) in (a) and (c) should be taken with a grain of salt. The most realistic model is the local-potential model. The mean-field approach relies on a number of uncontrolled approximations, and is particularly useful to provide order-of-magnitude estimates and physical insights, rather than accurate predictions.

### C. Discrepancy between theory and experiment

The theoretical curves in Figure 5 use a depinning current that is a factor of six to twenty too small to fit the experimental curves. The theory used the measured depinning current for one of the materials (NbCu), which by our estimates (Appendix A) is already too high to be due to point-like pinning centers (impurity doping). What could be the cause of the discrepancy?

As discussed in Section II and Appendix A, pinning on line-like impurities could be substantially stronger. Indeed, vortex pinning on Nb dislocation cell structures is known to reach values similar to those measured [55]. Such pinning could be enhanced by impurity doping, if the dopant preferentially segregated to the dislocation. We would anticipate that the annealing steps in the preparation of the niobium cavity would remove most of the dislocations. Pinning on grain boundaries<sup>14</sup>, if it is not inescapable, would likely produce large depinning fields and a residual resistance that depends on Brf, but the grain size in niobium is too large for our collective weak pinning theory to be applicable. The role of dislocations or grain boundaries for Nb3Sn is open for further study.

<sup>12</sup> We do not need average over samples in the mean-field model, which is deterministic.

<sup>13</sup> Note that we have used the elliptical shape of the Cornell cavities in the calculations of Appendix B. The correction factor for the CERN NbCu cavity, which has a QWR geometry, might be different.

<sup>14</sup> Pinning on tin-depleted regions in Nb3Sn, or other 3D defects, would likely behave similarly.

| Material | λ[nm]                  | ξ[nm]   |   | κ Bc[mT] | ρn[Ω·m]                | jd[A/m2      |     | ] f [GHz] BX[mT] |
|----------|------------------------|---------|---|----------|------------------------|--------------|-----|------------------|
| Doped Nb | 39 [50]                | 38 [50] | 1 |          | 152 1.8 × 10−8<br>[51] | 1010         | 1.3 | 16               |
| Nb3Sn    | 111 [31] 4.2 [52] 26.4 |         |   | 483      | 10−6<br>[53]           | 1010         | 1.3 | 8                |
| NbCu     | 30 [9]                 | 30 [9]  | 1 | 250      | 4.5 × 10−9             | 1010 [9, 17] | 0.1 | 28               |

TABLE I. Penetration depth λ, coherence length ξ, Ginzburg-Landau parameter κ ≡ λ/ξ, thermodynamic critical field (according to GL theory) <sup>B</sup><sup>c</sup> <sup>=</sup> <sup>φ</sup>0/(2<sup>√</sup> 2πλξ), normal state resistivity ρn, depinning current jd, and frequency f used in simulations for doped Nb, Nb3Sn and NbCu. The last column shows the crossover field BX, according to Eq. (26). For Nb3Sn, we have used values for j<sup>d</sup> that are higher than reported measurements [54] for tube-type Nb3Sn superconductors. In Appendix A, we do a sanity check of this higher threshold, calculating the pinning per impurity assuming each destroys superconductivity over some region. For a mean-free-path of ∼ 1nm, a 1% density of impurities destroying superconductivity over two lattice constants cubed will give depinning thresholds in this range, suggesting that our choice for j<sup>d</sup> is possible. The resistivity of the normal phase of NbCu has been estimated from DC Residual Resistivity Ratio measurements of a Nb film on quartz.

![](0__page_9_Figure_2.jpeg)

FIG. 5. Sensitivity of residual resistance to trapped flux as a function of the rf field for doped Nb (a), Nb3Sn (b) and NbCu (c) from experiments (red circles), analytical calculations (black line), and numerical simulations (blue and orange circles). Theory curves use j<sup>d</sup> = 10<sup>10</sup>A/m<sup>2</sup> . Note that we could obtain numerical agreement with the experimental data if we use larger depinning currents in our calculations. The value we use is that measured for bulk depinning in NbCu [9, 17] in (c), which admittedly has a very different morphology from the Nb cavity. This value is comparable to bulk pinning on dislocation cell structures in Nb [55]; pinning on surface roughness (relevant here) could be stronger especially in NbCu. In the theory curves for Nb3Sn, the value we use is the largest plausible value from point-like impurities (Appendix A); pinning on dislocations, grain boundaries, or tin-depleted regions would likely be stronger.

But what about NbCu, where the pinning current was measured? Here the depinning current was deduced by measuring the hysteresis as the external field was varied. This adds a force per unit length to the whole vortex (a bulk measurement), where the dissipation is due to a force on one end of the vortex. Pinning due to surface roughness, or due to defects that arise more often near the surface, could explain the discrepancy. NbCu surfaces are particularly rough, as are the current Nb3Sn surfaces. Surface roughness, like grain boundaries, would likely not be modeled well by collective weak pinning: each vortex would show little dissipation until pushed hard enough to detach from its pinning site. But a distribution of vortex surface pinning strengths could generate a field-dependent residual resistance.

Figure 5 shows the theoretical and experimental residual resistances per unit trapped flux. Is it possible that the experimental value for the trapped flux is in error? The cavities were cooled very slowly in a DC applied field (to avoid forces due to thermal gradients, which are usually maximized to expel flux [56–58]), and measurements show very little flux expulsion from the cavity as a whole <sup>15</sup> .

However, recent measurements [59, 60] show large heterogeneity in the heating due to trapped flux, both on the centimeter scale of the detector resolution and on the decimeter scale of the cavity. (The macroscale variations break the azimuthal symmetry, so are not due to the geometrical factors discussed in Appendix B). The simple theoretical picture of a uniform density of vortices independently oscillating with a single pinning strength is clearly inapplicable. The hypothesis that the flux remains homogeneous would demand that the cold regions have much larger pinning strength than the hot regions,

<sup>15</sup> Note that near T = 0, typical thermal gradients of ∼ Tc/m result in forces per length that are about 10<sup>6</sup> smaller than the pinning force used in our simulations for Nb. Most of the flux expulsion must happen during cool-down, when T is near Tc, since the depinning force vanishes as (T<sup>c</sup> − T) (5/12)(6−D) , according to GL theory. Flux expulsion by thermal gradients is still a topic of general interest, and deserves further investigation.

which seems tentatively unlikely since the grain sizes are larger in the cold regions (perhaps also indicating fewer dislocations within grains), and also the losses increased when dislocations were added deliberately [61]. The fact that the flux is not expelled from the cavity as a whole does not preclude the motion of flux within the cavity, either macroscopically or microscopically. If the vortices move, they either cluster into the hot regions, or they move within the cold regions to nearby traps where they are strongly pinned. The residual resistance due to the remaining vortices subject to collective weak pinning would be linear in Brf, with magnitude proportional by hj −1 d i. This motion to higher pinning would tend to reduce the dissipation per vortex. Also, a substantial fraction of the vortices moving to sites where they are inescapably and rigidly pinned (and hence not dissipating) could explain the discrepancy.

Measurements of the heterogeneity in the trapped flux would be useful. Macroscopically, is there more trapped flux in the hot regions? Microscopically, are the vortices trapped at grain boundaries or other structures? Is the pinning dependent on the grain orientation (and hence the orientation of the screw dislocations, dominant in BCC metals)? Is it dependent on the misorientation between grains? Answering these questions could be of practical use. Single-crystal cavities have been tried, but without controlling the surface orientation. One could also vary the grain orientation distribution or 'texture' by suitable plastic deformation before the final cavity is stamped into shape. In doped Nb, the goal likely is to reduce all pinning and to maximize thermal gradients during cooling to expel the flux. In Nb3Sn films grown on Nb and Nb films grown on Cu, thermal gradients cause thermoelectric currents which induce trapped flux, so slow cooling is necessary – perhaps making stronger pinning beneficial. This issue deserves further study.

### IV. FINAL REMARKS

We have used the a model of vortex dynamics and collective weak pinning theory to study vortex dissipation in superconductors. We then applied our analysis to experiments performed in 1.3GHz Nb3Sn, doped Nb, and 100MHz NbCu cavities. Using simple analytical calculations and standard numerical simulations, we describe the low-field linear regime of the sensitivity of the residual resistance to trapped magnetic flux. Our results agree well with experiments performed in CERN and Cornell. We define a crossover field BX, which increases with both inverse frequency and depinning current, and that marks a transition from a regime hysteretic to viscousdominated losses.

We propose the tuning of material parameters as a method to minimize the crossover field and reduce power dissipation in SRF cavities. Our simple approximated formulas for the slope of the sensitivity to trapped flux (Eq. (24)) and the crossover field (Eq. (26)) provide a systematic way to control and shed light into hystereticdominated trapped-flux dissipation in SRF cavities. The slope A and the crossover field B<sup>X</sup> scale as f λ2/j<sup>d</sup> and (ρn/f) 1/2 (jd/κ), respectively. As it should be anticipated, high-f, high-λ and low-j<sup>d</sup> cavities yield large dissipation. It would be interesting to apply our analysis to other Nb systems, such as the Fermilab N-doped Nb cavities [18], and to adapt or extend our theory in view of the exciting (ongoing) research developments on thermal flux expulsion, heterogeneous flux trapping and the role of extended defects such as dislocations and grain boundaries.

# Appendix A: Sanity checks

Here we discuss some approximations and sanity checks that are associated with the derivation and analytical solution of the mean-field model.

We begin with a discussion of the characteristic length scales that corroborate the collective weak pinning scenario and the point-force approximation. Let a<sup>y</sup> and a<sup>z</sup> be the vortex amplitudes of deformation in the y and z directions, respectively. We can use the solution derived in Section II B to show that,

$$a\_y \equiv \lambda \,\tilde{u}\,\left(\tilde{z} = 0; \tilde{t} = 1/4\right) = \frac{\kappa}{\sqrt{2} \,c\left(\kappa\right)} \left(\frac{B\_{rf}}{B\_c}\right)^2 x\_d,\tag{A1}$$

where

$$x\_d \equiv \frac{B\_c}{\mu\_0 \, j\_d},\tag{A2}$$

is a characteristic length ∝ λ jo/jd. The amplitude in the z direction is given by

$$a\_z \equiv \lambda \left. \tilde{z} \right|\_{\tilde{u}(1/4)\to 0} = \frac{B\_{rf}}{B\_c} x\_d. \tag{A3}$$

Also, the curvature radius of the vortex line at z = 0 is given by

$$r\_c = \frac{\lambda}{\tilde{u}''|\_{\tilde{t}\to 1/4}} = \frac{c\left(\kappa\right)}{\sqrt{2}\,\kappa} x\_d.\tag{A4}$$

To restore the physical boundary condition at z = 0 (du/dz = 0), we ad hoc bend the vortex line over a distance λ from the surface, so that |u <sup>00</sup>| ≈ |u 0 |/λ. The curvature radius at z = 0 then becomes

$$r\_{\lambda} = \frac{\lambda}{|\tilde{u}'|\_{\tilde{t}\to 1/4}} = \frac{c\left(\kappa\right)}{\sqrt{2}} \frac{B\_c}{B\_{rf}} \,\xi. \tag{A5}$$

For completeness, Eqs. (36) and (A2) result in

$$\left|L\_c\right|^2 = \frac{\sqrt{2}\,c\left(\kappa\right)}{\kappa}\,x\_d\,\xi.\tag{A6}$$

Figure 6 shows our mean-field solutions for a<sup>y</sup> (dashed curves), a<sup>z</sup> (dash-dotted), r<sup>λ</sup> (dotted), and L<sup>c</sup> (solid) for doped-Nb (green), Nb3Sn (blue) and NbCu (red) superconductors (note that all materials have the same az.) For all three materials, the collective weak pinning assumption L<sup>c</sup> ξ is safely satisfied. At large fields, the radii of curvature become small, and the amplitudes of motion become large, thus justifying the point-force approximation. Note that the transverse amplitudes of motion (ay) lie above the micron scale for fields above Brf ≈ 30-70mT. Grain sizes of Nb3Sn are of order 1µm, emphasizing the role played by extended defects in this case.

![](0__page_11_Figure_1.jpeg)

FIG. 6. Mean-field analytical calculations for the amplitude of deformation in the y (dashed) and z (dash-dotted) directions, the radii of curvature (dotted), and the depinning length (solid), for doped-Nb (green), Nb3Sn (blue) and NbCu (red) superconductors.

Next we discuss the area swept by the vortex oscillations to justify our assumption of independent vortex lines. The area sMF in the y-z plane that is swept by each vortex oscillation is related to the average dissipated power per vortex P1/sMF = 2 f f<sup>P</sup> , and is given by,

$$s\_{\rm MF} = 2\lambda^2 \int\_0^\infty \tilde{u}(\tilde{z}; 1/4) \, d\tilde{z} = \frac{\sqrt{2}}{3} \frac{\kappa}{c\left(\kappa\right)} \left(\frac{B\_{rf}}{B\_c}\right)^3 x\_d^{-2}.$$

Figure 7 shows a plot of sMF as a function of Brf for doped-Nb (green), Nb3Sn (blue) and NbCu (red) superconductors. Note that sMF approaches 1µm<sup>2</sup> at high fields, which is about the grain size of typical Nb3Sn, suggesting that discrepancies with experiments might arise due to the vortex interaction with grain boundaries. On the other hand, from Btrap/φ<sup>0</sup> = N/s, we estimate a density of one vortex per 10<sup>4</sup> -10<sup>3</sup>µm<sup>2</sup> for a trapped magnetic induction of about 5-50mG, suggesting that the approximation of non-interacting vortices is consistent.

We end this section with a discussion about the relationship between the depinning current and the density of impurities, and the high depinning current used in our simulations. Here we use Eqs. (36) and (34) to eliminate Lc, and derive a formula relating the density of impurities nD, the individual pinning force F<sup>i</sup> , and the depinning current jd,

$$n\_{\mathcal{D}}{}^2 = \frac{2\,\epsilon\,\phi\,\phi^3 j\_d{}^3}{F\_i{}^4\,\xi^{3-2\mathcal{D}}}.\tag{A7}$$

![](0__page_11_Figure_8.jpeg)

FIG. 7. Mean-field calculation of the area swept by one vortex oscillation in the y-z plane for doped Nb (green), Nb3Sn (blue) and NbCu (red) superconductors.

We estimate the individual pinning force from the condensation energy gained to move a vortex line from the border to the center of a defect potential well of size ξ, i.e.

$$F\_i \approx a^{3-\mathcal{D}} \xi^{\mathcal{D}} \frac{B\_c}{2\,\mu\_0 \xi},\tag{A8}$$

where we have assumed that the impurity destroys superconductivity over the volume a <sup>3</sup>−Dξ <sup>D</sup>, with a of order of an atomic size. Plugging (A8) back into (A7) results in

$$n\_{\mathcal{D}} \approx 32 \times 2^{1/4} \pi^2 \sqrt{c\left(\kappa\right)} \left(\frac{a^2}{\xi}\right)^{\mathcal{D}} \left(\frac{\xi}{a^2} \sqrt{\frac{\lambda}{x\_d}}\right)^3. \quad (A9)$$

We use Eq. (A9) to estimate the density of point-like impurities from the depinning current for a range of values of the atomic distance a. For Nb3Sn, we find a density of 2–130 Nb atoms per impurity for a ∼ 1-2 unit cell lengths, and a mean-free-path of ∼ 1nm, where we have used BCS formulas for the dependence of λ and ξ on mean free path [52]. Notice that this estimate is highly sensitive to the value of a, yet it does not rule out the high depinning current that we have used if the impurities affect a sufficiently large region. On the other hand, our estimates suggest that high depinning currents cannot be attributed to point-like impurities alone for doped-Nb and NbCu. Here we note that the term (a <sup>2</sup>/ξ) <sup>D</sup> in (A9) suggests that consistent densities of defects can be associated with larger depinning currents for extended defects (with D > 0). Additional experimental measurements of the depinning current and mean free path might help test our assumptions using collective weak pinning theory.

### Appendix B: Field-alignment correction

In our calculations of the residual resistance, we assume that each vortex is initially perpendicular to the superconductor surface, and is subject to the same value of the rf magnetic field. However, rf fields in real cavities

![](0__page_12_Figure_1.jpeg)

FIG. 8. Transverse radius (blue curve) and normalized amplitude of the rf magnetic field (red) as a function of the longitudinal distance z for the Cornell 1.3GHz Nb3Sn cavity. The inset illustrates a similarly arranged cavity (gray disk), with red circles corresponding to the rf field, and the black and yellow lines corresponding to two possible directions for the DC field that creates most of the trapped magnetic flux.

are larger near the equator. Figure 8 shows the normalized amplitude of the rf magnetic field (red curve) and the cavity radius (blue) as a function of the longitudinal coordinate z (not to be mistaken by the superconductor depth coordinate in the main text) for the Cornell Nb3Sn cavity. The inset illustrates the upper portion of a similarly arranged cavity (gray disk), with the red circles representing the rf magnetic field at the surface (the field becomes smaller near the poles), and the black and yellow lines representing two possible directions for the DC magnetic field that creates most of the trapped magnetic flux. The black horizontal and the yellow vertical lines correspond to the DC fields in the Cornell and Cern experimental setups, respectively. We then expect important corrections due to an interplay between field depletion at the cavity poles and a non-uniform density of vortices.

The density of vortices ρ = ρ(z, θ) for a DC magnetic

field BDC parallel or perpendicular to the z axis is given by,

$$\rho(z) \propto \frac{1}{\sqrt{1 + R'^2}} \times \begin{cases} |R'|, & \text{for } B\_{DC} \parallel z, \\ |\cos \theta|, & \text{for } B\_{DC} \perp z, \end{cases}$$

where R<sup>0</sup> ≡ dR/dz, and θ is the polar angle in cylindrical coordinates ((R, θ, z)). The surface area can be written as an integral over z and θ of the ring infinitesimal area dsring = R p 1 + R0<sup>2</sup> dz dθ. We also know the magnetic inductance Brf as a function of z. In the region where the sensitivity to trapped flux increases linearly with the rf field, the total dissipated power is proportional to R Brf 3 ρ R dz. In our model calculations, we have assumed Brf (z) = Brf (0) and uniform ρ. Thus, to make contact with the experimental results, we need correct our predictions by a factor G, defined as

$$\mathcal{G} = \frac{\int B\_{rf}(z)^3 \rho(z,\theta) R(z) \sqrt{1 + R'^2} dz \, d\theta}{\int B\_{rf}(0)^3 \cdot 1 \cdot R(z) \sqrt{1 + R'^2} dz \, d\theta}, \tag{B1}$$

where ρ is given by Eq. (B1). Using the data shown in Fig 8, we find G = 0.52 and 0.37 for BDC parallel and perpendicular to the z axis, respectively. This correction makes our theoretical prediction closer to the experimental results.

### ACKNOWLEDGMENTS

We thank useful conversations with A. Gurevich, S. Posen, D. Hartill, and S. Calatroni. DBL and JPS were supported by the US National Science Foundation under Award OIA-1549132, the Center for Bright Beams. DH, PNK and ML were supported by DOE Award No. DE-SC0008431, and NSF Award No. NSF PHY-1416318 and NSF PHY-1734189.

- [1] G. Blatter, M. V. Feigel'man, V. B. Geshkenbein, A. I. Larkin, and V. M. Vinokur, Rev. Mod. Phys. 66, 1125 (1994).
- [2] E. H. Brandt, Reports on Progress in Physics 58, 1465 (1995).
- [3] R. P. Huebener, Magnetic Flux Structures in Superconductors (Springer-Verlag, Berlin, 2001).
- [4] G. Blatter and V. B. Geshkenbein, "Vortex matter," in The Physics of Superconductors: Vol. I. Conventional and High-Tc Superconductors, edited by K. H. Bennemann and J. B. Ketterson (Springer Berlin Heidelberg, Berlin, Heidelberg, 2003) pp. 725–936.
- [5] H. Padamsee, J. Knobloch, and T. Hays, RF Superconductivity for Accelerators (WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim, 2008).
- [6] H. Padamsee, RF Superconductivity: Sicence, Technol-

ogy, and Applications (WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim, 2009).

- [7] S. Posen, N. Valles, and M. Liepe, Phys. Rev. Lett. 115, 047001 (2015).
- [8] D. B. Liarte, S. Posen, M. K. Transtrum, G. Catelani, M. Liepe, and J. P. Sethna, Superconductor Science and Technology 30, 033002 (2017).
- [9] A. Miyazaki and W. Venturini Delsolaro, ArXiv e-prints (2018), arXiv:1806.04443 [physics.acc-ph].
- [10] S. Posen, Understanding and Overcoming Limitation Mechanisms in Nb3Sn Supercomputing RF Cavities, Ph.D. thesis, Cornell University (2014).
- [11] D. Gonnella, The fundamental science of nitrogen-doping of niobium superconducting cavities, Ph.D. thesis, Cornell University (2016).
- [12] D. Gonnella, J. Kaufman, and M. Liepe, Jour-

nal of Applied Physics 119, 073904 (2016), http://dx.doi.org/10.1063/1.4941944.

- [13] M. Checchin, M. Martinello, A. Grassellino, A. Romanenko, and J. F. Zasadzinski, Superconductor Science and Technology 30, 034003 (2017).
- [14] C. Benvenuti, S. Calatroni, I. Campisi, P. Darriulat, M. Peck, R. Russo, and A.-M. Valente, Physica C: Superconductivity 316, 153 (1999).
- [15] A. Gurevich and G. Ciovati, Phys. Rev. B 87, 054502 (2013).
- [16] D. Hall, D. Liarte, M. Liepe, and J. Sethna (JACoW, Geneva, Switzerland, 2017) pp. 1127– 1129, https://doi.org/10.18429/JACoW-IPAC2017- MOPVA118.
- [17] A. Miyazaki, TTC Topical Workshop RF Superconductivity: Pushing Cavity Performance Limits (2018).
- [18] M. Checchin, M. Martinello, A. Grassellino, S. Aderhold, S. K. Chandrasekaran, O. S. Melnychuk, S. Posen, A. Romanenko, and D. A. Sergatskov, Applied Physics Letters 112, 072601 (2018), https://doi.org/10.1063/1.5016525.
- [19] O. M. Auslaender, L. Luan, E. W. Straver, J. E. Hoffman, N. C. Koshnick, E. Zeldov, D. A. Bonn, R. Liang, W. N. Hardy, and K. A. Moler, Nature Physics 5, 35 (2009).
- [20] J. I. Gittleman and B. Rosenblum, Phys. Rev. Lett. 16, 734 (1966).
- [21] J. P. Sethna, K. A. Dahmen, and C. R. Myers, Nature 410, 242 (2001).
- [22] D. S. Fisher, Physics Reports 301, 113 (1998).
- [23] J. P. Sethna, M. K. Bierbaum, K. A. Dahmen, C. P. Goodrich, J. R. Greer, L. X. Hayden, J. P. Kent-Dobias, E. D. Lee, D. B. Liarte, X. Ni, K. N. Quinn, A. Raju, D. Z. Rocklin, A. Shekhawat, and S. Zapperi, Annual Review of Materials Research 47, 217 (2017), https://doi.org/10.1146/annurev-matsci-070115-032036.
- [24] S. Atis, A. K. Dubey, D. Salin, L. Talon, P. Le Doussal, and K. J. Wiese, Phys. Rev. Lett. 114, 234502 (2015).
- [25] M. Kardar, G. Parisi, and Y.-C. Zhang, Phys. Rev. Lett. 56, 889 (1986).
- [26] A. M. Campbell, Journal of Physics C: Solid State Physics 4, 3186 (1971).
- [27] R. Willa, V. B. Geshkenbein, and G. Blatter, Phys. Rev. B 92, 134501 (2015).
- [28] E. H. Brandt, Phys. Rev. B 68, 054506 (2003).
- [29] J. Bardeen and M. J. Stephen, Phys. Rev. 140, A1197 (1965).
- [30] A. I. Larkin and Y. N. Ovchinnikov, Journal of Low Temperature Physics 34, 409 (1979).
- [31] M. Tinkham, Introduction to Superconductivity, 2nd ed. (McGraw-Hill, New York, 1996).
- [32] C. P. Bean, Phys. Rev. Lett. 8, 250 (1962).
- [33] A. Grassellino, A. Romanenko, D. Sergatskov, O. Melnychuk, Y. Trenikhina, A. Crawford, A. Rowe, M. Wong, T. Khabiboulline, and F. Barkov, Superconductor Science and Technology 26, 102001 (2013).
- [34] J. T. Maniscalco, D. Gonnella, and M. Liepe, Journal of Applied Physics 121, 043910 (2017), https://doi.org/10.1063/1.4974909.
- [35] M. Martinello, S. Aderhold, S. Chandrasekaran, M. Checchin, A. Grassellino, O. Melnychuk, S. Posen, A. Romanenko, and D. Sergatskov, Proceedings of SRF2017, Lanzhou, China (2017).
- [36] M. Liepe, R. Eichhorn, F. Furuta, G. Ge, D. Gonnella, G. Hoffstaetter, A. Crawford, A. Grassellino, A. Hocker, O. Melnychuk, A. Romanenko, A. Rowe, D. Sergatskov,

R. Geng, A. Palczewski, C. Reece, and M. Ross, Proceedings of IPAC2014, Dresden, Germany (2014).

- [37] P. Koufalis, M. Liepe, J. Maniscalco, and T. Oseroff, Proceedings of IPAC2018, Vancouver, Canada, paper WEPMF039 (2018).
- [38] P. Koufalis and M. Liepe, Proceedings of IPAC2018, Vancouver, Canada, paper WEPMF041 (2018).
- [39] S. Posen and D. L. Hall, Superconductor Science and Technology 30, 033004 (2017).
- [40] S. Posen and M. Liepe, Phys. Rev. ST Accel. Beams 17, 112001 (2014).
- [41] G. M¨uller, P. Kneisel, D. Mansen, H. Piel, J. Pouryamout, and R. W. R¨oth, Proceedings of EPAC1996, Barcelona, Spain , 2085 (1996).
- [42] S. Posen and M. Liepe, Proceedings of IPAC2011, San Sebastian, Spain (2011).
- [43] S. Posen, M. Liepe, and D. L. Hall, Applied Physics Letters 106, 082601 (2015), https://doi.org/10.1063/1.4913247.
- [44] Y. Kadi, Y. Blumenfeld, W. Venturini Delsolaro, M. A. Fraser, M. Huyse, A. Papageorgiou Koufidou, J. A. Rodriguez, and F. Wenander, J. Phys. G: Nucl. Part. Phys 44, 084003 (2017).
- [45] A. Sublet, I. Aviles Santillana, B. Bartova, S. Calatroni, A. Jecklin, I. Mondino, M. Therasse, W. Venturini Delsolaro, and P. Zhang, Proceedings of IPAC2014, Dresden, Germany , 2572 (2014).
- [46] A. Miyazaki, Y. Kadi, K. Schirm, A. Sublet, S. Teixeira, M. Therasse, and W. Venturini Delsolaro, 18th International Conference on RF Superconductivity, Lanzhou, China (2017).
- [47] A. Miyazaki, Y. Kadi, K. Schirm, A. Sublet, S. Teixeira, M. Therasse, and W. Venturini Delsolaro, 5th TTC Thin Film Working Group Meeting (2018).
- [48] W. Weingarten, Proc. 7th Workshop on RF Superconductivity (1995).
- [49] G. Ciovati and A. Gurevich, Proceedings of SRF2007, Peking Univ., Beijing, China (2007).
- [50] B. W. Maxfield and W. L. McLean, Phys. Rev. 139, A1515 (1965).
- [51] B. B. Goodman and G. Kuhn, J. Phys. France 29, 240 (1968).
- [52] T. P. Orlando, E. J. McNiff, S. Foner, and M. R. Beasley, Phys. Rev. B 19, 4545 (1979).
- [53] A. Godeke, Superconductor Science and Technology 19, R68 (2006).
- [54] M. Sumption, S. Bhartiya, C. Kovacks, X. Peng, E. Gregory, M. Tomsic, and E. Collings, Cryogenics 52, 91 (2012).
- [55] A. T. Santhanam, Journal of Materials Science 11, 1099 (1976).
- [56] A. Romanenko, A. Grassellino, A. C. Crawford, D. A. Sergatskov, and O. Melnychuk, Applied Physics Letters 105, 234103 (2014), https://doi.org/10.1063/1.4903808.
- [57] S. Huang, T. Kubo, and R. L. Geng, Phys. Rev. Accel. Beams 19, 082001 (2016).
- [58] S. Posen, M. Checchin, A. C. Crawford, A. Grassellino, M. Martinello, O. S. Melnychuk, A. Romanenko, D. A. Sergatskov, and Y. Trenikhina, Journal of Applied Physics 119, 213903 (2016), https://doi.org/10.1063/1.4953087.
- [59] S. Posen, M. Checchin, C. Crawford, A. Grassellino, M. Martinello, A. Melnychuk, A. Romanenko, D. Sergatskov, Z. Sung, and Y. Trenikhina, Tesla Technology

Collaboration (TTC) Meeting, Milan, Italy (2018).

- [60] M. Martinello, Z. Sung, and S. Posen, Tesla Technology Collaboration (TTC) Meeting, Saitama, Japan (2018).
- [61] S. Posen, Tesla Technology Collaboration (TTC) Meeting, Saitama, Japan (2018).