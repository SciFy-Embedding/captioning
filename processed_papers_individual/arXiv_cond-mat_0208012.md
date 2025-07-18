# arXiv:cond-mat_0208012

**Paper ID:** 3e11b901384b274e362b3aeca5bf7a96

**PDF Path:** apl/Superconductors/arXiv:cond-mat_0208012.pdf

**Processing Status:** complete

**Captions Added:** 11

**Generated:** 2025-06-24T13:44:28.117685

---

# Theory of de Haas-van Alphen Effect in Type-II Superconductors

Kouji Yasui and Takafumi Kita[∗](#page-14-0)

Division of Physics, Hokkaido University, Sapporo 060-0810, Japan

(Dated: December 24, 2021)

Theory of quasiparticle spectra and the de Haas-van Alphen (dHvA) oscillation in type-II superconductors are developed based on the Bogoliubov-de Gennes equations for vortex-lattice states. As the pair potential grows through the superconducting transition, each degenerate Landau level in the normal state splits into quasiparticle bands in the magnetic Brillouin zone. This brings Landaulevel broadening, which in turn leads to the extra dHvA oscillation damping in the vortex state. We perform extensive numerical calculations for three-dimensional systems with various gap structures. It is thereby shown that (i) this Landau-level broadening is directly connected with the average gap at H = 0 along each Fermi-surface orbit perpendicular to the field H; (ii) the extra dHvA oscillation attenuation is caused by the broadening around each extremal orbit. These results imply that the dHvA experiment can be a unique probe to detect band- and/or angle-dependent gap amplitudes. We derive an analytic expression for the extra damping based on the second-order perturbation with respect to the pair potential for the Luttinger-Ward thermodynamic potential. This formula reproduces all our numerical results excellently, and is used to estimate band-specific gap amplitudes from available data on NbSe2, Nb3Sn, and YNi2B2C. The obtained value for YNi2B2C is fairly different from the one through a specific-heat measurement, indicating presence of gap anisotropy in this material. C programs to solve the two-dimensional Bogoliubov-de Gennes equations are available at <http://phys.sci.hokudai.ac.jp/~kita/index-e.html>.

#### I. INTRODUCTION

The de Haas-van Alphen (dHvA) experiment on normal metals has been a unique and powerful tool to probe their Fermi surfaces.[1,2](#page-14-0) The main purpose of this paper is to establish theoretically that it can even be used to detect detailed gap structures of type-II superconductors.

Back in 1976, Graebner and Robbins discovered the dHvA oscillation in 2H-NbSe<sup>2</sup> persisting down through the superconducting upper critical field Hc2. [3](#page-14-0) It was after 15 years later when Onuki ¯ et al. first reconfirmed it.[4](#page-14-0) Since then, however, a considerable number of materials have been found to display the dHvA oscillation in the vortex state. They include: A15 superconductors V3Si[5,6](#page-14-0) Nb3Sn,[7](#page-14-0) a borocarbide superconductor YNi2B2C,[8,9](#page-14-0) heavy-fermion superconductors CeRu2, [10](#page-14-0) URu2Si2, [11](#page-14-0),[12](#page-14-0) UPd2Al3, [13](#page-14-0) CeCoIn5, [14](#page-14-0) and an organic superconductor κ-(BEDT-TTF)2Cu(NCS)2; [15](#page-14-0) see Refs. [16](#page-14-0) and [17](#page-14-0) for a recent review. Basic features of the oscillation are be summarized as follows: (i) the dHvA frequencies remain unchanged through the transition; (ii) the oscillation amplitude experience an extra attenuation; (iii) the cyclotron mass does not change except for strongly correlated heavy fermion systems.

It is somewhat surprising that the dHvA oscillation is observable even in superconductors without a welldefined Fermi surface. Many theories have been presented to explain the persistent oscillation and the extra damping,[18,19,20](#page-14-0),[21,22,23,24,25](#page-14-0),[26,27,28,29,30](#page-14-0) which may be classified into three categories.

The first approach applies a Bohr-Sommerfeld semiclassical quantization to either the Brandt-Pesch-Tewordt[31](#page-14-0) (BPT) Green's function near Hc2, [18](#page-14-0) the electron number N at H = 0,[19](#page-14-0) or Dyson's equation at H = 0,[20](#page-14-0) for obtaining the oscillatory behavior of the magnetization. As may be seen by this diversity of the applications, however, there is no unique semiclassical quantization scheme for quasiparticles superconductors; thus, the validity of the procedure is not clear. This category includes Maki's theory,[18](#page-14-0) which was later reproduced by Wasserman and Springford[21](#page-14-0) by treating the BTP self-energy as the extra broadening factor in the normal-state thermodynamic potential and then following Dingle's procedure.[32](#page-14-0) However, the BTP self-energy itself is obtained within the quasiclassical approximation without the Landau-level structure so that this approximation may also be questionable.

The second approach relies on some approximate analytic solutions for the Bogoliubov-de Gennes (BdG) equations or the equivalent Gor'kov equations, such as the coherent-potential approximation,[22,23](#page-14-0) the diagonalpairing approximation,[24](#page-14-0) or the Ginzburg-Landau (GL) expansion for the free energy with respect to the order parameter.[25,26,27](#page-14-0) However, quantitative estimations of those approximations are yet to be carried out. It should also be noted that the GL expansion necessarily integrates out the quasiparticle degrees of freedom, so that the physical origin of the extra oscillation damping may be obscured in the GL approach.

The third approach solves the BdG equations numerically without approximations.[29,30](#page-14-0) Norman et al. [29](#page-14-0) thereby extracted an analytic formula for the dHvA oscillation damping through a fitting to their numerical data.[29](#page-14-0) However, it is a two-dimensional calculation for the isotropic s-wave pairing where the number of Landau levels below the Fermi level is N<sup>F</sup> ∼ 10 at Hc2. As may be realized from the appearance of the quantized Hall effect, two-dimensional systems in high magnetic fields may be qualitatively different from threedimensional systems. Thus, the obtained formula may <span id="page-1-0"></span>not be appropriate for describing real three-dimensional materials with N<sup>F</sup> ≫ 1. On the other hand, another calculation by Miller and Gy¨orffy for a two-dimensional lattice model[30](#page-14-0) corresponds to the low-field limit near Hc<sup>1</sup> [33](#page-14-0) and may not be suitable to explain the experiments. Moreover, calculations for lattice models have a flaw that they cannot yield continuous magnetic oscillation due to the commensurability condition between the underlying lattice and the vortex lattice.

Notice finally that most of the above theories consider only the isotropic s-wave pairing. Especially, no numerical studies have been performed yet for anisotropic pairings or three-dimensional systems.

With these observations, we will perform both numerical and analytic calculations for three-dimensional BdG equations with various gap structures. They can be solved efficiently by the Landau-level-expansion method (LLX), which was formulated for vortex-lattice states of arbitrary pairing symmetry[34](#page-14-0) and used successfully to compare low-energy quasiparticle spectra between s- and d-wave pairings.[35](#page-14-0) We will thereby clarify how the discrete Landau levels experience quantitative changes as the pair potential grows below Hc2. Another purpose is to find out the connection between the gap anisotropy and the extra dHvA amplitude attenuation. Terashima et al. [9](#page-14-0) reported a dHvA experiment on YNi2B2C where an oscillation is observed to persist down to a field ∼0.2Hc2. On the other hand, a specific-heat experiment at H = 0 shows a power-law behavior ∝T <sup>3</sup> at low temperatures,[36](#page-14-0) indicating existence of gap anisotropy in this material. Indeed, Izawa et al. [37](#page-14-0) recently reported presence of four point nodes in the gap based on a thermal-conductivity measurement. Miyake[19](#page-14-0) argued that point or line nodes along the extremal orbit may weaken the damping, and proposed to use the dHvA effect as a probe for gap anisotropy. We examine this possibility in full details and present a quantitative theory on the issue.

This paper is organized as follows. Section II provides a formulation to solve the BdG equations for vortexlattice states. Section [III](#page-3-0) presents calculated quasiparticle spectra and the dHvA oscillation for two-dimensional systems to clarify their basic features as well as the origin of the extra dHvA oscillation damping in the vortex state. In Sec. [IV,](#page-7-0) it is demonstrated that the gap anisotropy at H = 0 can be detectable via the dHvA oscillation in the vortex state based on both numerical and analytic calculations for three-dimensional systems with various gap structures. Section [V](#page-11-0) presents estimations of energy gap for NbSe2, Nb3Sn, and YNi2B2C using the analytic formula obtained in Appendix [C](#page-13-0). Section [VI](#page-11-0) concludes the paper with a brief summary. In Appendix [A,](#page-11-0) we derive a convenient expression for the thermodynamic potential. Appendix B summarizes the expressions of basis functions and overlap integrals used in the numerical calculations. In Appendix C, we derive an analytic expression for the extra dHvA oscillation damping in the vortex state. A brief report of the contents was already presented in Ref. [38](#page-14-0).

## II. FORMULATION

### A. Bogoliubov-de Gennes equations

Throughout the paper we will rely on the mean-field BdG equations, which obtain the quasiparticle wavefunctions u<sup>s</sup> and v ∗ <sup>s</sup> with a positive eigenvalue E<sup>s</sup> > 0 by

$$\begin{aligned} \int d\mathbf{r}\_2 \begin{bmatrix} \frac{\mathcal{H}(\mathbf{r}\_1, \mathbf{r}\_2)}{\Delta(\mathbf{r}\_1, \mathbf{r}\_2)} & \frac{\Delta(\mathbf{r}\_1, \mathbf{r}\_2)}{\Delta^\*(\mathbf{r}\_1, \mathbf{r}\_2)} \\ \Delta^\dagger(\mathbf{r}\_1, \mathbf{r}\_2) & -\underline{\mathcal{H}}^\*(\mathbf{r}\_1, \mathbf{r}\_2) \end{bmatrix} \begin{bmatrix} \mathbf{u}\_s(\mathbf{r}\_2) \\ -\mathbf{v}\_s^\*(\mathbf{r}\_2) \end{bmatrix} \\ &= E\_s \begin{bmatrix} \mathbf{u}\_s(\mathbf{r}\_1) \\ -\mathbf{v}\_s^\*(\mathbf{r}\_1) \end{bmatrix}. \end{aligned} \tag{1}$$

Here ∆ is the pair potential and H denotes the normalstate Hamiltonian in the magnetic field; both are 2×2 matrices to describe the spin degrees of freedom. The symbol † denotes Hermitian conjugate in both the coordinate and spin variables as [∆† (r1, r2)]σ1σ<sup>2</sup> =∆<sup>∗</sup> σ2σ<sup>1</sup> (r2, r1) with σ<sup>j</sup> =↑, ↓. With this definition, we can see immediately that the matrix in Eq. (1) is Hermitian.

We adopt the free-particle Hamiltonian for H:

$$\underline{\mathcal{H}}(\mathbf{r}\_1, \mathbf{r}\_2) = \delta(\mathbf{r}\_1 - \mathbf{r}\_2) \left\{ \frac{\left[ -i\hbar \nabla\_2 + \frac{e}{c} \mathbf{A}(\mathbf{r}\_2) \right]^2}{2m\_e} - \varepsilon\_\mathcal{F} \right\} \underline{\mathbf{1}},\tag{2}$$

where me, −e (e > 0), c, and ε<sup>F</sup> are the electron mass, the electron charge, the light velocity, and the chemical potential, respectively. We will not consider the spin paramagnetism throughout. We also neglect the spatial variation of the magnetic field as appropriate for the relevant high-κ materials. Then, the vector potential A can be expressed using the symmetric gauge as

$$\mathbf{A}(\mathbf{r}) = -\frac{1}{2}B\hat{\mathbf{z}} \times \mathbf{r} \,, \tag{3}$$

where B denotes the average flux density, and we have chosen the field along −ˆz for convenience.

The pair potential in turn is given with respect to the quasiparticle wavefunctions as

$$
\underline{\Delta}(\mathbf{r}\_1, \mathbf{r}\_2) = V(\mathbf{r}\_1 - \mathbf{r}\_2) \, \underline{\Phi}(\mathbf{r}\_1, \mathbf{r}\_2) \,, \tag{4}
$$

where V denotes the interaction and Φ is the order parameter defined by

$$\underline{\Phi}(\mathbf{r}\_1, \mathbf{r}\_2) \equiv \sum\_s \left[ \mathbf{u}\_s(\mathbf{r}\_1) \mathbf{v}\_s^T(\mathbf{r}\_2) - \mathbf{v}\_s(\mathbf{r}\_1) \mathbf{u}\_s^T(\mathbf{r}\_2) \right]$$

$$\times \frac{1}{2} \tanh \frac{E\_s}{2k\_B T},\tag{5}$$

with T the temperature and <sup>T</sup> denoting the transpose.

It is shown in Appendix [A](#page-11-0) that the thermodynamic potential corresponding to Eqs. (1)-(4) is given by

$$
\Omega = -k\_\mathrm{B} T \sum\_s \ln(1 + \mathbf{e}^{-E\_s/k\_\mathrm{B}T}) - \sum\_s E\_s \int |\mathbf{v}\_s(\mathbf{r})|^2 \, d\mathbf{r}
$$

$$
-\frac{1}{2} \int \int \mathrm{Tr} \, \underline{\Delta}^\dagger(\mathbf{r}\_1, \mathbf{r}\_2) \, \underline{\Phi}(\mathbf{r}\_2, \mathbf{r}\_1) \, \mathrm{d}\mathbf{r}\_1 \mathrm{d}\mathbf{r}\_2 \,, \tag{6}
$$

<span id="page-2-0"></span>where Tr denotes taking trace over spin variables. This expression will be useful to obtain an analytic expression for the extra dHvA amplitude attenuation in the vortex state. The magnetization is then calculated by

$$M = -\frac{\partial(\Omega/\mathcal{V})}{\partial B},\tag{7}$$

where V is the volume of the system.

#### B. Vortex lattices and basic vectors

Solving the above equations for general non-uniform systems is a formidable task. For lattice states, however, it can be reduced into a numerically tractable problem using the symmetry that they are periodic with a single flux quantum φ<sup>0</sup> ≡ hc/2e per unit cell. We hence define a pair of basic vectors by

$$\begin{cases} \mathbf{a}\_1 \equiv (a\_{1x}, a\_{1y}, 0) \\ \mathbf{a}\_2 \equiv (0, a\_2, 0) \end{cases}, \quad (\mathbf{a}\_1 \times \mathbf{a}\_2) \cdot \hat{\mathbf{z}} = \frac{\phi\_0}{B} = \pi l\_B^2 \,, \quad (8)$$

where l<sup>B</sup> ≡ p ~c/eB is the magnetic length. The basic vectors of the corresponding reciprocal lattice are then defined by

$$\begin{cases} \mathbf{b}\_1 \equiv 2(\mathbf{a}\_2 \times \hat{\mathbf{z}}) / l\_B^2 \\\ \mathbf{b}\_2 \equiv 2(\hat{\mathbf{z}} \times \mathbf{a}\_1) / l\_B^2 \end{cases} . \tag{9}$$

We now introduce magnetic Bloch vectors for quasiparticle eigenstates by[34](#page-14-0)

$$\mathbf{k} \equiv \frac{\mu\_1}{\mathcal{N}\_\mathbf{f}} \mathbf{b}\_1 + \frac{\mu\_2}{\mathcal{N}\_\mathbf{f}} \mathbf{b}\_2 \qquad \left( -\frac{\mathcal{N}\_\mathbf{f}}{4} < \mu\_j \le \frac{\mathcal{N}\_\mathbf{f}}{4} \right), \tag{10}$$

and those for the center-of-mass coordinate by

$$\mathbf{q} \equiv \frac{\mu\_1}{\mathcal{N}\_\mathbf{f}} \mathbf{b}\_1 + \frac{\mu\_2}{\mathcal{N}\_\mathbf{f}} \mathbf{b}\_2 \qquad \left( -\frac{\mathcal{N}\_\mathbf{f}}{2} < \mu\_j \le \frac{\mathcal{N}\_\mathbf{f}}{2} \right), \tag{11}$$

where N<sup>f</sup> is an even integer with <sup>N</sup> <sup>2</sup> <sup>f</sup> denoting the number of flux quanta in the system. Notice that q covers an area four times as large as that of k.

#### C. Landau-level-expansion method (LLX)

It has been shown[34](#page-14-0) that the pair potential of the conventional Abrikosov lattice can be expanded in two ways with respect to (r1, r2) and (R, r) ≡ ( r1+r<sup>2</sup> 2 , r1−r2) as

$$\begin{split} \frac{\Delta(\mathbf{r}\_1, \mathbf{r}\_2)}{\Delta(\mathbf{r}\_1, \mathbf{r}\_2)} &= \\ &= \sum\_{\mathbf{k}\alpha} \sum\_{N\_1 N\_2} \Delta\_{N\_1 N\_2}^{(\mathbf{k}p\_z)} \psi\_{N\_1 \mathbf{k}\alpha}(\mathbf{r}\_1) \, \psi\_{N\_2 \mathbf{q} - \mathbf{k}\alpha}(\mathbf{r}\_2) \, \frac{\mathbf{e}^{ip\_z(z\_1 - z\_2)}}{L} \\ &= \frac{\mathcal{N}\_\mathbf{f}}{\sqrt{2}} \sum\_{N\_\mathbf{c}} \sum\_{N\_\mathbf{r} m p\_z} (-1)^{N\_\mathbf{r}} \hat{\Delta}\_{N\_\mathbf{r} p\_z}^{(N\_\mathbf{c} m)} \psi\_{N\_\mathbf{c} \mathbf{q}}^{(\mathbf{c})}(\mathbf{R}) \, \psi\_{N\_\mathbf{r} m}^{(\mathbf{r})}(\mathbf{r}) \frac{\mathbf{e}^{ip\_z z}}{L} . \end{split} \tag{12}$$

Here ψNk<sup>α</sup> is a quasiparticle basis function with N denoting the Landau level, k defined by Eq. (10), and α (=1,2) signifying signifying two-fold degeneracy of every orbital state. On the other hand, ψ (c) Ncq and ψ (r) <sup>N</sup>r<sup>m</sup> are basis functions for the center-of-mass and relative coordinates, respectively, with N<sup>c</sup> and N<sup>r</sup> denoting the corresponding Landau levels, q defined by Eq. (11), and m an eigenvalue for the relative orbital angular momentum operator ˆlz. The quantities p<sup>z</sup> and L are, respectively, the wavenumber and the system length along the z direction parallel to the magnetic field; we adopt a notation of using p as a wavevector in zero field to distinguish it from the two-dimensional magnetic Bloch vector k perpendicular to the field. As noted in Ref. [39,](#page-14-0) an arbitrary single q suffices to describe the conventional Abrikosov lattices due to the broken translational symmetry of the vortex lattice. Then the first expansion of Eq. (12) tells us that, by choosing q = 0, we get an complete analogy with the uniform system in that the pairing occurs between (k, pz) and (−k, −pz). Finally, the two expansion coefficients ∆(kpz) N1N<sup>2</sup> and ∆¯ (Ncm) Nrp<sup>z</sup> are connected by

$$\Delta\_{N\_1 N\_2}^{(\mathbf{k}p\_z)} = \frac{\mathcal{N}\_\mathbf{f}}{2} \sum\_{N\_\mathbf{c} N\_\mathbf{r}} \langle N\_1 N\_2 | N\_\mathbf{c} N\_\mathbf{r} \rangle \sum\_m \langle 2\mathbf{k} - \mathbf{q} | m + N\_\mathbf{r} \rangle$$

$$\times (-1)^{N\_\mathbf{r}} \Delta\_{N\_\mathbf{r} p\_z}^{(N\_\mathbf{c} m)},\qquad(13a)$$

$$\bar{\Delta}\_{N\_\mathbf{r} p\_z}^{(N\_\mathbf{c} m)} = \frac{2}{\mathcal{M}} \sum \left\langle N\_\mathbf{c} N\_\mathbf{r} | N\_1 N\_2 \right\rangle \sum \langle m + N\_\mathbf{r} | 2\mathbf{k} - \mathbf{q} \rangle$$

$$
\Delta\_{N\_r p\_z}^{(N\_c m)} = \frac{2}{\mathcal{N}\_\mathbf{f}} \sum\_{N\_1 N\_2} \langle N\_\mathbf{c} N\_\mathbf{r} | N\_1 N\_2 \rangle \sum\_\mathbf{k} \langle m + N\_\mathbf{r} | 2\mathbf{k} - \mathbf{q} \rangle
$$

$$
\times (-1)^{N\_\mathbf{r}} \Delta\_{N\_1 N\_2}^{(\mathbf{k} p\_z)}, \tag{13b}
$$

where hN1N2|NcNri and h2k−q|m+Nri are elements of unitary matrices for the basis change, i.e. overlap integrals. Their explicit expressions together with those for ψNkα, ψ (c) Ncq , and ψ (r) <sup>N</sup>r<sup>m</sup> are given in Appendix [B.](#page-12-0)

A great advantage of using Eq. (12) is that it enables us to transform Eq. [\(1](#page-1-0)) into a numerically tractable problem, as mentioned before. Indeed, expanding the quasiparticle wavefunctions as

$$\mathbf{u}(\mathbf{r}) = \sum\_{N\mathbf{k}\alpha p\_x} \mathbf{u}\_s(N) \,\psi\_{N\mathbf{k}\alpha}(\mathbf{r}) \,\frac{\mathbf{e}^{ip\_z z}}{\sqrt{L}},\tag{14a}$$

$$\mathbf{v}(\mathbf{r}) = \sum\_{N\mathbf{k}\alpha p\_x} \mathbf{v}\_s(N) \,\psi\_{N\mathbf{q}-\mathbf{k}\alpha}(\mathbf{r}) \,\frac{\mathbf{e}^{-ip\_z z}}{\sqrt{L}},\quad \text{(14b)}$$

Eq. [\(1](#page-1-0)) is reduced to a separate matrix eigenvalue problem for each kαpz, and the eigenstate is labelled by s=νkαpzσ with ν and σ denoting the quasiparticle band and its spin, respectively. Explicitly, Eq.([1\)](#page-1-0) becomes

$$\sum\_{N\_2} \begin{bmatrix} \underbrace{\mathcal{U}\_{N\_1 N\_2}^{(p\_z)}}\_{N\_1 N\_2} & \Delta\_{N\_1 N\_2}^{(\mathbf{k} p\_x)} \\ \Delta\_{N\_1 N\_2}^{(\mathbf{k} p\_z) \dagger} & -\underline{\mathcal{U}}\_{N\_1 N\_2}^{(p\_x)} \end{bmatrix} \begin{bmatrix} \mathbf{u}\_s(N\_2) \\ -\mathbf{v}\_s^\*(N\_2) \end{bmatrix} = E\_s \begin{bmatrix} \mathbf{u}\_s(N\_1) \\ -\mathbf{v}\_s^\*(N\_1) \end{bmatrix},\tag{15}$$

where ∆(kpz) N1N<sup>2</sup> is given by Eq. (13a), and H (pz) N1N2 is diag<span id="page-3-0"></span>onal as

$$\mathcal{H}\_{N\_1 N\_2}^{(p\_z)} = \delta\_{N\_1 N\_2} \left[ (N\_1 + \frac{1}{2}) \hbar \omega\_B + \frac{\hbar^2 p\_z^2}{2m\_e} - \varepsilon\_\mathcal{F} \right] \underline{\mathsf{L}}, \quad (16)$$

with ω<sup>B</sup> ≡eB/M c the cyclotron frequency.

The self-consistency equation([4\)](#page-1-0) are also simplified greatly. Let us define

$$\begin{split} V\_{\mathbf{pp'}} & \equiv \int V(\mathbf{r}) \, \mathrm{e}^{-\mathrm{i}(\mathbf{p}-\mathbf{p'}) \cdot \mathbf{r}} \, \mathrm{d}^3 r \\ &= 4\pi \sum\_{\ell=0}^{\infty} \sum\_{m=-\ell}^{\ell} \bar{V}\_{\ell}(p,p') Y\_{\ell m}(\hat{\mathbf{p}}) Y\_{\ell m}^\*(\hat{\mathbf{p'}}) \,, \end{split} \tag{17}$$

where Yℓm(pˆ) ≡ Θℓm(θp) <sup>√</sup> 1 2π exp(imϕp) is the spherical harmonic.[40](#page-14-0) We also expand both ∆ and Φ in terms of the center-of-mass and relative coordinates as the last line of Eq.([12\)](#page-2-0). Then Eq. [\(4](#page-1-0)) is transformed into an equation for the expansion coefficients of each (Nc, m) as

$$\bar{\Delta}\_{N\_{\rm r}p\_{z}}^{(N\_{c}m)} = \frac{1}{2\pi l\_{B}^{2}L} \sum\_{\ell N\_{\rm r}'p\_{z}'} \bar{V}\_{\ell}(p,p') \Theta\_{\ell m}(\theta\_{\mathbf{p}}) \Theta\_{\ell m}(\theta\_{\mathbf{p}'}) \bar{\underline{\Phi}}\_{N\_{\rm r}'p\_{z}'}^{(N\_{c}m)} \tag{18}$$

with p= p Nr/l<sup>2</sup> <sup>B</sup> +p 2 <sup>z</sup> and θ<sup>p</sup> = tan−<sup>1</sup> √ Nr/l<sup>B</sup> p<sup>z</sup> .

Let us further assume that a single ℓ is relevant in Eq. (17) and take the corresponding V¯ <sup>ℓ</sup>(p, p′ ) in a separable form as

$$
\bar{V}\_{\ell}(p, p') = V\_{\ell} \, W\_{\ell}(\xi) W\_{\ell}(\xi') \,, \tag{19}
$$

where Wℓ(ξ) is some cut-off function with respect to ξ≡ ~ 2p <sup>2</sup>/2me−ε<sup>F</sup> satisfying <sup>W</sup>ℓ(0)= 1. Then can rewrite Eq. (18) as

$$
\bar{\Delta}\_{N\_{\rm r}p\_z}^{(N\_{\rm c}m)} = \tilde{\Delta}^{(N\_{\rm c}m)} W\_{\ell}(\xi) \Theta\_{\ell m}(\theta\_{\mathbf{p}}) \,, \tag{20a}
$$

with ξ=~ 2 (Nr/l<sup>2</sup> <sup>B</sup> +p 2 z )/2me−ε<sup>F</sup> and

$$\underline{\tilde{\Delta}}^{(N\_{c}m)} = \frac{V\_{\ell}}{2\pi l\_{B}^{2}L} \sum\_{N\_{\mathbf{r}}'p\_{x}'} W\_{\ell}(\xi') \Theta\_{\ell m}(\theta\_{\mathbf{p}'}) \,\underline{\underline{\Phi}}\_{N\_{\mathbf{r}}'p\_{x}'}^{(N\_{c}m)} \,. \tag{20b}$$

Thus, we only need a self-consistent solution for a set of discrete parameters {∆˜ (Ncm) (T, B)} through Eqs. [\(15](#page-2-0)) and (20) using Eq. [\(13\)](#page-2-0).

It has been shown[39,](#page-14-0)[41](#page-15-0) that retaining a few Nc's, e.g., N<sup>c</sup> = 0, 6, 12 for the hexagonal lattice, is sufficient to describe the vortex lattices of H &0.05Hc2. Thus, the original problem of obtaining self-consistency for ∆(r1, r2) at all space points is now reduced to the one for a few expansion coefficients {∆˜ (Ncm) (T, B)}. This situation is analogous to the zero-field case where a single parameter ∆0(T ) specifies the pair potential.

The linearized self-consistency equation is obtained by substituting into Eq. (20b) the expression of Φ¯ (Ncm) Nrp<sup>z</sup> linear in the pair potential:[34](#page-14-0)

$$\bar{\underline{\Phi}}\_{N\_{\rm r}p\_{\rm z}}^{(N\_{\rm c}m)} = -\frac{1}{2} \sum\_{N\_1N\_2} \langle N\_{\rm c}N\_{\rm r}|N\_1N\_2 \rangle \frac{\tanh \frac{\xi\_1}{2T} + \tanh \frac{\xi\_2}{2T}}{\xi\_1 + \xi\_2}$$

$$\times \sum\_n (-1)^n \langle N\_1N\_2|N\_{\rm c} + n|N\_{\rm r} - n \rangle \Delta\_{N\_{\rm r} - n\_{\rm p\_z}}^{(N\_{\rm c} + n \, m + n)} . \tag{21}$$

Equation (20) with Eq. (21) determines the mean-field Hc2(T ), i.e. Tc(H). If we use the asymptotic expression [\(B6](#page-13-0)) for the overlap integral hNcNr|N1N2i and replace the sum over N<sup>1</sup> by the integral over x ≡ (N<sup>1</sup> − N2)/ p 2(Nc+Nr), we reproduce the smooth quasiclassical H quasi c2 (T ) obtained, for example, for the s-wave pairing by Helfand and Werthamer.[42](#page-15-0)

Finally, Eq. [\(4](#page-1-0)) for two-dimensional systems can be transformed similarly. It is also obtained from Eqs. (17)- (20) by replacing Vℓ(p, p′ ) → V (m) (p, p′ ), extending the summation over m in Eq. (17) from −∞ to ∞, and finally restricting the summation over ℓ and p<sup>z</sup> only to ℓ= 0 and p<sup>z</sup> = 0, respectively.

#### III. TWO-DIMENSIONAL CALCULATIONS

We first consider a couple of two-dimensional models and perform fully self-consistent calculations. Our purposes in this section are summarized as follows: (i) to clarify the essential features of the results by selfconsistent calculations; (ii) to see whether point nodes in the gap really enhances the dHvA signals as Miyake claims.[19](#page-14-0)

#### A. Models

The one-particle Hamiltonian([2\)](#page-1-0) yields an isotropic Fermi surface specified by a unit vector pˆ = (cosϕp,sin ϕp). As for the pairing interaction (17), we adopt the following models:

$$V\_{\mathbf{pp'}} = \begin{cases} V\_0 \, W(\xi) W(\xi') \\ V\_2 \, W(\xi) (\hat{\mathbf{p}}\_x^2 - \hat{\mathbf{p}}\_y^2) \, W(\xi') (\hat{\mathbf{p}}\_x'^2 - \hat{\mathbf{p}}\_y'^2) \end{cases},\tag{22}$$

where

$$W(\xi) = \exp\left[-\frac{1}{2} \left(\frac{\xi}{\hbar \omega\_{\rm D}}\right)^{4}\right] \tag{23}$$

is a smooth cut-off function with ω<sup>D</sup> a cut-off frequency. The second model of Eq. (22) is beyond the original isotropic interaction (17), but it is convenient for the above-mentioned purposes. In zero field, the two interactions yield the s- and dx2−<sup>y</sup> <sup>2</sup> -wave gaps as

$$\Delta\_{\mathbf{p}} = \begin{cases} \Delta\_0 W(\xi) \, i \underline{\sigma}\_2 & \text{: } s \text{-wave} \\ \Delta\_0 W(\xi) (\hat{\mathbf{p}}\_x^2 - \hat{\mathbf{p}}\_y^2) \, i \underline{\sigma}\_2 & \text{: } d\_{x^2 - y^2} \text{-wave} \end{cases}, \quad (24)$$

<span id="page-4-0"></span>respectively. The corresponding ∆¯ (Ncm) N<sup>r</sup> of Eq. [\(20](#page-3-0)) for Bkzˆ is given by

$$\underline{\Delta}\_{N\_{\rm r}}^{(N\_{\rm c}m)} = \begin{cases} \tilde{\Delta}^{(N\_{\rm c})} W(\xi) \,\delta\_{m0} \, i \underline{\sigma}\_2 \\\\ \tilde{\Delta}^{(N\_{\rm c})} W(\xi) \frac{\delta\_{m2} + \delta\_{m-2}}{2} \, i \underline{\sigma}\_2 \end{cases}, \qquad (25a)$$

with <sup>ξ</sup>≡<sup>~</sup> <sup>2</sup>Nr/2mel 2 <sup>B</sup> −ε<sup>F</sup> and

$$\tilde{\Delta}^{(N\_c)} = \begin{cases} \frac{V\_0}{4\pi l\_B^2} \sum\_{N\_r'} W\_\ell(\xi') \,\bar{\Phi}\_{N\_r'}^{(N\_c, 0)} \\\\ \frac{V\_2}{4\pi l\_B^2} \sum\_{N\_r'} W\_\ell(\xi') \,\frac{\bar{\Phi}\_{N\_r'}^{(N\_c, 2)} + \bar{\Phi}\_{N\_r'}^{(N\_c, -2)}}{2} \end{cases} \text{(25b)}$$

Here we have adopted a normalization for ∆˜ (Nc) different from Eq. [\(20\)](#page-3-0) so that ∆˜ (Nc) acquires a direct correspondence to the maximum gap ∆<sup>0</sup> in Eq.([24\)](#page-3-0); the factor <sup>1</sup> 2 in the second case stems from cos 2ϕ<sup>p</sup> in Eq. [\(24](#page-3-0)).

We have chosen V<sup>ℓ</sup> in Eq.([22\)](#page-3-0) as

$$g\_{\ell} \equiv -N(0)V\_{\ell} = 0.5\ ,\tag{26}$$

where N(0) = me/2π~ 2 is the density of states per spin at the Fermi level. Another important parameter is the zero-temperature coherence length defined by

$$
\xi\_0 \equiv \hbar v\_{\rm F} / \Delta\_0 \,, \tag{27}
$$

with v<sup>F</sup> the Fermi velocity. We have adopted pFξ<sup>0</sup> = 5 for our calculations. The above two quantities fix our models completely; the cut-off ~ω<sup>D</sup> in Eq. [\(23](#page-3-0)) and Tc(B = 0) are calculated using the gap equation.

It should be noted that choosing pFξ<sup>0</sup> also determines the following quantities: (i) the ratio <sup>~</sup>ω<sup>H</sup> quasi c2 /kBTc, where <sup>~</sup>ω<sup>H</sup> quasi c2 is the zero-temperature cyclotron energy at the quasiclassical upper critical field H quasi c2 ; (ii) the number N<sup>F</sup> of the Landau levels below the Fermi level at H quasi c2 . Indeed, using the usual cut-off model <sup>W</sup>(ξ)=θ(~ω<sup>D</sup> − |ξp|) with <sup>θ</sup> the step function, we obtain the following results for the s-wave pairing:

$$\begin{cases} \hbar \omega\_{H\_{c2}^{\text{quasi}}} / k\_{\text{B}} T\_{\text{c}} = 6.28 / p\_{\text{F}} \xi\_{\text{0}} \, , \\\ N\_{\text{F}} \equiv \varepsilon\_{\text{F}} / \hbar \omega\_{H\_{c2}^{\text{quasi}}} = 0.140 (p\_{\text{F}} \xi\_{\text{0}})^2 \, . \end{cases} \tag{28}$$

To reproduce the experimental situation <sup>~</sup>ω<sup>H</sup> quasi c2 /kBT<sup>c</sup> = 1 ∼ 3 within the present model, we should go into the quantum limit pFξ<sup>0</sup> = 6 ∼ 2, but we then have N<sup>F</sup> = 5 ∼ 1. In real materials, however, <sup>~</sup>ω<sup>H</sup> quasi c2 /kBT<sup>c</sup> and pFξ<sup>0</sup> are apparently independent parameters due to effects not covered by the free-particle model such as the energy band structure. Indeed, pFξ<sup>0</sup> is of the order of 30 (NbSe2) or even larger, whereas <sup>~</sup>ω<sup>H</sup> quasi c2 /kBT<sup>c</sup> = 1∼3; see Table [I](#page-11-0) below. Also, N<sup>F</sup> ≫1 for those materials. The failures to describe these situations are among the main difficulties of the free-particle model of Eq.([2\)](#page-1-0).

Motivated by these observations, we also perform another calculations with much more Landau levels below the Fermi level. This is achieved by including the effect of the band dispersion. Specifically, we apply the Onsager-Lifshiz (OL) quantization scheme to H of Eq. ([1\)](#page-1-0), i.e. the procedure which has been very successful for describing the dHvA oscillations in the normal state.[1](#page-14-0),[2](#page-14-0) Given the density of states per spin N(ε) and the average flux density B, the Nth Landau level ε<sup>N</sup> (N = 0, 1, 2, · · ·) is determined by

$$2\left(N+\frac{1}{2}\right)\frac{\hbar^2}{l\_B^2} = 4\pi\hbar^2 \int\_0^{\varepsilon\_N} N(\varepsilon') \,d\varepsilon'.\tag{29}$$

We fix H (pz) N1N<sup>2</sup> of Eq. [\(15](#page-2-0)) in this way assuming it is diagonal. In contrast, we use the same expression for ∆(kpz) N1N<sup>2</sup> of Eq. [\(15\)](#page-2-0) as the free-particle case. Finally, we choose ξ=εNr/2−ε<sup>F</sup> in Eq. (25) based on the consideration of the free-particle model.[34](#page-14-0) With these prescriptions together with the transformation([13\)](#page-2-0), the coupled equations [\(15](#page-2-0)) and (25) are defined unambiguously. We adopt the model density of states:

$$N(\varepsilon) = \frac{m\_{\text{e}}}{2\pi\hbar^2} \left( 1 + \frac{\alpha\Gamma}{\varepsilon^2 + \Gamma^2} \right),\tag{30}$$

and choose the numerical constants (α, Γ)= (2.1, 2.7) and (5.0, 1.0) for the s- and d-wave models of Eq.([24\)](#page-3-0), respectively. We also use Eq. (26) for the pairing interaction and fix ~ω<sup>D</sup> = 0.5ε<sup>F</sup> in Eq. [\(23](#page-3-0)). We thereby obtain <sup>~</sup>ω<sup>H</sup> quasi c2 ≈ kBT<sup>c</sup> at T = 0 and N<sup>F</sup> ∼ 30 at H = H quasi c2 , which describe the experimental situation much better than the free-particle model.

#### B. Numerical procedures

Coupled equations [\(15\)](#page-2-0) and (25) are solved iteratively with the help of the transformation [\(13](#page-2-0)) to obtain selfconsistent {∆˜ (Nc)}'s and the corresponding quasiparticle eigenstates. The hexagonal (square) vortex lattice is assumed for the s-wave (dx2−y<sup>2</sup> -wave) model, as expected theoretically in high magnetic fields[43,44](#page-15-0) and observed recently in La1.83Sr0.17CuO4+δ. [45](#page-15-0) It should be noted, however, that the precise lattice structure is not important for the theory of the dHvA oscillation in superconductors. We set q= 1 2 (b1+b2) in the relevant equations so that a core of the pair potential is located at the origin R=0. [39](#page-14-0) An advantage of this choice is that the corresponding quasiparticle energies have the rotational symmetry of the hexagonal (square) lattice around k=0; other choices would shift the rotation axis from the origin. We then perform calculations of Eqs.([15\)](#page-2-0) and (25) for a set of discrete k's defined by Eq.([10\)](#page-2-0), where N<sup>f</sup> is chosen as a multiple of 12 to include all the high-symmetry points Γ, M, and K (Γ, X, and M) of the hexagonal (square) lattice. Three different values N<sup>f</sup> = 12, 24, 36 are used to see the size dependence, and it has been checked that the results do not differ for the three cases. The hexagonal (square) symmetry enables us to restrict the summation

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

**Caption:** Figure 1 represents the computed upper critical field H_c2 as a function of temperature for s-wave (panel a) and d-wave (panel b) superconductors normalized by the quasiclassical upper critical field. This analysis, using a theoretical model, highlights the distinct temperature dependencies and oscillatory deviations in different superconducting states, thus contributing to an understanding of critical field behavior in various superconducting configurations.


FIG. 1: The upper critical field H<sup>c</sup><sup>2</sup> as a function of T for (a) s-wave and (b) d-wave of Eq.([24\)](#page-3-0). Here pFξ<sup>0</sup> = 5, and H<sup>c</sup><sup>2</sup> is normalized by the quasiclassical upper critical field H quasi c2 (T = 0).

over k into approximately 1/12 (1/8) area of the Brillouin zone. Thus, the calculations can be reduced greatly with due care on the degeneracy of high-symmetry points. Finally, the obtained eigenvalues and eigenstates are substituted into Eq. [\(6](#page-1-0)) to calculate the magnetization by Eq.([7\)](#page-2-0). All the calculations are performed at T = 0.1Tc.

![](_page_5_Figure_4.jpeg)

**Caption:** Figure 2 displays expansion coefficients Δ῀(Nc) as a function of the magnetic field (B) for s-wave (first column) and d-wave superconductors (second column). These coefficients, calculated at T=0.1Tc, provide insight into superconductivity's response under external fields. The dotted lines indicate the square-root behavior predicted by quasiclassical theory. The panels reveal significant oscillatory and singular behaviors that deviate from traditional expectations, emphasizing the influence of field-induced effects on superconducting gaps.


FIG. 2: The expansion coefficients ∆˜ (Nc) in Eq.([25\)](#page-4-0) as a function of B for the s-wave (first column) and the d-wave (second column) with T = 0.1T<sup>c</sup> and pFξ<sup>0</sup> = 5. The dotted lines in the first row signify the square-root behavior expected from the quasiclassical theory.

![](_page_5_Figure_6.jpeg)

**Caption:** Figure 3 presents quasiparticle energy dispersion within a magnetic Brillouin zone for s-wave (left column) and d-wave (right column) lattices at varying B/H quasi c2 ratios. Displayed for B/H quasi c2=0.8, 0.5, and 0.1, these plots reveal substantial dispersion contrasts under the presence of a pair potential. The energy configurations under different field intensities showcase how superconducting energy states are influenced by lattice type and magnetic field.


FIG. 3: Quasiparticle dispersion in the magnetic Brillouin zone for the s-wave hexagonal lattice (first column) and the dwave square lattice (second column). Here pFξ<sup>0</sup> = 5, T = 0.1Tc, and B/Hquasi c2 (0) is equal to 0.8, 0.5, and 0.1 from top to bottom, respectively

#### C. Results

The above self-consistent procedure is known to give rise to oscillatory singular behaviors in both Hc<sup>2</sup> and ∆˜ (Nc) in the field range where the dHvA oscillation persists.[46](#page-15-0),[47](#page-15-0),[48](#page-15-0) Figure 1 displays Hc2(T ) calculated selfconsistently for the s- and d-wave models; it is normalized by the quasiclassical upper critical field H quasi c2 (T = 0). An oscillatory behavior sets in around T . 0.2Tc, and Hc<sup>2</sup> deviates substantially from the smooth Helfand-

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

**Caption:** Figure 4 displays the oscillatory magnetization, M_osc, for both (a) s-wave and (b) d-wave superconductors, considering the range 0.8 ≤ H_{c2}^{quasi}(0)/B ≤ 2.0 (1.2 ≥ B/H_{c2}^{quasi}(0) ≥ 0.5) at T = 0.1T_c with p_Fξ^0 = 5. The dotted lines indicate the magnetization for the normal state. The figure underscores the superconducting state deviations from normal state behaviors under a magnetic field, reflecting on how the oscillatory component of magnetization differs between s-wave and d-wave superconductors when subjected to external magnetic fields. This serves to highlight the impact of superconductivity on magnetization characteristics in the presence of strong magnetic fields.


FIG. 4: Oscillatory part of magnetization Mosc for (a) the swave and (b) the d-wave, over 0.8≤ H quasi c2 (0)/B ≤2.0 (1.2≥ B/Hquasi c2 (0) ≥ 0.5) with T = 0.1T<sup>c</sup> and pFξ<sup>0</sup> = 5. The dotted lines are the curves of the corresponding normal state.

Werthamer behavior[42](#page-15-0) predicted by the quasiclassical theory. The number of the Landau levels below the Fermi level is N<sup>F</sup> ∼ 10 around H quasi c2 (0), which is considerably smaller than those for the real materials. Figure [2](#page-5-0) shows ∆˜ (Nc) as a function of B at T = 0.1T<sup>c</sup> for the s-wave hexagonal lattice (first column) and the dwave square lattice (second column); they are real and finite only for N<sup>c</sup> = 0, 6, 12, · · · (0, 4, 8, · · ·) for the hexagonal (square) lattice,[39](#page-14-0)[,41](#page-15-0) as already mentioned. We observe that ∆˜ (Nc) 's are also singular, and the dominant ∆˜ (0) component cannot be described by the square-root behavior near H quasi c2 (0) expected from the quasiclassical theory. However, those singular behaviors disappear gradually as B decreases.

Figure [3](#page-5-0) displays the quasiparticle energies in the magnetic Brillouin zone for the s-wave hexagonal lattice (first column) and the d-wave square lattice (second column) at B/Hquasi c2 (0)= 0.8, 0.5, and 0.1. At B/Hquasi c2 (0)= 0.8, we already observe large dispersion for E .2∆<sup>0</sup> where the pair potential is effective. In contrast, the flat Landaulevel structure remains for E & 2∆<sup>0</sup> where the pair potential vanishes in the present cut-off model of Eq. [\(23\)](#page-3-0). Thus, the dispersion is caused clearly by the scattering from the growing pair potential, and as will be discussed below, it is the origin of the extra dHvA oscillation damping in the vortex state. We also notice that, for B/Hquasi c2 (0) & 0.5, almost no qualitative difference can be seen between the s- and d-wave cases. At a lower field of B/Hquasi c2 (0)= 0.1, however, a marked difference grows around E . ∆0. The s-wave energy bands of E . 0.7∆<sup>0</sup> are flat and occur in pairs with the level spacing of the order of ∆<sup>2</sup> 0 /εF. As already pointed out by Norman et al., [29](#page-14-0) these corresponds to the bound core states of an isolated vortex with little tunneling probability between adjacent cores. In contrast, the d-wave bands in the same region are densely packed with large dispersion, indicating the extended nature of the corresponding quasiparticle wavefunctions. From this comparison, we conclude that no bound states exist for the d-wave model even in the zero-field limit of an isolated vortex, in agreement with the result of Franz and Teˇsanovi´c.[49](#page-15-0) This difference in the low-energy dispersion at low fields was already reported

![](_page_6_Figure_5.jpeg)

**Caption:** Figure 5 illustrates the magnetic field's impact on the expansion coefficients Δ῀(Nc) within s-wave (first column) and d-wave (second column) superconductors, utilizing non-quadratic dispersion at T=0.1Tc. The dotted lines present expected quasiclassical behavior, highlighting deviation due to field effects. The observations of large oscillatory deviations underscore the critical role of magnetic fields in influencing superconductive properties across different gap symmetries.


FIG. 5: The expansion coefficients ∆˜ (Nc) in Eq.([25\)](#page-4-0) as a function of B for the s-wave (first column) and the d-wave (second column). Here T = 0.1Tc, and the non-quadratic dispersion given by Eq. [\(30\)](#page-4-0) is used. The dotted lines in the first row signify the square-root behavior expected from the quasiclassical theory.

in Ref. [35.](#page-14-0)

Figure 4 shows oscillatory part of the magnetization Mosc calculated numerically by Eq.([7\)](#page-2-0), where curves of the corresponding normal state are also plotted for comparison. The damping starts from above H quasi c2 (0) where ∆˜ (0) is already finite as in Fig. [2,](#page-5-0) and develops rapidly as ∆˜ (0) grows in decreasing B. Thus, the mean-field theory predicts that the dHvA oscillation comes together with the oscillatory singular behaviors in Hc<sup>2</sup> and ∆˜ (Nc) . Combined with the energy dispersion given in Fig. [3,](#page-5-0) we are now able to attribute the origin of the extra damping unambiguously to the Landau-level broadening due to the pair potential. The oscillations are rather irregular in both the s-wave and d-wave cases, in accordance with the singular behaviors of ∆(Nc) in Fig. [2.](#page-5-0) We also see no qualitative difference between the two cases.

However, the free-particle model has several inappropriate points as discussed already around Eq. [\(28](#page-4-0)). For example, the number of Landau levels below the Fermi level N<sup>F</sup> is necessarily N<sup>F</sup> ∼ 10 at H quasi c2 for pFξ<sup>0</sup> = 5, which is much smaller than the values of the materials displaying the dHvA oscillation. Hence the above numerical results may not be sufficient to say anything quantitative about the dHvA attenuation or the differences between the s- and d-wave cases. We have thus

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

**Caption:** Figure 1 illustrates the oscillatory component of magnetization (M_osc) normalized to the upper critical field (H_c2) for s-wave and d-wave superconductors as a function of the ratio H_c2(0)/B. Panel (a) represents s-wave symmetry, while panel (b) corresponds to d-wave symmetry. The plots showcase how the magnetization oscillates as the magnetic field varies, indicating a distinction between s-wave and d-wave states under varying magnetic fields. These findings are critical for understanding the vortex state's contribution to the dHvA effect, displaying that the oscillation damping is not significantly different between s-wave and d-wave types, suggesting that average gap plays the main role.


FIG. 6: Oscillatory part of magnetization Mosc for (a) the s-wave and (b) the d-wave, over 0.8 . H quasi c2 (0)/B ≤ 1.7, i.e. 1.25 & B/Hquasi c2 (0) ≥ 0.59. Here T = 0.1Tc, and a nonquadratic dispersion given by Eq.([30\)](#page-4-0) is used. The dotted lines are the curves of the corresponding normal state.

performed another calculations for the model described around Eqs. [\(29](#page-4-0)) and [\(30](#page-4-0)) where <sup>~</sup>ω<sup>H</sup> quasi c2 /kBT<sup>c</sup> ∼ 1 and quasi (0).

N<sup>F</sup> ∼30 at B =H c2 Figure [5](#page-6-0) shows the field-dependence of the expansion coefficients ∆˜ (Nc) calculated self-consistently for the swave hexagonal lattice (first column) and the d-wave

square lattice (second column). Singular oscillatory behaviors are manifest in both cases as in the case of the quadratic dispersion, which originate from the singular density of states of Landau levels.[48](#page-15-0) For example, the dominant ∆˜ (0) component have a nonzero value from above H quasi c2 and deviates substantially from the quasiclassical square-root behavior (dotted lines).

Figure 6 displays the corresponding oscillatory part of the magnetization Mosc calculated numerically by Eq. ([7\)](#page-2-0), where normal-state results (dotted lines) are also plotted for comparison. The main features are summarized as follows: (i) The oscillations are seen to decrease from above the quasiclassical H quasi c2 , due to the reentrant behavior of ∆˜ (Nc) , to be reduced considerably around B ∼ 0.8H quasi c2 , i.e. H quasi c2 /B ∼ 1.25. However, they do not disappear completely in lower fields. (ii) This extra attenuation is due to the broadening of the Landau levels caused by the pair potential, as in the case of the quadratic dispersion. Indeed, we have obtained quasiparticle spectra similar to those of Fig. [3.](#page-5-0) (iii) The period of the oscillation remains unchanged above ∼0.8H quasi c2 , but some irregularity appears in lower fields. These features are in agreement with the results by Norman et al. [29](#page-14-0) (iv) Little difference can be seen between the s- and d-wave attenuations.

#### D. Summary of Two-Dimensional Calculations

Let us summarize results and conclusions from our twodimensional calculations. (i) Combining Figs. [3](#page-5-0) and [4](#page-6-0), we are now able to attribute the origin of the extra dHvA oscillation damping unambiguously to the Landau-level broadening due to the scattering by the pair potential. (ii) As may be realized from Fig. 6, presence of point

![](_page_7_Figure_9.jpeg)

**Caption:** Figure 7 compares the oscillatory magnetization M_osc for s-wave (panel a) and d-wave (panel b) superconductors under quasiclassical conditions, reflected through dotted line baselines. Here, the graph emphasizes that average gap rather than point nodes primarily determines attenuation levels. Observations indicate that oscillatory damping occurs prominently around H quasi c2, providing pivotal insights into oscillatory magnetization under varying gap conditions.


FIG. 7: Oscillatory part of magnetization Mosc for (a) the s-wave and (b) the d-wave. The difference from Fig. 7 lies in the use of quasiclassical ∆(0)'s given by the dotted lines in Fig. [5](#page-6-0).

nodes along the extremal orbit does not weaken the attenuation, contrary to the statement by Miyake.[19](#page-14-0) This fact suggests that the attenuation is determined by the average gap along the extremal orbit. (iii) The meanfield theory predicts that the dHvA oscillation comes together with the oscillatory behaviors in Hc<sup>2</sup> and ∆˜ (Nc) . This will be so in three dimensional models where Hc2(T ) also shows an oscillatory behavior.[48](#page-15-0) However, such singular behaviors of Hc<sup>2</sup> have never been identified definitely in any materials displaying the dHvA oscillation, and reported Hc<sup>2</sup> curves show more or less the smooth quasiclassical behavior. This discrepancy between the mean-field theory and the dHvA experiments remains a puzzle to be resolved in the future. (iv) The oscillation attenuates considerably around B ∼ 0.8H quasi c2 , i.e. H quasi c2 /B <sup>∼</sup>1.25, although we have set <sup>~</sup>ω<sup>H</sup> quasi c2 /kBT<sup>c</sup> ∼1 and N<sup>F</sup> ≫1 at H quasi c2 . Thus, the two dimensional models fail to explain the experiment by Terashima et al. [9](#page-14-0) which shows a persistent oscillation down to 0.2Hc2. In addition, the models cannot say anything about whether presence of a line node along the extremal orbit weakens the attenuation. (v) The approximation of retaining only ∆˜ (0) works excellently for calculating Mosc. Indeed, we have checked that the curves of Mosc thereby obtained are almost indistinguishable from those of Fig. 6. (vi) The discrepancy mentioned in (iii) above suggests that we should rather use ∆˜ (0) obtained quasiclassically to reproduce the smooth behaviors of Hc<sup>2</sup> in real materials. Figure 7 plots curves of Mosc calculated using quasiclassical ∆˜ (0), i.e. the dotted lines of Fig. [5](#page-6-0). The oscillations are seen more regular than those of Fig. 6, but the amplitudes attenuate almost similarly and are reduced considerably around H quasi c2 /B ∼1.25. We hence realize that using the quasiclassical ∆˜ (0) suffices for the theory of the oscillation damping. This statement is especially true in the low-field region ∼ 0.2H quasi <sup>c</sup><sup>2</sup> where ∆˜ (0) approaches to the quasiclassical behavior, as may be realized from Fig. [5](#page-6-0).
## <span id="page-8-0"></span>IV. THREE-DIMENSIONAL CALCULATIONS

Having clarified basic features of the dHvA oscillation for two-dimensional models as well as the mechanism of the extra oscillation damping, we proceed to consider three-dimensional models with various gap structures which are more relevant to real materials. Our purposes in this section are summarized as follows: (i) to clarify the connection between the extra dHvA oscillation damping and the gap anisotropy by numerical calculations; (ii) to obtain an analytic formula for the extra oscillation damping; (iii) to estimate the gap magnitudes of various materials using the obtained analytic formula.

## A. Model

The one-particle Hamiltonian [\(2](#page-1-0)) yields a spherical Fermi surface in the normal state. As for the pairing interaction, we consider three different models:

$$V\_{\mathbf{pp'}} = \begin{cases} V\_0 \, W(\xi) W(\xi') \\ V\_2 \, W(\xi) (\hat{\mathbf{p}}\_x^2 - \hat{\mathbf{p}}\_y^2) \, W(\xi') (\hat{\mathbf{p}}\_x'^2 - \hat{\mathbf{p}}\_y'^2) \, . \end{cases} \\ (31)$$
 
$$V\_1 \, W(\xi) \, \hat{\mathbf{p}} \cdot \hat{\mathbf{c}} \, W(\xi') \, \hat{\mathbf{p}}' \cdot \hat{\mathbf{c}}$$

Here W(ξ) is a cut-off function given by Eq.([23\)](#page-3-0), pˆ ≡ (sin θ<sup>p</sup> cosϕp,sin θ<sup>p</sup> sin ϕp, cos θp) specifies a point on the Fermi surface, and cˆ≡(sin θc, 0, cos θc) denotes the direction of the crystal c-axis, in the coordinate frame where Bkzˆ. Again the latter two models of Eq. (31) are beyond the original spherical interaction [\(17](#page-3-0)), but they are convenient for the above-mentioned purposes. In zero field, Eq. (31) yield

$$\Delta\_{\mathbf{p}} = \begin{cases} \Delta\_0 W(\xi) \, i \underline{\sigma}\_2 & \text{: } s \text{-wave} \\ \Delta\_0 W(\xi) (\hat{\mathbf{p}}\_x^2 - \hat{\mathbf{p}}\_y^2) \, i \underline{\sigma}\_2 & \text{: } d\_{x^2 - y^2} \text{-wave} \\ \Delta\_0 W(\xi) \, \hat{\mathbf{p}} \cdot \hat{\mathbf{c}} \, i \underline{\sigma}\_3 \underline{\sigma}\_2 & \text{: } p\_z \text{-wave} \end{cases}, \quad (32)$$

which denote the isotropic s-wave state, a threedimensional dx2−<sup>y</sup> <sup>2</sup> -wave state with four point nodes in the extremal orbit perpendicular to B, and the p-wave polar state with a line node perpendicular to cˆ, respectively. The corresponding ∆¯ (Ncm) Nrp<sup>z</sup> in Eq.([18\)](#page-3-0) can be written as

$$\bar{\Delta}\_{N\_{\rm r}p\_{\rm z}}^{(N\_c m)} = \begin{cases} \bar{\Delta}^{(N\_c)} W(\xi) \,\delta\_{m0} \, i \underline{\sigma}\_2 \\\\ \bar{\Delta}^{(N\_c)} W(\xi) \sin^2 \theta\_{\mathbf{p}} \, \frac{\delta\_{m2} + \delta\_{m-2}}{2} \, i \underline{\sigma}\_2 \\\\ \bar{\Delta}^{(N\_c)} W(\xi) \left[ \cos \theta\_{\mathbf{p}} \cos \theta\_{\mathbf{c}} \, \delta\_{m0} \\\\ + \sin \theta\_{\mathbf{p}} \sin \theta\_{\mathbf{c}} \, \frac{\delta\_{m1} + \delta\_{m-1}}{2} \right] i \underline{\sigma}\_3 \underline{\sigma}\_2 \end{cases}, \text{ (33a)}$$

$$
\bar{\Delta}^{(N\_c)} = \begin{cases}
\frac{V\_0}{4\pi l\_B^2} \sum\_{N\_i' p\_z'} W\_\ell(\xi') \,\bar{\Phi}\_{N\_i' p\_z'}^{(N\_c, 0)} \\
\frac{V\_2}{4\pi l\_B^2} \sum\_{N\_i' p\_z'} W\_\ell(\xi') \sin^2 \theta\_{\mathbf{p}'} \,\frac{\bar{\Phi}\_{N\_i' p\_z'}^{(N\_c, 2)} + \bar{\Phi}\_{N\_i' p\_z'}^{(N\_c, -2)}}{2} \\
\frac{V\_1}{4\pi l\_B^2} \sum\_{N\_i' p\_z'} W\_\ell(\xi') \left[\cos \theta\_{\mathbf{p}'} \cos \theta\_\mathbf{c} \,\bar{\Phi}\_{N\_i' p\_z'}^{(N\_c, 0)} \\
\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\}\tag{33}
$$

$$
\left\{ \begin{aligned}
\frac{\bar{\Phi}\_{N\_i' p\_z'}^{(N\_c, 1)}}{2} \frac{\bar{\Phi}\_{N\_i' p\_z'}^{(N\_c, 1)}}{2} \end{aligned} \right. \end{cases} \tag{33b}
$$

Here we have adopted a normalization for ∆˜ (Nc) different from Eq. [\(20](#page-3-0)) so that this quantity acquires a direct correspondence to the maximum gap ∆<sup>0</sup> in Eq. (32). The factors <sup>1</sup> 2 in the second and third cases stem from cos 2ϕ<sup>p</sup> and cosϕ<sup>p</sup> in Eq. (32), respectively.

The coefficients ∆(Nc) = ∆(Nc) (B, T ) in Eq. (33) completely specify the pair potential, as already mentioned. Based on the reasoning given in Sec. [III D\(](#page-7-0)vi), we here adopt a quasiclassical ∆˜ (Nc) rather than the fully selfconsistent one. Then the dominant ∆˜ (0) near Hc<sup>2</sup> follows the mean-field square-root behavior to an excellent approximation:

$$
\tilde{\Delta}^{(0)} = a(1 - B/H\_{c2})^{1/2}.\tag{34}
$$

See the dotted lines in Figs. [2](#page-5-0) and [5](#page-6-0), for example. In addition, other components ∆˜ (Nc>0) can be neglected for the relevant region B & 0.1Hc2, as pointed out in Sec. [III D\(](#page-7-0)v). We hence use the lowest-Landau-level approximation of retaining only ∆˜ (0). The coefficient a=a(T ) in Eq. (34) is determined by requiring that the maximum of

$$\frac{1}{\mathcal{V}} \int d\mathbf{R} \left| \int d\mathbf{r} \,\Delta(\mathbf{r}\_1, \mathbf{r}\_2) \,\mathrm{e}^{-i\mathbf{p}\cdot\mathbf{r}/\hbar} \right|^2 \tag{35}$$

be equal to ∆<sup>2</sup> 0 (1 − B/Hc2), where ∆0(T ) denotes the maximum gap obtained from the weak-coupling theory. This procedure yields

$$a \approx \sqrt{0.5}\,,\tag{36}$$

for all the three cases of Eq. (33). Substituting Eq. (34) into Eq.([13a\)](#page-2-0) with the choice <sup>~</sup>ω<sup>D</sup> <sup>∼</sup> 10∆0(<sup>T</sup> = 0), the off-diagonal elements of Eq. [\(15](#page-2-0)) are fixed completely.

The above non-self-consistent procedure has another advantage that we can choose ~ωB=Hc<sup>2</sup> and ε<sup>F</sup> in Eq. ([16\)](#page-3-0) independently. We have set

$$
\hbar\omega\_{H\_{c2}} = k\_{\rm B}T\_c \quad \text{at } T = 0 \,, \tag{37}
$$

in accordance with <sup>~</sup>ω<sup>H</sup>c2 /kBT<sup>c</sup> = 1 <sup>∼</sup> 3 and <sup>N</sup><sup>F</sup> <sup>≫</sup> 1 for relevant materials (see Table [I](#page-11-0) below). Also, we have

<span id="page-9-0"></span>![](0__page_9_Figure_1.jpeg)

**Caption:** Figure 8 shows the oscillatory magnetization M_osc for the s-wave model (Panel a), illustrating its behavior in the vortex state compared to the normal state. The blue and sky-blue lines depict the vortex-state and normal-state magnetizations, respectively. Panel (b) presents the corresponding Dingle plot (logarithmic plot of resistance) with error bars, with theoretical predictions by Maki, NMA, and other models. This figure emphasizes that the magnetization oscillation persists even at low fields, and theoretical models predict different behaviors in terms of damping under various superconducting conditions.


FIG. 8: (a) The oscillatory part of the magnetization Mosc in the vortex state (blue line) as compared with the normal-state one (sky-blue line) for the s-wave model of Eq. [\(32](#page-8-0)) at T = 0. (b) The corresponding Dingle plot (points with error bars) as compared with various theoretical predictions; see text for details.

chosen ε<sup>F</sup> in such a way that there are about 50 Landau levels below ε<sup>F</sup> for the extremal orbit at Hc2. Now, the matrix elements of Eq.([15\)](#page-2-0) are specified completely. Hexagonal, square, and hexagonal lattices are assumed for the three cases of Eq. [\(33](#page-8-0)), respectively.

## B. Numerical Procedures

The wavevector p<sup>z</sup> of 0≤p<sup>z</sup> ≤1.2p<sup>F</sup> is discretized into ∼ 1000 points with an equal interval. For each of them, we have diagonalized Eq. [\(15](#page-2-0)) with the same procedure as described in Sec. [III B.](#page-4-0) The obtained results are substituted into Eq. [\(6](#page-1-0)) to calculate the magnetization by Eq.([7](#page-2-0)). All the calculations are performed at T = 0.

## C. Numerical Results

We first focus on the dHvA oscillation of the s-wave model in Eq. [\(32](#page-8-0)) to clarify its basic features. Using our numerical data, we also test the applicability of various theoretical formulas presented so far.

Figure 8(a) presents oscillation of the s-wave magnetization (blue line) as compared with the normal-state one (sky-blue line). With ~ωHc<sup>2</sup> =kBTc, the oscillation is observed to persist down to a fairly low field of Hc2/B .1.8, i.e., B & 0.55Hc2, which is smaller than 0.8Hc<sup>2</sup> around which ~ω<sup>B</sup> becomes equal to the spatial average of the energy gap, Eq. [\(35](#page-8-0)). This is partly because the gap is smaller within the extremal orbit, as shown quasiclassically by Brandt et al. [31](#page-14-0) Indeed, Fig. 9 calculated at B = 0.968Hc<sup>2</sup> demonstrates that the dispersion for p<sup>z</sup> = 0 is smaller than that for p<sup>z</sup> = 0.9pF. This tendency remains in the high-field region of B &0.5Hc2.

It has become conventional to express this extra attenuation in the vortex state by introducing an additional factor R<sup>s</sup> for the dHvA oscillation amplitude:

$$R\_{\rm s} = \exp\left(-\frac{\pi}{\omega\_B \tau\_{\rm s}}\right) = \exp\left(-\frac{2\pi^2 k\_{\rm B} T\_{\Delta}}{\hbar \omega\_B}\right),\qquad(38)$$

![](0__page_9_Figure_11.jpeg)

**Caption:** Figure 11 illustrates the dHvA effect in the vortex state for a p-wave model, depicting oscillatory magnetization (M_osc) as a function of magnetic field orientation. The p-wave superconductivity in panels (a) portrays varied tilts of the c-axis relative to the magnetic field, from θ = 0 to π/4. Accompanying Dingle plots on the right offer empirical and theoretical comparisons. This figure distinguishes how directional shifts influence magnetization behavior in complex states, reflecting the nuanced interplay between magnetic orientation and superconducting properties.


FIG. 9: Quasiparticle dispersion in the magnetic Brillouin zone for the s-wave model at B = 0.968H<sup>c</sup>2. (a) pz=0; (b) pz=0.9pF.

where the parameters τ<sup>s</sup> and T<sup>∆</sup> are directly connected with the extra Landau-level broadening Γ<sup>s</sup> in the vortex state as Γ<sup>s</sup> = ~/2τ<sup>s</sup> = πkBT∆. The points with error bars in Fig. 8(b) shows ln R<sup>s</sup> as a function of 1/B, i.e. the Dingle plot, obtained by numerical differentiation. This extra damping at high fields shows the behavior ∝1−B/Hc<sup>2</sup> in the logarithmic scale, but irregularity sets in around 0.55Hc<sup>2</sup> where the oscillation disappears. We attribute this irregularity to the effect of the bound-state formation in the core region.

The lines in Fig. 8(b) are the predictions from various theoretical formulas. Maki's formula[18](#page-14-0) reproduces the correct functional behavior ∝ 1−B/Hc<sup>2</sup> at high fields, but the prefactor is seen too large. The NMA formula,[29](#page-14-0) deduced from the two-dimensional self-consistent numerical results with N<sup>F</sup> ∼ 10 at Hc2, predicts a more rapid attenuation incompatible with our numerical data. One reason for this discrepancy may originate from the fact that their numerical data with N<sup>F</sup> ∼ 10 at Hc<sup>2</sup> are not appropriate for obtaining an analytic formula by fitting. Another may be attributed to the difference in dimensions. Indeed, the dHvA oscillation in three dimensions differs from that in two dimensions on the point that some finite region δp<sup>z</sup> around the extremal orbit is relevant. Most of the Landau levels in the region do not satisfy the particle-hole symmetry with respect to εF, so that the effect of the pair potential becomes smaller than that in two dimensions. Another theory by Dukan and Teˇsanovi´c,[24](#page-14-0) which would predict Rs= 0 in the clean limit of T = 0, is also inconsistent with the data.

The red line in Fig. 8(b) is due to our formula for the extra Dingle temperature:

$$k\_{\rm B}T\_{\Delta} = 0.5 \tilde{\Gamma} \langle |\Delta\_{\rm p}|^2 \rangle\_{\rm eo} \frac{m\_{\rm b}c}{\pi e \hbar} \frac{1 - B/H\_{c2}}{B},\tag{39}$$

which is derived in Appendix [C](#page-13-0) based on the second-order perturbation with respect to the pair potential. Here h|∆p| 2 ieo denotes the average gap along the extremal or-

<span id="page-10-0"></span>![](0__page_10_Figure_1.jpeg)

**Caption:** Figure 10 details the oscillatory magnetization (M_osc) for the d-wave model at zero temperature (T=0). Panel (a) shows the magnetization in the vortex state (blue line) and normal state (sky-blue line), while Panel (b) is a Dingle plot comparing the theoretical predictions with empirical data. This instance indicates that the d-wave superconductors also exhibit substantial oscillatory behavior in the vortex state, and highlights inconsistencies in some existing theoretical predictions.


FIG. 10: (a) The oscillatory part of the magnetization Mosc in the vortex state (blue line) as compared with the normalstate one (sky-blue line) for the d-wave model of Eq.([32\)](#page-8-0) at T = 0. (b) The corresponding Dingle plot (points with error bars) as compared with the theoretical prediction([39\)](#page-9-0).

bit at B = 0, and m<sup>b</sup> is the band mass. The numerical constant 0.5 stems from Eq. [\(36](#page-8-0)), and Γ is a dimension- ˜ less quantity characterizing the Landau-level broadening due to the pair potential. This unknown parameter Γ is ˜ determined by a best fit to the s-wave numerical data, i.e. the points with error bars in Fig. [8\(](#page-9-0)b). This procedure yields

Γ= 0 ˜ .125 .

We observe in Fig. [8](#page-9-0)(b) that Eq. [\(39](#page-9-0)), which predicts the dependence ∝ 1−B/Hc<sup>2</sup> for ln Rs, agrees with the numerical results. This formula will be seen below to reproduce other numerical data excellently without any adjustable parameters.

A difference of Eq.([39](#page-9-0)) from Maki's formula[18](#page-14-0) lies in the prefactor where the Fermi velocity v<sup>F</sup> is absent. Indeed, a dimensional analysis on the second-order perturbation tells us that the Landau-level broadening in the vortex state should be of order <sup>|</sup>∆˜(0)(B)<sup>|</sup> <sup>2</sup>/~ωB, where ∆˜(0)(B) <sup>∝</sup> p h|∆p| <sup>2</sup>ieo(1−B/Hc2) is essentially the average gap along the extremal orbit. This leads to Eq. [\(39](#page-9-0)) except for the numerical constant.

We now turn our attention to see how the presence of point nodes affect the dHvA oscillation. Figure 10(a) shows the oscillation of the d-wave magnetization (blue line) as compared with the normal-state one (sky-blue line). Although the d-wave gap in Eq. [\(32](#page-8-0)) has four point nodes on the Fermi surface along the extremal orbit, the damping is seen strong and not much different from the s-wave case. From this fact, we may conclude that it is the average gap along the extremal orbit which is relevant for the extra dHvA oscillation damping. Figure 10(b) presents the corresponding Dingle plot (points with error bars), which is compared with the prediction of Eq. [\(39](#page-9-0)). The formula with the average gap h|∆p| 2 ieo reproduces the numerical result for Hc2/B . 1.8 excellently without adjustable parameters, thereby providing a strong support for the above statement.

This d-wave result is in disagreement with Miyake's theory that point nodes in the extremal orbit should weaken the attenuation.[19](#page-14-0) Indeed, Miyake's theory is based on a semiclassical quantization for the expression

![](0__page_10_Figure_9.jpeg)

**Caption:** Figure 11 illustrates the oscillatory part of the magnetization M_osc for a p-wave model, with the left panels showing the vortex state against normal state (indicated by sky-blue lines) while tilting the crystal c-axis by θ_c = 0, π/6, and π/4, respectively. The right panels provide Dingle plots to compare with theoretical models, contingent upon field tilting. This provides valuable insights into how directional changes of magnetic fields in relation to the crystal axis affect vortex state magnetization, emphasizing the role of nodal plane alignment in determining superconducting characteristics and oscillatory damping.


FIG. 11: Left figures: the oscillatory part of the magnetization Mosc in the vortex state (blue lines) as compared with the normal-state one (sky-blue lines) for the p-wave model of Eq.([32\)](#page-8-0) at T = 0. The crystal c axis, which is perpendicular to the nodal plane, is tilted from the field B by θ<sup>c</sup> = 0 (top), θ<sup>c</sup> =π/6 (second), and θ<sup>c</sup> =π/4 (bottom). Right figures: the corresponding Dingle plots (points with error bars) compared with Eq. [\(39](#page-9-0)).

of the electron number N<sup>e</sup> at B = 0. Neither his starting point Ne(B = 0) nor the use of the semiclassical quantization may be justified for describing the dHvA oscillation observed mainly near Hc2.

We finally consider the p-wave model with a line node in Eq. [\(32](#page-8-0)) to double-check the applicability of Eq. [\(39](#page-9-0)). The left figures in Fig. 11 display the dHvA oscillation for the line-node model, where the crystal c-axis is tilted from the magnetic-field direction by θ<sup>c</sup> = 0 (top), θ<sup>c</sup> =π/6 (second), and θ<sup>c</sup> = π/4 (bottom). The damping is seen weakest in the top figure where the gap vanishes along the extremal orbit, but increases gradually as finite gap opens along the orbit for θ<sup>c</sup> = 0 → π/4. These results indicate conclusively that the average gap along the extremal orbit is relevant for the extra dHvA attenuation. However, the non-zero extra damping in the top figure implies that not only the extremal orbit alone but some finite region around it contributes to the extra damping. Theoretically, this corresponds to the fact that we have to perform the Fresnel integral R <sup>∞</sup> −∞exp[−i( √ 2πN<sup>F</sup> pz/pF) 2 ] dp<sup>z</sup> for obtaining the LK formula in the normal state. Our data show that this off-extremal-orbit contribution cannot be neglected in the case where the gap vanishes exactly

<span id="page-11-0"></span>TABLE I: Parameters characterizing three superconductors displaying the dHvA oscillation in the vortex state, together with values of the average gap along the extremal orbit p h|∆p| <sup>2</sup>ieo estimated by Eq. [\(39](#page-9-0)). Here the symbol α and γ are band indices. These values of p h|∆p| <sup>2</sup>ieo are to be compared with the band- and/or angle-averaged quantity ∆(0) extracted from a specific-heat experiment,[36](#page-14-0) a far-infrared measurement,[51](#page-15-0) a tunneling experiment,[52](#page-15-0) or a Raman-scattering experiment.[53](#page-15-0)

| Compound | Tc (K) | Hc2(T = 0) (T)                 | mb/me          | ~ωHc2<br>(K) | ~ωHc2<br>/kBTc | p<br>h ∆p <br>2ieo<br>(meV) | ∆(0) (meV) |
|----------|--------|--------------------------------|----------------|--------------|----------------|-----------------------------|------------|
| NbSe2    | 7.216  | ◦<br>16<br>8.01 (θ = 68.6<br>) | 16<br>0.61 (α) | 17.616       | 2.2016         | 1.1 (±0.04)                 | 1.151,52   |
| Nb3Sn    | 18.37  | 7<br>19.7 (H kc)               | 7<br>1.10 (γ)  | 24.17        | 1.317          | 3.2 (±0.19)                 | 3.253      |
| YNi2B2C  | 14.59  | 9<br>8.8 (H kc)                | 9<br>0.35 (α)  | 33.89        | 2.339          | 1.5 (±0.28)                 | 2.536      |

and completely at the extremal orbit. However, this offextremal contribution is expected to become less important where finite gap is present along the extremal orbit. The right figures in Fig. [11](#page-10-0) show the corresponding Dingle plot (points with error bars), which is compared with the prediction of Eq. [\(39\)](#page-9-0). Except for the weak damping of θ<sup>c</sup> = 0 due to the off-extremal-orbit contribution, the formula is observed to reproduce the numerical results excellently.

## V. ESTIMATION OF ENERGY GAP

Our calculations in Sec. [IV C](#page-9-0) have clarified that (i) the gap anisotropy can be detected by measuring the extra dHvA oscillation damping in the vortex state, and (ii) Eq. [\(39](#page-9-0)) is particularly useful for this purpose. Using the formula, we finally provide quantitative estimations of the average gap along the extremal orbit for several materials displaying the dHvA oscillation in the vortex state. Table I summarizes parameters describing three relevant materials. These materials commonly have fairly high Tc's, and the ratio ~ω<sup>H</sup>c<sup>2</sup> /kBT<sup>c</sup> ranges from 1 to 3. These features seem to be basic conditions for observing the dHvA oscillation in the vortex state. The values for p h|∆p| <sup>2</sup>ieo are obtained by applying Eq.([39\)](#page-9-0) to the observed dHvA attenuation in the vortex state. In doing so, we have adopted as m<sup>b</sup> in Eq.([39\)](#page-9-0) the values from dHvA experiments rather than those from band calculations, as indicated by the theory of Luttinger.[50](#page-15-0) For comparison, we have also listed the values ∆(0) estimated by a specific-heat experiment,[36](#page-14-0) a far-infrared measurement,[51](#page-15-0) a tunneling experiment,[52](#page-15-0) or a Ramanscattering experiment.[53](#page-15-0) Thus, ∆(0) is expected to represent band- and/or angle-averaged energy gap. As seen in Table I, the two quantities coincide excellently for NbSe<sup>2</sup> and Nb3Sn, indicating uniformly opened gap in these materials. On the other hand, p h|∆p| <sup>2</sup>ieo = 1.5 for the α band of YNi2B2C is considerably smaller than ∆(0)= 2.5 from a specific-heat experiment.[36](#page-14-0) This fact implies that YNi2B2C have large band- and/or angle-dependent gap anisotropy. Indeed, Bintley et al. [55](#page-15-0) have recently carried out a detailed dHvA experiment on this material, rotating the field direction and observing the extra attenuation. They have reported a large angle dependence of the attenuation magnitude. They have also pointed out that their result is in agreement with the model with point nodes presented by Izawa et al. [37](#page-14-0) based on a thermalconductivity measurement.

## VI. SUMMARY

We have carried out the first three-dimensional numerical calculations on the dHvA oscillation in the vortex state for various gap structures. We have thereby clarified the relation between gap anisotropy and persistence of the oscillation. We have also derived an analytic formula for the extra dHvA attenuation in the vortex state.

Our main results are given by Figs. [8](#page-9-0)-[11](#page-10-0) and Eq. [\(39](#page-9-0)). Those figures indicate clearly that the extra dHvA attenuation in the vortex state is directly connected with the average gap along the extremal orbit at B = 0. The derived formula [\(39](#page-9-0)) have been shown to reproduce the numerical results excellently. Our theory attributes the origin of the extra dHvA damping to the Landau-level broadening caused by the pair potential. Hence the periodicity of the vortex lattice assumed here is almost irrelevant, and the theory is applicable also to the cases with irregularity such as a random array of vortices. Using Eq. ([39\)](#page-9-0), we have estimated average gap amplitudes along the extremal orbit for NbSe2, Nb3Sn, and YNi2B2C. The results indicate presence of large gap anisotropy in YNi2B2C.

Thus, we have shown explicitly that the dHvA effect in the vortex state can be a powerful tool to probe the average gap along the extremal orbit. Our results imply that, by rotating the field direction and observing the attenuation amplitude, we can obtain unique information on the band- and/or angle-dependent gap structure. Such an experiment has recently been performed on UPd2Al<sup>3</sup> by Inada et al. [13](#page-14-0) and on YNi2B2C by Bintley et al., [55](#page-15-0) and the latter group indeed has detected large gap anisotropy in the ab plane. Equation [\(39](#page-9-0)) will be useful in similar experiments for estimating band- and/or angle-dependent gap amplitudes.

## Acknowledgments

We are grateful to F. J. Ohkawa for enlightening discussions. This research is supported by Grant-in-Aid for Scientific Research from the Ministry of Education, Culture, Sports, Science, and Technology of Japan.

## <span id="page-12-0"></span>APPENDIX A: THERMODYNAMIC POTENTIAL

The Luttinger-Ward thermodynamic potential corresponding to Eq.([1\)](#page-1-0) is given by[56](#page-15-0)

$$\Omega = -\frac{k\_{\rm B}T}{2} \sum\_{n} \text{Tr} \ln \begin{bmatrix} \mathcal{H} - i\varepsilon\_{n} & \Delta \\ \Delta^{\dagger} & -\mathcal{H}^{\*} - i\varepsilon\_{n} \end{bmatrix}$$

$$\times \begin{bmatrix} \mathbf{e}^{i\varepsilon\_{n}0\_{+}} & 0 \\ 0 & \mathbf{e}^{-i\varepsilon\_{n}0\_{+}} \end{bmatrix} - \frac{1}{2} \text{Tr} \, \Delta^{\dagger} \Phi \,, \qquad (\text{A1})$$

where we have adopted a compact notation of using x ≡ rσ with ∆σ1σ<sup>2</sup> (r1, r2) → ∆(x1, x2), etc., and Tr also implies both integration and summation over r and σ, respectively. The quantity εn/~ denotes the Matsubara frequency, and 0<sup>+</sup> is an infinitesimal positive constant. Now, Eq. [\(1](#page-1-0)) tells us that the first matrix in Eq. (A1) can be diagonalized as[56](#page-15-0)

$$\begin{aligned} & \left[ \begin{array}{cc} \mathcal{H}(x, x') & \Delta(x, x') \\ \Delta^{\dagger}(x, x') & -\mathcal{H}^{\*}(x, x') \end{array} \right] \\ &= \sum\_{s} \left[ \begin{array}{cc} u\_{s}^{\*}(x) & -v\_{s}(x) \\ v\_{s}^{\*}(x) & -u\_{s}(x) \end{array} \right] \left[ \begin{array}{cc} E\_{s} & 0 \\ 0 & -E\_{s} \end{array} \right] \left[ \begin{array}{cc} u\_{s}(x') & v\_{s}(x') \\ -v\_{s}^{\*}(x') & -u\_{s}^{\*}(x') \end{array} \right] . \end{aligned} \tag{A2}$$

Substituting Eq. (A2) into Eq. (A1), the first term on the right-hand side becomes

$$\begin{split}-\frac{k\_{\rm B}T}{2}\sum\_{ns}\int \mathrm{d}x\left\{\left[|u\_{s}(x)|^{2}\mathrm{e}^{z\_{n}\,0\_{+}}+|v\_{s}(x)|^{2}\mathrm{e}^{-z\_{n}\,0\_{+}}\right]\right.\\ \qquad\times\ln(E\_{s}-z\_{n})\\ +\left[|v\_{s}(x)|^{2}\mathrm{e}^{z\_{n}\,0\_{+}}+|u\_{s}(x)|^{2}\mathrm{e}^{-z\_{n}\,0\_{+}}\right]\ln(-E\_{s}-z\_{n})\Big\},\end{split} \tag{A3}$$

with z<sup>n</sup> ≡ iεn. The summation over n are then transformed with a standard technique[57](#page-15-0) into a contour integral just above and below the real axis, using f(z) ≡ (ez/kB<sup>T</sup> + 1)−<sup>1</sup> and <sup>f</sup>(−z) for the terms with e<sup>z</sup>n0<sup>+</sup> and e <sup>−</sup>zn0<sup>+</sup> , respectively. Considering the poles inside the two contours and using R [|us(x)| <sup>2</sup>+|vs(x)<sup>|</sup> 2 ]dx= 1, we obtain Eq.([6\)](#page-1-0).

# APPENDIX B: BASIS FUNCTIONS AND OVERLAP INTEGRALS

We here present explicit expressions for the quantities appearing in Eqs. [\(12](#page-2-0)) and [\(13](#page-2-0)); see Ref. [34](#page-14-0) for their detailed derivations. It should be noted that we here adopt the symmetric gauge([3\)](#page-1-0) which is more convenient than the Landau gauge used in Ref. [34.](#page-14-0) Hence there is an extra factor due to the gauge transformation in every expression of the basis functions, such as e−ixy/2<sup>l</sup> 2 <sup>B</sup> in Eq. (B1) below.

The basis function ψNk<sup>α</sup> (N = 0, 1, 2, · · · ; α = 1, 2) in

Eq. [\(12](#page-2-0)) is defined by

$$\psi\_{N\mathbf{k}\alpha}(\mathbf{r}) = \sum\_{n=-\mathcal{N}\_l/2+1}^{N\_l/2} \mathbf{e}^{i[k\_y(y+l\_B^2k\_x/2)+na\_{1x}(y+l\_B^2k\_x-na\_{1y}/2)/l\_B^2]},$$

$$\times \mathbf{e}^{-ixy/2l\_B^2-(x-l\_B^2k\_y-na\_{1x})^2/2l\_B^2+i(\alpha-1)n\pi}$$

$$\times \sqrt{\frac{a\_{1x}/l\_B}{2^N N! \sqrt{\pi} \mathcal{S}}} \, H\_N^\*(\frac{x-l\_B^2k\_y-na\_{1x}}{l\_B}),\,\,(\mathbf{B}1)^2$$

where S ≡ πl<sup>2</sup> <sup>B</sup><sup>N</sup> <sup>2</sup> f , and H<sup>N</sup> (x) ≡ e x 2 <sup>−</sup> <sup>d</sup> dx N e −x 2 is the Hermite polynomial.[58](#page-15-0) The basis function ψ (c) Nq for the center-of-mass coordinates is obtained from Eq. (B1) by putting k→q, α= 1, and l<sup>B</sup> →l<sup>c</sup> ≡lB/ √ 2 as

$$\psi\_{N\mathbf{q}}^{(c)}(\mathbf{r}) = \sum\_{n=-\mathcal{N}\_{l}/2+1}^{N\_{l}/2} \mathbf{e}^{i[q\_{y}\left(y+l\_{c}^{2}q\_{x}/2\right)+na\_{1x}\left(y+l\_{c}^{2}q\_{x}-na\_{1y}/2\right)/l\_{c}^{2}]} \times \\ \times \mathbf{e}^{-ixy/2l\_{c}^{2}-(x-l\_{c}^{2}q\_{y}-na\_{1x})^{2}/2l\_{c}^{2}} \\ \times \sqrt{\frac{a\_{1x}/l\_{c}}{2^{N}N!\sqrt{\pi}\mathcal{S}}} \, H\_{N}\bigg(\frac{x-l\_{c}^{2}q\_{y}-na\_{1x}}{l\_{c}}\bigg) . \quad \text{(B2)} $$

Finally, the basis function ψNm for the relative coordinates, which is conveniently chosen as an eigenstate of the orbital angular momentum operator ˆlz, is given by

$$
\psi\_{Nm}^{\{\mathbf{r}\}}(\mathbf{r}) = \frac{(-1)^N}{\sqrt{2\pi}} \sqrt{\frac{N!}{(N+m)!}} \zeta^m \mathbf{e}^{-|\zeta|^2/2} L\_N^{\{m\}}(|\zeta|^2) \,, \tag{\text{B3}}
$$

where ζ ≡ (x+iy)/ √ 2l<sup>r</sup> with l<sup>r</sup> ≡ √ 2lB, and L (m) <sup>N</sup> (x) ≡ 1 N! e xx −m d dx N e <sup>−</sup><sup>x</sup>x <sup>N</sup>+<sup>m</sup> is the generalized Laguerre polynomial satisfying <sup>N</sup> <sup>+</sup> <sup>m</sup> <sup>≥</sup> 0.[58](#page-15-0)

We next provide expressions of the overlap integrals in Eq.([13\)](#page-2-0). The first one, which was obtaind by Rajagopal and Ryan,[59](#page-15-0) is given by

$$\begin{aligned} \langle N\_{\mathrm{c}}N\_{\mathrm{r}}|N\_{1}N\_{2}\rangle\\ \equiv \delta\_{N\_{1}+N\_{2},N\_{\mathrm{c}}+N\_{\mathrm{r}}} \sqrt{\frac{N\_{1}!N\_{2}!N\_{\mathrm{c}}!N\_{\mathrm{r}}!}{2^{N\_{1}+N\_{2}}}}\\ \times \sum\_{n=\max(0,N\_{\mathrm{c}}-N\_{2})}^{\min(N\_{1},N\_{\mathrm{c}})} \frac{(-1)^{N\_{2}-N\_{\mathrm{c}}+n}}{n!\left(N\_{1}-n\right)!\left(N\_{\mathrm{c}}-n\right)!\left(N\_{2}-N\_{\mathrm{c}}+n\right)!},\end{aligned} \tag{B4}$$

which satisfies

$$
\left< N\_{\rm c} N\_{\rm r} | N\_1 N\_2 \right> = (-1)^{N\_{\rm r}} \left< N\_{\rm c} N\_{\rm r} | N\_2 N\_1 \right> , \tag{B5a}
$$

$$\sum\_{N\_1 N\_2} \left< N\_{\rm c}' N\_{\rm r}' | N\_1 N\_2 \right> \left< N\_1 N\_2 | N\_{\rm c} N\_{\rm r} \right> = \delta\_{N\_{\rm c}' N\_{\rm c}} \delta\_{N\_{\rm r}' N\_{\rm r}} \ . \tag{B5b}$$

<span id="page-13-0"></span>The asymptotic expression of Eq.([B4\)](#page-12-0) for N<sup>c</sup> ≪ N1,N<sup>2</sup> is given by[34](#page-14-0)

$$
\langle N\_{\rm c}N\_{\rm r}|N\_1N\_2\rangle \approx \delta\_{N\_1+N\_2, N\_c+N\_{\rm r}} \left(-1\right)^{N\_2} \left[\frac{2}{\pi (N\_{\rm c}+N\_{\rm r})}\right]^{1/4}
$$

$$
\times e^{-x^2/2} \frac{H\_{N\_{\rm c}}(x)}{\sqrt{2^{N\_{\rm c}}N\_{\rm c}!}},\tag{B6}
$$

with x≡ (N1−N2)/ p 2(Nc+Nr). This expression forms the basis for quasiclassical approximations.

The second overlap integral is given by

$$
\langle \mathbf{q} | m + N\_{\mathbf{r}} \rangle = \sqrt{2\pi} \, l\_{\mathbf{r}} \, \left( -1 \right)^{m + N\_{\mathbf{r}}} \left[ \psi\_{m + N\_{\mathbf{r}} \mathbf{q}}^{\left( \mathbf{r} \right)} \left( \mathbf{0} \right) \right]^{\*}, \qquad \text{(B7)}
$$

where ψ (r) m+Nrq is another basis function of the relative coordinates defined by

$$\psi\_{N\mathbf{q}}^{\left(\mathbf{r}\right)}(\mathbf{r}) = \sum\_{n=-\mathcal{N}\_{\mathbf{r}}/4+1}^{N\_{\mathbf{r}}/4} \mathbf{e}^{i\left[q\_{\mathbf{p}}\left(y+l\_{\mathbf{r}}^{2}q\_{\mathbf{r}}/4\right)/2+2na\_{1x}\left(y+l\_{\mathbf{r}}^{2}q\_{\mathbf{r}}/2-na\_{1y}\right)/l\_{\mathbf{r}}^{2}\right]}{\times \mathbf{e}^{-ixy/2l\_{\mathbf{r}}^{2}-(x-l\_{\mathbf{r}}^{2}q\_{\mathbf{p}}/2-2na\_{1x})^{2}/2l\_{\mathbf{r}}^{2}}$$

$$\times \sqrt{\frac{2a\_{1x}/l\_{\mathbf{r}}}{2^{N}N!\sqrt{\pi}\mathcal{S}}}H\_{N}\bigg(\frac{x-l\_{\mathbf{r}}^{2}q\_{\mathbf{p}}/2-2na\_{1x}}{l\_{\mathbf{r}}}\bigg). \tag{B8}$$

# APPENDIX C: ANALYTIC FORMULA FOR DHVA OSCILLATION DAMPING

In order to derive an analytic expression for the dHvA oscillation damping in the vortex states, we start from the thermodynamic potential of Eq.([6\)](#page-1-0). The last term − 1 <sup>2</sup>Tr∆ Φ may be expressed solely with respect to the pair potential, so that they can be neglected in the present model to consider the oscillatory part. Since we are interested in the extra damping in the vortex state, we adopt as E<sup>s</sup> and v<sup>s</sup> the expressions from the secondorder perturbation with respect to ∆. They are obtained as

$$E\_{N\mathbf{k}\alpha p\_z\sigma} = |\xi\_{Np\_z\sigma}| + \eta\_{N\mathbf{k}p\_z}^{(1)}\operatorname{sign}(\xi\_{Np\_z\sigma})\,,\tag{C1}$$

$$\int |\mathbf{v}\_{N\mathbf{k}\alpha p\_z \sigma}(\mathbf{r})|^2 d\mathbf{r} = \theta(-\xi\_{Np\_z \sigma}) + \eta\_{N\mathbf{k}p\_z}^{(2)} \text{sign}(\xi\_{Np\_z \sigma})\,,\tag{C2}$$

where θ is the step function, and η (n) Nkp<sup>z</sup> is defined by using Eq. [\(13a](#page-2-0)) as

$$\eta\_{N\mathbf{k}p\_x}^{(n)} \equiv \sum\_{N'} \frac{|\Delta\_{NN'}^{(\mathbf{k}p\_x)}|^2}{(\xi\_{Np\_x} + \xi\_{N'-p\_x})^n}.\tag{C3}$$

The first terms on the right-hand side of Eqs. (C1) and (C2) are just the normal-state results. The second terms, on the other hand, denote the finite quasiparticle dispersion in the magnetic Brillouin zone and the smearing of the Fermi surface, respectively, due to the scattering by the growing pair potential. It is useful to express η (n) Nkp<sup>z</sup> in terms of ∆˜(0)(B) and the cyclotron energy ~ω<sup>B</sup> of the extremal orbit as

$$\eta\_{N\mathbf{k}p\_z}^{(n)} = \frac{|\tilde{\Delta}^{(0)}(B)|^2}{(\hbar \omega\_B)^n} \tilde{\eta}\_{N\mathbf{k}p\_z}^{(n)} \,. \tag{C4}$$

The quantity ˜η (n) Nkp<sup>z</sup> thus defined is dimensionless, and we realize that the main B dependence in Eq. (C4) lies in the prefactor <sup>|</sup>∆˜(0)(B)<sup>|</sup> <sup>2</sup>/(~ωB) <sup>n</sup>. The explicit expression of ˜η (n) Nkp<sup>z</sup> is obtained by using Eqs.([13a](#page-2-0)) and [\(33](#page-8-0)). Considering the case θ<sup>c</sup> = 0, for simplicity, it is given by

$$\begin{split} \tilde{\eta}^{(n)}\_{N\mathbf{k}p\_z} &= \frac{\mathcal{N}\_{\mathbf{f}}^2}{4} \sum\_{N'mm'} \frac{|\langle NN'|0 \, N + N'\rangle|^2 \langle N + N'+m|2\mathbf{k} - \mathbf{q}\rangle}{[N+N'-2(N\_{\mathbf{F}}+\delta)]^n} \\ &\times \langle 2\mathbf{k} - \mathbf{q}|N + N'+m'\rangle \times \begin{cases} \delta\_{m0}\delta\_{m'0} \\ \delta\_{m,\pm2}\delta\_{m',\pm2}\sin^4\theta\_{\mathbf{p}} \\ \delta\_{m0}\delta\_{m'0}\cos^2\theta\_{\mathbf{p}} \end{cases}, \end{split} \tag{C5}$$

where the quantity δ = δ(B, pz) (|δ| < 1/2) specifies the location of ε<sup>F</sup> between the two closest Landau levels, and the overlap integrals are defined by Eqs.([B4\)](#page-12-0) and (B7). The corresponding normalized density of states:

$$D\_{Np\_z}^{(n)}(\tilde{\eta}) \equiv \frac{2}{\mathcal{N}\_{\rm f}^2} \sum\_{\mathbf{k}\alpha} \delta(\,\tilde{\eta} - \,\tilde{\eta}\_{N\mathbf{k}p\_z}^{(n)})\,,\tag{C6}$$

will play a central role in the following.

Substituting Eqs. (C1) and (C2) into Eq. [\(6](#page-1-0)), we find that the terms containing η (2) <sup>N</sup>kp<sup>z</sup> may be neglected due to the cancellation between the particle and hole contributions. The remaining term can be transformed with the standard procedure.[1](#page-14-0) We thereby obtain, for the first harmonic of Ω/V , the expression:

$$\frac{\Omega\_1}{V} = -\frac{k\_\mathrm{B}T}{2\pi^2 l\_B^2} \sum\_{\sigma} \int\_{-1/2}^\infty dN \cos(2\pi N) \int\_{-\infty}^\infty dp\_z \int\_{-\infty}^\infty d\tilde{\eta}$$

$$\times D\_{Np\_z}^{(1)}(\tilde{\eta}) \ln\left[1 + e^{-\left(\xi\_{Np\_z\sigma} + \tilde{\eta}\left|\tilde{\Delta}^{(0)}(B)\right|^2/\hbar\omega\_B\right)/k\_B T}\right]. \quad \text{(C7)}$$

The function D (1) Np<sup>z</sup> (˜η) depends on (N, pz), but may be replaced by a representative one D (1) ℓ (˜η) to be placed outside the N and p<sup>z</sup> integrals,[60](#page-15-0) where the recovered index ℓ specifies the s-, d-, or p-wave case of Eq. (C5). It may also be acceptable to use a Lorenzian for it: D (1) ℓ (˜η) = Γ˜ <sup>ℓ</sup>/π(˜η <sup>2</sup>+Γ˜<sup>2</sup> ℓ ).[61](#page-15-0) We thereby obtain an expression for the magnetization which carries an extra damping factor:

$$\begin{split} R\_{\mathsf{s}}(B) & \equiv \int\_{-\infty}^{\infty} \overline{D}\_{\ell}^{(1)}(\bar{\eta}) \exp\left[ -2\pi i \tilde{\eta} |\bar{\Delta}^{(0)}(B)|^{2} / (\hbar \omega\_{B})^{2} \right] d\bar{\eta} \\ &= \exp\left[ -2\pi \tilde{\Gamma}\_{\ell} |\bar{\Delta}^{(0)}(B)|^{2} / (\hbar \omega\_{B})^{2} \right]. \end{split} \tag{C8}$$

<span id="page-14-0"></span>Thus, the superconductivity gives rise to an extra Dingle temperature of <sup>k</sup>BT<sup>∆</sup> <sup>≡</sup> Γ˜ <sup>ℓ</sup> <sup>|</sup>∆˜(0)(B)<sup>|</sup> <sup>2</sup>/π~ωB, or equivalently, the extra scattering rate of τ −1 <sup>s</sup> <sup>≡</sup>2πkBT∆/~.

Equation [\(C8](#page-13-0)) has an advantage that one can trace the origin of the extra dHvA damping definitely to the growing pair potential, which brings finite quasiparticle dispersion in the magnetic Brillouin zone as Eq. [\(C5](#page-13-0)), and the corresponding Landau-level broadening as Eq. ([C6](#page-13-0)). Moreover, Eq.([C5](#page-13-0)) reveals that this broadening near Hc<sup>2</sup> is closely connected with the zero-field gap structure given by Eq.([32\)](#page-8-0).

- <sup>∗</sup> URL: [http://phys.sci.hokudai.ac.jp/~kita/index-e.](http://phys.sci.hokudai.ac.jp/~kita/index-e.html) h[tml](http://phys.sci.hokudai.ac.jp/~kita/index-e.html); Electronic address: [kita@phys.sci.hokudai.ac.jp](mailto:kita@phys.sci.hokudai.ac.jp)
- 1 I. M. Lifshitz and A. M. Kosevich: J. Exp. Theor. Phys. 29 (1955) 730 [Sov. Phys. JETP 2 (1956) 636].
- <sup>2</sup> D. Shoenberg: Magnetic Oscillations in Metals (Cambridge University Press, Cambridge, 1984).
- 3 J. E. Graebner and M. Robbins, Phys. Rev. Lett. 36, 422 (1976).
- <sup>4</sup> Y. Onuki, I. Umehara, T. Ebihara, N. Nagai and K. Takita: ¯ J. Phys. Soc. Jpn. 61 (1992) 692.
- <sup>5</sup> F. M. Mueller, D. H. Lowndes, Y. K. Chang, A. J. Arko, and R. S. List, Phys. Rev. Lett. 68, 3928 (1992).
- <sup>6</sup> R. Corcoran, N. Harrison, S. M. Hayden, P. Meeson, M. Springford, and P. J. van der Wel, Phys. Rev. Lett. 72, 701 (1994).
- <sup>7</sup> N. Harrison, S. M. Hayden, P. Messon, M. Springford, P. J. van der Wel, and A. A. Menovsky: Phys. Rev. B50, 4208 (1994).
- <sup>8</sup> M. Heinecke and K. Winzer: Z. Phys. B98, 147 (1995); G. Goll, M. Heinecke, A. G. M. Jansen, W. Joss, L. Nguyen, E. Steep, K. Winzer and P. Wyder, Phys. Rev. B53, R8871 (1996).
- <sup>9</sup> T. Terashima, H. Takeya, S. Uji, K. Kadowaki and H. Aoki: Solid State Commun. 96, 459 (1995); T. Terashima, C. Haworth, H. Takeya, S. Uji, H. Aoki, and K. Kadowaki, Phys. Rev. B56, 5120 (1997).
- <sup>10</sup> M. Hedo, Y. Inada, T. Ishida, E. Yamamoto, Y. Haga, Y. Onuki, M. Higuchi, and A. Hasegawa, J. Phys. Soc. Jpn. ¯ 64, 4535 (1995).
- <sup>11</sup> H. Ohkuni, T. Ishida, Y. Inada, Y. Haga, E. Yamamoto, Y. Onuki, and S. Takahashi, J. Phys. Soc. Jpn. ¯ 66, 945 (1997); H. Ohkuni, Y. Inada, Y. Tokiwa, K. Sakurai, R. Settai,T. Honma, Y. Haga, E. Yamamoto, Y. Onuki, H. ¯ Yamagami, S. Takahashi, and T. Yanagisawa, Phil. Mag. B79, 1045 (1999).
- <sup>12</sup> C. Bergemann, S. R. Julian, G. J. McMullan, B. K. Howard, G. G. Lonzarich, P. Lejay, J. P. Brison, and J. Flouquet, Physica B230-232, 348 (1997).
- <sup>13</sup> Y. Inada, H. Yamagami, Y. Haga, K. Sakurai, Y. Tokiwa, T. Honma, E. Yamamoto, Y. Onuki, and T. Yanagisawa, ¯ J. Phys. Soc. Jpn. 68, 3643 (1999).
- <sup>14</sup> R. Settai, H. Shishido, S. Ikeda, Y. Murakawa, M. Nakashima, D. Aoki, Y. Haga, H. Harima, and Y. Onuki, ¯ J. Phys. Condens. Matter 13, L627 (2001).
- <sup>15</sup> P. J. van der Wel, J. Caulfield, R. Corcoran, P. Day, S. M. Hayden, W. Hayes, M. Kurmoo, P. Meeson, J. Singleton, and M. Springford, Physica C235-240, 2453 (1994).

There seems to be no analytic way to estimate Γ˜ <sup>s</sup>, so we fix it through the best fit to the numerical data of Fig. [8](#page-9-0)(b). Using Eq.([34\)](#page-8-0) with a <sup>2</sup> = 0.5∆<sup>2</sup> 0 , the procedure yields Γ˜ <sup>s</sup> = 0.125, as mentioned before. It is also clear both from Figs. [10](#page-10-0) and [11](#page-10-0) and from Eq.([C5](#page-13-0)) that the average gap around the extremal orbit is relevant for the extra attenuation. We hence put a 2Γ˜ <sup>ℓ</sup> = 0.5h|∆p| 2 <sup>i</sup>eoΓ˜ s. We thereby obtain Eq. [\(39](#page-9-0)), which yields excellent fits to the d- and p-wave numerical data without any adjustable parameters, as seen in Figs. [10](#page-10-0) and [11.](#page-10-0)

- <sup>16</sup> T. J. B. M. Janssen, C. Haworth, S. M. Hayden, P. Meeson, and M. Springford, Phys. Rev. B57, 11698 (1998).
- <sup>17</sup> Y. Inada and Y. Onuki, Fizika Nizkikh Temp. ¯ 25, 775 (1999).
- <sup>18</sup> K. Maki, Phys. Rev. B44, 2861 (1991).
- <sup>19</sup> K. Miyake, Physica B186-188, 115 (1993).
- <sup>20</sup> L. P. Gor'kov and J. R. Schrieffer, Phys. Rev. Lett. 80, 3360 (1998).
- <sup>21</sup> A. Wasserman and M. Springford, Physica B194-196, 1801 (1994).
- <sup>22</sup> M. J. Stephen, Phys. Rev. B45, 5481 (1992).
- <sup>23</sup> S. H. Curnoe, Phys. Rev. B62, 12413 (2000).
- <sup>24</sup> S. Dukan and Z. Teˇsanovi´c, Phys. Rev. Lett. 74, 2311 (1995).
- <sup>25</sup> T. Maniv, A. I. Rom, I. D. Vagner, and P. Wyder, Phys. Rev. B46, 8360 (1992).
- <sup>26</sup> M. G. Vavilov and V. P. Mineev, Zh. Exsp. Teor. Fiz. 112, 1873 (1997) [Sov. Phys. JETP 85, 1024 (1997]; V. P. Mineev, Phil. Mag. B80, 307 (2000).
- <sup>27</sup> G. M. Bruun, V. N. Nicopoulos, and N. F. Johnson, Phys. Rev. B56, 809 (1997).
- <sup>28</sup> V. M. Gvozdikov and M. V. Gvozdikova, Phys. Rev. B58, 8716 (1998).
- <sup>29</sup> M. R. Norman, A. H. MacDonald, and H. Akera, Phys. Rev. B51, 5927 (1995).
- <sup>30</sup> P. Miller and B. L. Gy¨orffy, J. Phys. Condens. Matter 7, 5579 (1995).
- <sup>31</sup> U. Brandt, W. Pesch, and L. Tewordt, Z. Phys. 201, 209 (1967).
- <sup>32</sup> R. B. Dingle, Proc. Roy. Soc. A211, 517 (1952).
- <sup>33</sup> This fact may be realized by observing that there is a clear zero-bias peak at the core (Fig. 8a) as well as the BCS gap structure (Fig. 8d) in their density of states. The sharp peaks in Fig. 8 may be due to a very small value of pFξ<sup>0</sup> (∼ 1) in their model.
- <sup>34</sup> T. Kita, J. Phys. Soc. Jpn. 67, 2075 (1998).
- <sup>35</sup> K. Yasui and T. Kita, Phys. Rev. Lett. 86, 1836 (1999).
- <sup>36</sup> R. Movshovich, M. F. Hundley, J. D. Thompson, P. C. Canfield, B. K. Cho and A. V. Chubukov: Physica C227 (1994) 381.
- <sup>37</sup> K. Izawa, K. Kamata, Y. Nakajima, Y. Matsuda, T. Watanabe, M. Nohara, H. Takagi, P. Thalmeier, and K. Maki, [cond-mat/0205178](http://arxiv.org/abs/cond-mat/0205178).
- <sup>38</sup> K. Yasui and T. Kita: J. Phys. Soc. Jpn. 70, 2852 (2001).
- <sup>39</sup> T. Kita, J. Phys. Soc. Jpn. 67, 2067 (1998).
- <sup>40</sup> See, e.g., L. I. Schiff: quantum mechanics 3rd ed. (McGraw-Hill, New York, 1981).
- <span id="page-15-0"></span><sup>41</sup> J. C. Ryan and A. K. Rajagopal: Phys. Rev. B47 (1993) 8843.
- <sup>42</sup> E. Helfand and N. R. Werthamer: Phys. Rev. 147 (1966) 288.
- <sup>43</sup> H. Won and K. Maki, Phys. Rev. B53, 5927 (1996).
- <sup>44</sup> M. Ichioka, A. Hasegawa, and K. Machida, Phys. Rev. B59, 8902 (1999).
- <sup>45</sup> R. Gilardi, J. Mesot, A. Drew, U. Divakar, S. L. Lee, E. M. Forgan, O. Zaharko, K. Conder, V. K. Aswal, C. D. Dewhurst, R. Cubitt, N. Momono, and M. Oda, Phys. Rev. Lett. 88, 217003 (2002).
- <sup>46</sup> M. Rasolt, Phys. Rev. Lett. 58, 1482 (1987).
- <sup>47</sup> A. H. MacDonald, H. Akera, and M. R. Norman, Phys. Rev. B45, 10147 (1992).
- <sup>48</sup> For a review on this topic, see, M. Rasolt and Z. Teˇsanovi´c, Rev. Mod. Phys. 64, 709 (1992).
- <sup>49</sup> M. Franz and Z. Teˇsanovi´c, Phys. Rev. Lett. 80, 4763 (1998).
- <sup>50</sup> J. M. Luttinger, Phys. Rev. 121, 1251 (1961).
- <sup>51</sup> B. P. Clayman and R. F. Frindt, Solid State Commun. 9, 1881 (1971).
- <sup>52</sup> D. H. Lee, L. W. Dubeck, and F. Rothwarf, Phys. Lett.
- 53A, 379 (1975).
- <sup>53</sup> R. Hackl, R. Kaiser, and S. Schicktanz, J. Phys. C 16, 1729 (1983).
- <sup>54</sup> J. C. F. Brock, Solid State Commun. 7, 1789 (1969).
- <sup>55</sup> D. Bintley and P. J. Meeson, unpublished.
- <sup>56</sup> T. Kita: J. Phys. Soc. Jpn. 65 (1996) 1355.
- <sup>57</sup> J. M. Luttinger and J. C. Ward, Phys. Rev. 118, 1417 (1960).
- <sup>58</sup> See, e.g., M. Abramowitz and I. A. Stegun: Handbook of Mathematical Functions 10th ed. (Dover, New York, 1972).
- <sup>59</sup> A. K. Rajagopal and J. C. Ryan: Phys. Rev. B44 (1991) 10280.
- <sup>60</sup> The p<sup>z</sup> integral in Eq.([C7\)](#page-13-0) passes through regions where δ≈0 in Eq. [\(C5](#page-13-0)), i.e., the second-order perturbation is not justified. However, the main contribution to the integral certainly comes from the regions where we can use Eq. [\(C1\)](#page-13-0). Notice that δ=−1/8 at the maximal orbit when the magnetization takes a local maximum.
- <sup>61</sup> It is found numerically that the variance of Eq. [\(C6\)](#page-13-0) depends little on the values of N<sup>F</sup> and N (∼NF), as expected. It is ∼0.2 for the s-wave pairing with δ=−1/8.