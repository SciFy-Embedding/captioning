# arXiv:1704.06805

**Paper ID:** 43f840cde8e08aa1a8752289460425cd

**PDF Path:** apl/Superconductors/arXiv:1704.06805.pdf

**Processing Status:** complete

**Captions Added:** 0

**Generated:** 2025-06-24T13:44:27.148341

---

# Creating better superconductors by periodic nanopatterning

M.P. Allan1\*, M.H. Fischer2,3, O. Ostojic<sup>1</sup> , A. Andringa<sup>1</sup>

1 Leiden Institute of Physics, Leiden University, Niels Bohrweg 2, 2333 CA Leiden, The Netherlands

2 Department of Condensed Matter Physics, Weizmann Institute of Science, Rehovot 7610001, Israel

3 Institute for Theoretical Physics, ETH Zurich, 8093 Zurich, Switzerland \* allan@physics.leidenuniv.nl

# Abstract

The quest to create superconductors with higher transition temperatures is as old as superconductivity itself. One strategy, popular after the realization that (conventional) superconductivity is mediated by phonons, is to chemically combine different elements within the crystalline unit cell to maximize the electronphonon coupling. This led to the discovery of NbTi and Nb3Sn, to name just the most technologically relevant examples. Here, we propose a radically different approach to transform a 'pristine' material into a better (meta-) superconductor by making use of modern fabrication techniques: designing and engineering the electronic properties of thin films via periodic patterning on the nanoscale. We present a model calculation to explore the key effects of different supercells that could be fabricated using nanofabrication or deliberate lattice mismatch, and demonstrate that specific pattern will enhance the coupling and the transition temperature. We also discuss how numerical methods could predict the correct design parameters to improve superconductivity in materials including Al, NbTi, and MgB<sup>2</sup>

Conventional — i.e., phonon-mediated — superconductors include many elemental metals with transition temperatures between 1K and 10K, simple alloys like NbTi and Nb3Sn with transition temperatures up to ∼20K, and MgB<sup>2</sup> with a record transition temperature of 39K at ambient pressure [\[1\]](#page-14-0). Mainly because of superior material properties that make fabrication and handling easy, these materials have widespread technological applications, ranging from medical magnetic resonance imaging to quantum information technologies. The effort to improve the quality of these conventional superconductors for applications has all but stopped with the discovery of high-temperature superconductors [\[2,](#page-14-1) [3,](#page-14-2) [4\]](#page-14-3) (with notable exceptions [\[5,](#page-14-4) [6,](#page-14-5) [7\]](#page-14-6)). Yet, any improvement of the quality of conventional superconductors has immediate and wide-ranging technological impact.

Here, we present a new method for such improvement using modern nanofabrication that allows for the creation of new materials with specially designed electronic (and phononic)

![](_page_1_Figure_2.jpeg)

<span id="page-1-0"></span>Figure 1: Electron-phonon interaction and the electron-phonon coupling parameter λ. a, Diagrammatic representation of the coupling between electrons with momentum k, k 0 through the exchange of a phonon with momentum q and interaction matrix element g 0 kq (here dependent only on q, g 0 kq = g 0 q ). b, Due to the kinematic constraint (Eq. [\(1\)](#page-2-0)), only scattering vectors connecting the Fermi surface points are relevant (red arrow). c, Kinematic constraint along a high-symmetry direction. d, The structure of the interaction matrix element determines what kinematic constraints lead to a high electron-phonon interaction λ; in the example here a large Fermi surface is beneficial.

structures that maximize the pairing interaction (Fig. [1\)](#page-1-0). In short, the idea is to engineer the phononic and electronic structure of a given material by introducing specific nanofabricated periodic supercell structures on thin films. In many ways, the strategy we propose here is similar to creating new superconductors by chemically changing their unit cell as it has been done mainly in the middle of the 20th century [\[3\]](#page-14-2), but approaching instead from the long-range limit, as current nanofabrication already allows structures of roughly 5 to 50 unit cells [\[8,](#page-14-7) [9,](#page-15-0) [10,](#page-15-1) [11,](#page-15-2) [12,](#page-15-3) [13,](#page-15-4) [14,](#page-15-5) [15\]](#page-15-6).

Before elaborating on the model calculation, we sketch methods to fabricate the nanofabricated patterns (Fig. [2\)](#page-2-1). The starting point is a 'pristine' material which is generally a thin film of a known superconductor (or any other material). One can pattern the film using standard cleanroom tools such as electron beam lithography [\[8\]](#page-14-7), photolithography, or focused ion beam lithography with He or Gd ions [\[9\]](#page-15-0), to make a supercell of a given size and shape (Fig. [2b](#page-2-1)-d). Using these methods it is possible to make periodic supercell structures down to a few nanometers in size, and even smaller patterns are possible using Moir´e engineering (Fig. 2e) [\[11,](#page-15-2) [12,](#page-15-3) [13\]](#page-15-4), self-assembly [\[16\]](#page-15-7), or atomic scale manipulation with scanning probe microscopy (Fig. 2f) [\[14,](#page-15-5) [15\]](#page-15-6). The choice of material, periodicity and supercell shape will allow for considerable freedom to design the desired electronic and phononic structure, together or individually.

The transition temperature of a conventional superconductor depends on the electronic and phononic structure as well as the coupling matrix elements in between them; the effect of all three is conveniently summarized in the dimensionless electron-phonon coupling parameter λ. In Bardeen-Cooper-Schrieffer theory [\[17,](#page-15-8) [18\]](#page-15-9), the critical temperature depends exponentially on λ, T<sup>c</sup> ∝ ωDe <sup>−</sup>1/λ, where ω<sup>D</sup> is the Debye frequency. λ thus represent an ideal figure of merit. We can calculate λ of the pristine material from the interaction matrix element g 0 kq for the scattering of an electron with momentum k to momentum k + q and the electron (εk) and phonon dispersions (ωq), by integrating over all possible scattering processes shown in

![](_page_2_Figure_2.jpeg)

<span id="page-2-1"></span>Figure 2: Fabrication methods. a, Modern nanofabrication tools allow one to make nanopatterned shapes with supercell periodicities of .5 to 50 lattice constants. b, Different shapes have different effects on the resulting electron-phonon coupling parameter λ. c, Different layers of (insulating) materials on top of the thin films will influence the phonon and electron dispersions individually and can increase the phonon energies. d, Stacking allows for 3D materials. e,f, Smaller patterning are possible using Moir´e engineering or single atom manipulation.

Fig. [1](#page-1-0)a [\[3,](#page-14-2) [18,](#page-15-9) [19,](#page-15-10) [20\]](#page-15-11). In the adiabatic limit it is given by

<span id="page-2-0"></span>
$$
\lambda^{\text{pristine}} = \sum\_{\mathbf{k}, \mathbf{q}} \frac{2}{\omega\_{\mathbf{q}} N(0)} |g\_{\mathbf{k}\mathbf{q}}^{0}|^{2} \delta(\varepsilon\_{\mathbf{k}}) \delta(\varepsilon\_{\mathbf{k}+\mathbf{q}}).\tag{1}
$$

The delta functions ensure that only states at the Fermi level participate. The main ingredients for λ are thus (i) the interaction matrix element of each process, (ii) the electronic density of states at the Fermi level N(0) ∝ P kq δ(εk), which determines the number of allowed processes, and (iii) the kinematic constraints from the Fermi surface and the coupling matrix element, given by the two delta functions and their matching with the momentum-space structure of the interaction matrix element (Fig. [1](#page-1-0)b,c). (Note that all sums are understood to be conventionally normalized by the number of lattice sites.) In the following, we discuss how to exploit these ingredients.

We use a model calculation that contains the key knobs of the method to explore the opportunities of designed electron dispersions for enhancing T<sup>c</sup> of a pristine material with a given coupling strength g 0 kq, and to demonstrate the feasibility of the concept. Our starting point is a two-dimensional pristine material defined on a square lattice. The electrons and phonons are coupled through the local shift of the chemical potential an electron feels due to the deformation of the positively charged lattice background from a phonon. The resulting interaction Hamiltonian reads [\[21\]](#page-15-12)

$$\mathcal{H}\_{\rm int} = D \sum\_{\mathbf{r}} (\nabla \cdot \mathbf{u}\_{\mathbf{r}}) c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}},$$

where c † <sup>r</sup> creates an electron on lattice site r and u<sup>r</sup> is the phonon displacement field. The proportionality D indicates the change of the chemical potential per volume change and is commonly called displacement potential. Note that in general D is not a constant, but reflects the shape of the atomic potential. We further describe the electrons by a nearest-neighbor tight-binding model and use harmonic potentials for the phonons. The details of the model are described in the Appendix.

![](_page_3_Figure_2.jpeg)

<span id="page-3-0"></span>Figure 3: Increased electron-phonon coupling parameter λ. a, Illustration of our model with a square lattice and a 6 × 6 supercell with a 2 × 2 hole; results for different supercell sizes and shapes are in Fig. [4.](#page-5-0) b, The supercell periodicity reduces the size of the Brillouin zone by a factor L (black line). The extended Brillouin zone is plotted as we consider umklapp scattering between different zones. The bands ξ ν k are shown in gray, the blue color marks the 'weight'. c, Density of states of the pristine material (red) and the material with supercell (blue). d, The phase space for scattering. The darkness indicates how much phase space is available, the red-blue indicates the strength of the interaction matrix element g 0 kq = g 0 q . In the pristine material, only small scattering vectors are allowed due to the kinematic constraints; in the modified material, larger q, where the interaction is stronger (red) are possible. e, f, λ<sup>q</sup> as an indication of how much different modes couple, as a colorplot (e) and along a high-symmetry direction (f) where the width of the line indicates the contribution of a phonon with that wave vector.

Next, we take the effect of the nano-patterned supercell into account and investigate how this allows to increase the coupling. We concentrate on the electronic structure, as in Bardeen-Cooper-Schrieffer theory, the details of the phonon dispersion has little effect on the critical temperature. We use a supercell of L × L lattice sites and a hole in the center (Fig. [3](#page-3-0)a) as a model of the realization shown in Fig. [2](#page-2-1)a, both because it is theoretically accessible and because it appears most promising. Note, however, that within the model, we can include different forms of supercells. For the electrons, the new supercell periodicity is reflected in a reduction of the Brillouin zone area by a factor of L 2 (Fig. [3](#page-3-0)b) and L <sup>2</sup> back-folded bands (ξ ν K) with band gaps at the reduced Brillouin zone boundary (ν is the band index). Fig. [3](#page-3-0)b shows the resulting bands for a 6×6 supercell. These back-folded bands ("shadow bands") can thus in principle help to overcome the kinematic constraints discussed above or in Eq. [\(1\)](#page-2-0). The scattering between the backfolded bands corresponds to umklapp scattering between different reduced Brioullin zones. The strength of these umklapp processes that connect different states is determined by a 'weighting' related to the transformation between old and new basis, which is closely related to the overlap of the new states with the states of the pristine material (denoted by blue lines in [3](#page-3-0)b). It is this weight that ensures that in the limit of large supercells, the enhancement of λ must vanish. Choosing the shape and size of the supercell allows to influence the weight and have the new scattering vectors align with the interaction matrix element of the pristine material at specific q points. Here, we absorb the weighting into a new interaction matrix element g νν<sup>0</sup> Kq. The electron-phonon coupling constant of Eq. [\(1\)](#page-2-0) now takes the form

$$
\lambda^{\rm new} = \sum\_{\mathbf{K}, \mathbf{q}, \nu, \nu'} \frac{2}{\omega\_{\mathbf{q}} N(0)} |g\_{\mathbf{K} \mathbf{q}}^{\nu \nu'}|^2 \delta(\xi\_{\mathbf{K}}^{\nu}) \delta(\xi\_{\mathbf{K} + \mathbf{q}}^{\nu'}). \tag{2}
$$

The sum runs over all momenta K inside the reduced Brillouin zone, and all q vectors in the Brillouin zone of the pristine material. This is equivalent to allowing umklapp scattering, but only within the first Brillouin zone of the pristine material, and allows to make a meaningful comparison with the pristine material.

Qualitatively, to achieve the highest coupling constant λ, we need to move weight from the original Fermi surface to match with the points of strongest electron-phonon coupling. For example, in many materials the interaction matrix element favors certain large scattering vectors [\[6\]](#page-14-5). It is then beneficial to allow many processes, where these wave vectors can scatter. In our example, we achieve precisely that: because the shadow bands cover much more area in the Brillouin zone, more phase space is available where it matters. This is relevant for materials like MgB2, where the coupling is strong for a small region in q space only, and giving more space in k space would be beneficial. In such a situation, the nanopatterning also weakens the Kohn anomaly and thus increases the phonon energy, further benefitting the superconductivity [\[5\]](#page-14-4).

Figure [4](#page-5-0)a summarizes our results for different supercell shapes and sizes. Even for larger supercell sizes, coinciding with what is presently possible with nanofabrication, one can improve the electron-phonon coupling by 'aligning' the kinematic constraints. Further, there is a clear dependence on the shape of the supercell. Finally, a strong effect stems from the increased density of states at certain energies due to the opening of band gaps. Fig. [4](#page-5-0)b shows the total coupling as a function of filling. The enhancements/suppressions at different fillings stem from density of states effects and from alterations of the phase space.

The electronic structure is not the only driver of the coupling. The periodic supercell structure can have a strong effect on the phonons as well. Designing phonons is well studied in opto-mechanics [\[22\]](#page-15-13) and sonic engineering [\[23\]](#page-15-14), but in the Eliashberg approximation they do not significantly influence the overall coupling [\[3\]](#page-14-2). More accurate calculations that take the retardation effects better into account go beyond the scope of this paper, but we note that recent Monte Carlo simulations revealed a strong dependence of the phonon energy on

![](_page_5_Figure_0.jpeg)

<span id="page-5-0"></span>Figure 4: Enhancement of λ and Tc. a, Dependence of the coupling constant λ on the supercell size L and shape, normalized by the change in N(0) (see Appendix for details). The labels are for all markers with the same color. All values are relative to the pristine material and normalized for density of states effects. The filling in both pristine and meta-superconductor is 0.5% for all supercell sizes, to ensure that we are below the first band gap and thus concentrate on the kinematic constraints. b Dependence of the coupling constant λ on the filling. Here, the resulting change is a combination of density of states effects and kinematic effects.

the superconducting transition temperature [\[24\]](#page-16-0), showing the additional potential of phonon engineering. We further mention that important effects such as the suppression of competing orders (e.g., charge density waves), or the improvement of screening [\[25\]](#page-16-1) similar to recently proposed colloid arrays [\[26,](#page-16-2) [27\]](#page-16-3) could yield further opportunities for improved superconductors.

While a model calculation as presented here allows to make educated guesses for a nanopatterning strategy on real materials, more sophisticated approaches can be used for specific materials. Local density approximation calculations can determine the exact phononic and electronic structure of simple materials, with supercell sizes in the range of several tens of lattice sites, and even take into account effects from dangling bonds. Second, finite-element calculations in combination with a tight-binding approach have proven to be a powerful tool to calculate the phononic structures at very low frequencies [\[23,](#page-15-14) [22\]](#page-15-13). For materials, where these branches are known to dominate electron-phonon coupling, this approach, generalized also for the electronic structure, will be useful. Third, new theoretical models that directly connect available parameters like ultrasonic attenuation with the coupling parameter λ might be useful for materials, where less is known about the underlying microscopic structure. Importantly, these methods should allow searches in a wide parameter space for specific materials.

To conclude this paper we would like to note that 'natural' nanoscale patterning has been seen in many quantum materials [\[28,](#page-16-4) [29,](#page-16-5) [30,](#page-16-6) [31,](#page-16-7) [32,](#page-16-8) [33\]](#page-16-9). (The effects described in this Letter are weaker but present in structures with clear spatial correlation length.) It is known that, depending on the lenghtscale, granular aluminum [\[32,](#page-16-8) [33\]](#page-16-9) can have enhanced T<sup>c</sup> compared to the constituents, but the cause for this is still debated. In high-temperature superconductors, a granular structure seems to be native to the material [\[29,](#page-16-5) [30\]](#page-16-6). Further, some interface superconductors can show superstructures due to Moir´e effects. Finally, in fullerene superconductors, the C<sup>60</sup> molecules form native supercells [\[28\]](#page-16-4). Some of these superconductors might not be driven by phonons, but most of our calculations are rather independent of the character of the mediating bosons. Whether there is a connection between these materials and the method presented here, and whether artificially designed granularity can be superior to the current state in these materials remains to be seen.

# Acknowledgements

We are thankful to Vadim Cheianov for guidance and to Jan Aarts, Irene Battisti, Koen Bastiaans, Andrea Caviglia, J.C. S´eamus Davis, John P. Davis, Simon Gr¨oblacher, Alex Kemper, Amir Safavi-Naeni, Manfred Sigrist, Ronny Thomale, Jasper van Wezel for helpful discussions.

Funding information We acknowledge support from the Netherlands Organization for Scientific Research (NWO) as part of the Frontiers of Nanoscience programme and the Vidi talent scheme (M.P.A.) and the Swiss Society of Friends of the Weizmann Institute (M.H.F).

# A Appendix: Model calculation

In this Appendix, we describe in more detail our model calculation for a square lattice with a nano-patterned supercell. We use the following strategy:

- 1. For illustration and for comparison, we first consider the model for the pristine material with the coupling matrix element g 0 kq, the resulting coupling constant λ, and the transition temperature Tc.
- 2. We model the supercell by setting the electron hopping to zero to the sites that are part of the hole, and find the new basis in which the electron Hamiltonian is diagonal. Further, we find the eigenvalues and transformation matrices.
- 3. We take the pristine interaction Hamiltonian, but replace the original electron operators by the new basis. This allows us to obtain the new interaction matrix element g νν<sup>0</sup> Kq between the new eigenstates. Here, we ignore the phonons because the details of their dispersion is not crucial in BSC theory.
- 4. We calculate the coupling constants λ from the new interaction matrix elements and electron dispersions.

#### A.1 Model for the pristine material

We start by describing the model for the pristine material (before the nano-patterning), a two-dimensional square lattice with N × N sites and Hamiltonian

$$
\mathcal{H} = \mathcal{H}\_{\rm el} + \mathcal{H}\_{\rm ph} + \mathcal{H}\_{\rm el-ph},
\tag{3}
$$

where Hel, Hph, Hel−ph are the electronic, phononic, and interaction parts of the Hamiltonian, respectively.

For the electronic part, we use a tight-binding description on the ions' equilibrium positions assuming nearest-neighbour hopping only with hopping element t,

<span id="page-7-0"></span>
$$\mathcal{H}\_{\rm el} = -t\sum\_{(\mathbf{r}, \mathbf{r}')} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}'} - \mu \sum\_{\mathbf{r}} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}},\tag{4}$$

where c † <sup>r</sup> (cr) is the electron creation (annihilation) operator, µ is the chemical potential and r denote the lattice sites. The electron creation operators in momentum space are

$$c\_{\mathbf{k}}^{\dagger} = \frac{1}{N} \sum\_{\mathbf{r}} e^{-i\mathbf{k} \cdot \mathbf{r}} c\_{\mathbf{r}}^{\dagger},\tag{5}$$

with

$$k\_x, k\_y \in \{ -\frac{\pi}{a}, -\frac{\pi}{a} \frac{N-1}{N}, \dots, \frac{\pi}{a} \frac{N-1}{N} \}, \tag{6}$$

where a is the lattice constant (in the following, we set a = 1). These operators diagonalize the electronic Hamiltonian Eq. [\(4\)](#page-7-0),

$$\mathcal{H}\_{\rm el} = \sum\_{\mathbf{k}} \varepsilon\_{\mathbf{k}} c\_{\mathbf{k}}^{\dagger} c\_{\mathbf{k}},\tag{7}$$

with ε<sup>k</sup> = −2t(cos k<sup>x</sup> + cos ky) − µ the electron dispersion.

We consider acoustic phonons stemming from nearest-neighbor (κ) and next-nearestneighbor (κ 0 ) springs on a square lattice,

$$\mathcal{H}\_{\rm ph} = \sum\_{\mathbf{r}} \frac{\mathbf{p}\_{\mathbf{r}}^2}{2m} + \frac{\kappa}{2} \sum\_{\langle \mathbf{r}, \mathbf{r}' \rangle} \left( \mathbf{e}\_{nn} \cdot (\mathbf{u}\_{\mathbf{r}} - \mathbf{u}\_{\mathbf{r}'}) \right)^2 + \frac{\kappa'}{2} \sum\_{\langle \mathbf{r}, \mathbf{r}' \rangle} \left( \mathbf{e}\_{nnn} \cdot (\mathbf{u}\_{\mathbf{r}} - \mathbf{u}\_{\mathbf{r}'}) \right)^2 + \frac{\kappa''}{2} \sum\_{\mathbf{r}} \mathbf{u}\_{\mathbf{r}}^2,\tag{8}$$

where p<sup>r</sup> are the ion momenta, u<sup>r</sup> are the deviations from equilibrium of the ions at site r, m is the ion mass, and κ, κ <sup>0</sup> are the spring constants. enn and ennn denote the unit vectors in the direction of nearest and next-nearest neighbours, respectively. To facilitate numerics, we added a "mass-term" κ <sup>00</sup> that removes the modes with zero energy. In the above sum, (r, r 0 ) denotes nearest neighbours and [r, r 0 ] denotes next-nearest neighbours. It is convenient to write the potential part of the phonon Hamiltonian in matrix form,

$$\mathcal{H}\_{\rm ph} = \sum\_{\mathbf{q}} \begin{pmatrix} u\_{\mathbf{q}}^x & u\_{\mathbf{q}}^y \end{pmatrix} \mathbf{H} \begin{pmatrix} u\_{-\mathbf{q}}^x \\ u\_{-\mathbf{q}}^y \end{pmatrix}, \tag{9}$$

with the matrix

$$\mathbf{H} = \begin{bmatrix} 2\kappa \sin^2(\frac{q\_x}{2}) + \kappa'(1 - \cos q\_x \cos q\_y) + \kappa'' & \kappa' \sin q\_x \sin q\_y\\ \kappa' \sin q\_x \sin q\_y & 2\kappa \sin^2(\frac{q\_y}{2}) + \kappa'(1 - \cos q\_x \cos q\_y) + \kappa'' \end{bmatrix} . \tag{10}$$

Diagonlaizing this matrix leads to the phonon spectrum

$$\begin{split} \omega\_{\mathbf{q}}^{\pm} &= \quad \frac{2}{m} \Big[ \kappa \sin^{2} \left( \frac{q\_{x}}{2} \right) + \kappa \sin^{2} \left( \frac{q\_{y}}{2} \right) + \kappa' \left( 1 - \cos q\_{x} \cos q\_{y} \right) + \kappa'' \Big] \\ &\quad \pm \frac{2}{m} \sqrt{\frac{\kappa^{2}}{4} (\cos q\_{y} - \cos q\_{x})^{2} + \kappa'^{2} \sin^{2}(q\_{x}) \sin^{2}(q\_{y})}. \end{split} \tag{11}$$

The (normalized) eigenvectors of the matrix, e ± q , indicate the polarisations of the normal modes. Along the high-symmetry axes (the nearest and next-nearest neighbour directions), as well as in the limit q = |q| 1, e + <sup>q</sup> points exactly along the longitudinal direction, and e − q along the transversal direction. For the parameters considered here, the deviation of e + q from the longitudinal direction is small: e + q · q/q ≥ 0.95 throughout the Brillouin zone. We thus approximate e + <sup>q</sup> ≈ e long <sup>q</sup> = q/q.

<span id="page-8-0"></span>The phonon Hamiltonian can be written in second-quantized form by introducing the phonon creation and annihilation operators aq,±, a † <sup>q</sup>,±. The displacement of the ion at site r is then

$$\begin{split} \mathbf{u\_{r}} &= \frac{1}{N} \sum\_{\mathbf{q}, \pm} \mathbf{e\_{q}^{\pm}} e^{i \mathbf{q} \cdot \mathbf{r}} u\_{\mathbf{q}}^{\pm} \\ &= \frac{1}{N} \sum\_{\mathbf{q}, \pm} \mathbf{e\_{q}^{\pm}} e^{i \mathbf{q} \cdot \mathbf{r}} \left( \frac{\hbar}{2m \omega\_{\mathbf{q}}^{\pm}} \right)^{\frac{1}{2}} \left( a\_{\mathbf{q}, \pm} + a\_{-\mathbf{q}, \pm}^{\dagger} \right) . \end{split} \tag{12}$$

We introduce the coupling of the electrons to the lattice by considering the energy change of the electronic states when the background (ion) density changes as the crystal expands/contracts, analogous to the dependence of the chemical potential on the density for a free electron gas. This leads to the interaction Hamiltonian

$$\mathcal{H}\_{\rm el-ph} = D \sum\_{\mathbf{r}} \frac{\Delta V\_{\mathbf{r}}}{V} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}}.\tag{13}$$

The constant D indicates the proportionality between change of the chemical potential and volume change ∆V /V ; it is commonly called displacement potential. For acoustic phonons, the volume change is approximately given by the divergence of the displacement field,

$$
\frac{\Delta V\_{\mathbf{r}}}{V} \approx \nabla \cdot \mathbf{u\_{r}}.\tag{14}
$$

Using the expression in Eq. [\(12\)](#page-8-0), we can write

$$\begin{split} \nabla \cdot \mathbf{u\_{r}} &= \frac{1}{N} \sum\_{\mathbf{q}, \pm} \nabla \cdot \mathbf{e\_{q}^{\pm}} e^{i \mathbf{q} \cdot \mathbf{r}} \left( \frac{\hbar}{2m\omega\_{\mathbf{q}}^{\pm}} \right)^{\frac{1}{2}} \left( a\_{\mathbf{q}, \pm} + a\_{-\mathbf{q}, \pm}^{\dagger} \right) \\ &= \frac{i}{N} \sum\_{\mathbf{q}, \pm} \mathbf{q} \cdot \mathbf{e\_{q}^{\pm}} e^{i \mathbf{q} \cdot \mathbf{r}} \left( \frac{\hbar}{2m\omega\_{\mathbf{q}}^{\pm}} \right)^{\frac{1}{2}} \left( a\_{\mathbf{q}, \pm} + a\_{-\mathbf{q}, \pm}^{\dagger} \right) \\ &\approx \frac{i}{N} \sum\_{\mathbf{q}} q e^{i \mathbf{q} \cdot \mathbf{r}} \left( \frac{\hbar}{2m\omega\_{\mathbf{q}}^{\pm}} \right)^{\frac{1}{2}} \left( a\_{\mathbf{q}, +} + a\_{-\mathbf{q}, +}^{\dagger} \right), \end{split} \tag{15}$$

where we approximated q · e + <sup>q</sup> ≈ q and q · e − <sup>q</sup> ≈ 0 as described above, i.e. only the phonons with energy ω + q , which are approximately the longitudinal phonons, couple to electrons, in accordance to Adler's theorem. For simplicity of notation, we will drop the '+' in the following. Note that there are different models that lead to a similar coupling Hamiltonian; overviews can be found in Refs [\[34,](#page-16-10) [35,](#page-16-11) [36,](#page-16-12) [21\]](#page-15-12). We also note that in the case of the lattice considered there, the divergence is of the form ∇ · u<sup>r</sup> ≈ [sin<sup>2</sup> qxu x <sup>q</sup> + sin<sup>2</sup> qyu y <sup>q</sup> ], however, as this will not change the ratios displayed in Fig. 4 of the main text, we continue with the continuum
approximation above. Inserting Eq. [\(15\)](#page-8-0) into Eq. [\(13\)](#page-8-1) and using q = k − k <sup>0</sup> yields

$$\begin{split} \mathcal{H}\_{\text{el}-\text{ph}} &= D \sum\_{\mathbf{r}} \frac{\Delta V\_{\mathbf{r}}}{V} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}} \\ &= \frac{i}{N^3} \sum\_{\mathbf{r}} \sum\_{\mathbf{k}\mathbf{q}} e^{i(\mathbf{q}+\mathbf{k}-\mathbf{k}') \cdot \mathbf{r}} D\_{\mathbf{q}} \Big( \frac{\hbar}{2m\omega\_{\mathbf{q}}} \Big)^{\frac{1}{2}} (a\_{\mathbf{q}} + a\_{-\mathbf{q}}^{\dagger}) c\_{\mathbf{k}'}^{\dagger} c\_{\mathbf{k}} \\ &= \frac{i}{N} \sum\_{\mathbf{k}\mathbf{q}} g\_{\mathbf{k}\mathbf{q}}^{0} (a\_{\mathbf{q}} + a\_{-\mathbf{q}}^{\dagger}) c\_{\mathbf{k}+\mathbf{q}}^{\dagger} c\_{\mathbf{k}}, \end{split} \tag{16}$$

with the coupling matrix element

<span id="page-9-0"></span>
$$g\_{\mathbf{kq}}^{0} = g\_{\mathbf{q}}^{0} = Dq \left(\frac{\hbar}{2m\omega\_{\mathbf{q}}}\right)^{\frac{1}{2}} = D\_{\mathbf{q}} \left(\frac{\hbar}{2m\omega\_{\mathbf{q}}}\right)^{\frac{1}{2}},\tag{17}$$

where we also introduced the momentum dependent proportionality Dq. In the simple approximation above, it is given by the expression above with D being a constant, but more generally it is related to the Fourier transform of the atomic potential, leading to different, material-specific expressions.

Given the interaction matrix element in Eq. [\(16\)](#page-9-0), the dimensionless coupling parameter λ can be expressed as

$$
\lambda = \frac{1}{N^4} \sum\_{\mathbf{kq}} \frac{2}{\omega\_\mathbf{q} N(0)} |g\_\mathbf{q}^0|^2 \delta(\varepsilon\_\mathbf{k}) \delta(\varepsilon\_\mathbf{k+q}). \tag{18}
$$

The product of delta functions δ(εk)δ(εk+q) ensures that only electrons at the Fermi level contribute (we assume that phonon energies are much smaller than the Fermi energy), yielding a kinematic constraint. It is often instructive to calculate the q or k dependence of the coupling constant, λ<sup>q</sup> and λk, by summing over all other variables. This yields the contributions of a given phonon mode q or the contributions of a given electronic state k to the coupling constant λ.

Finally, we can calculate the transition temperature T<sup>c</sup> using the standard Bardeen-Cooper-Schrieffer (BCS) theory or using the Allen-Dynes approximation in Eliashberg theory. In BCS, we have the standard exponential dependence,

<span id="page-9-1"></span>
$$T\_c = 1.13 \omega\_D e^{-\frac{1}{\lambda}},\tag{19}$$

where ω<sup>D</sup> is a measure of the phonon energy, usually taken to be the Debey energy. In Eliashberg theory we can approximate the transition temperature by

$$k\_B T\_c = \frac{\hbar \omega\_{\rm ln}}{1.2} \exp\left(\frac{1.04(1+\lambda)}{\lambda - \mu^\*(1+0.62\lambda)}\right),\tag{20}$$

where µ ∗ is the Coulomb pseudopotential. ωln is referred to as 'logarithmic average' of the phonon energy, defined by

$$
\omega\_{\rm ln} = \exp\left(\frac{2}{\lambda} \int\_0^\infty d\nu \ln(\nu) \frac{\alpha^2 F(\nu)}{\nu}\right),
\tag{21}
$$

<span id="page-9-2"></span>with

$$
\alpha^2 F(\nu) = \frac{1}{N^4} \frac{1}{N(0)} \sum\_{\mathbf{k}\mathbf{q}} \delta(\nu - \omega\_\mathbf{q}) |g\_{\mathbf{k}\mathbf{q}}^0|^2 \delta(\varepsilon\_\mathbf{k}) \delta(\varepsilon\_\mathbf{k+q}).\tag{22}
$$

# A.2 Electron part of the Hamiltonian with a supercell

We now consider a supercell with L×L sites as discussed in the main text. First, we introduce some nomenclature: We denote the number of supercells with M<sup>2</sup> = (N/L) 2 , each containing L 2 ions, giving the same total number of atomic sites as our pristine model, N<sup>2</sup> . We indicate the equilibrium position of each ion with r = R + τ , where R is the position of the supercell and τ is the position within the supercell. As before, the deviation of the ions from their equilibrium position is denoted by u<sup>r</sup> and the electron creation (annihilation) operators by c † <sup>r</sup> (cr). To introduce the supercell, we now allow for arbitrary chemical potentials µ<sup>τ</sup> at site τ inside the supercell and arbitrary hopping constants tτ,<sup>τ</sup> <sup>0</sup> for neighbouring sites τ and τ 0 within the supercell, all preserving the L-periodicity,

$$\mathcal{H}\_{\rm el} = -\sum\_{(\mathbf{r}, \mathbf{r}')} t\_{\mathbf{r}, \mathbf{r}'} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}'} - \sum\_{\mathbf{r}} \mu\_{\mathbf{r}} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}}.\tag{23}$$

We can bring this Hamiltonian in block-diagonal form by introducing a Fourier transform of the electron operators with respect to the new periodicity,

$$c\_{\{\mathbf{r} = \mathbf{R} + \boldsymbol{\tau}\}} = \frac{1}{M} \sum\_{\mathbf{K}} e^{i\mathbf{K} \cdot \mathbf{R}} c\_{\mathbf{K}}^{\boldsymbol{\tau}}.\tag{24}$$

Note that here and in the following, we use capital letters K = (Kx, Ky) to denote the reciprocal wave vectors with respect to the supercell periodicity L, i.e.,

$$K\_x, K\_y \in \{ -\frac{\pi}{La}, -\frac{\pi}{La} \frac{M-1}{M}, \dots, \frac{\pi}{La} \frac{M-1}{M} \}.$$

This leads to the block diagonal Hamiltonian

$$\mathcal{H}\_{\rm el} = \sum\_{\mathbf{K}} \mathbf{c}\_{\mathbf{K}}^{\dagger}[\mathbf{H}\_{\mathbf{K}}] c\_{\mathbf{K}},\tag{25}$$

where we introduced the vectors of operators c<sup>K</sup> = (c 1 K, c<sup>2</sup> K, . . . cL×<sup>L</sup> <sup>K</sup> ). For simplicity, when used as an index, we write τ instead of a vector τ for the position within the supercell.

Each block Hamiltonian [HK] is an L <sup>2</sup> × L <sup>2</sup> matrix. It contains diagonal elements with the chemical potentials of all L 2 sites, all the hopping elements, as well as the phase factors for connections between sites in adjacent supercells. Specifically, its elements are

$$\begin{split} [\mathbf{H}\mathbf{K}]^{\tau\tau'} &= -\mu\_{\tau}\delta\_{\tau\tau'} - t\_{\tau,\tau'}\delta\_{\{\tau,\tau'\}} - t\_{\tau,\tau'}\delta\_{\{\tau,\tau'\}} \\ &- t\_{\tau,\text{up}}\delta\_{\{\tau,\text{up}\}}e^{-iLK\_{y}} - t\_{\tau,\text{down}}\delta\_{\{\tau,\text{down}\}}e^{iLK\_{y}} \\ &- t\_{\tau,\text{right}}\delta\_{\{\tau,\text{right}\}}e^{-iLK\_{x}} - t\_{\tau,\text{left}}\delta\_{\{\tau,\text{left}\}}e^{iLK\_{x}}, \end{split} \tag{26}$$

where again we use a single index to denote the position in the supercell, and δττ<sup>0</sup> = 1 only when τ and τ <sup>0</sup> are the same while δ(ττ0) = 1 when τ and τ <sup>0</sup> are nearest neighbours. δ(τ,right) = 1 when τ is nearest neighbour to a site in the next supercell to the right, and similar for left, up and down. It is instructive to look at this in one dimension, in which case the block Hamiltonian [HK] is an L × L matrix:

$$\begin{aligned} \begin{bmatrix} \mathbf{H}\_{\mathbf{K}} \end{bmatrix} = \begin{pmatrix} -\mu\_{1} & -t\_{12} & -t\_{1,L}e^{-i(La)K} \\ -t\_{21} & -\mu\_{2} & \\ & \ddots & \\ & & -\mu\_{L-1} & -t\_{L-1,L} \\ -t\_{L,1}e^{i(La)K} & -t\_{L,L-1} & -\mu\_{L} \end{pmatrix} . \end{aligned} \tag{27}$$

Finally, diagonalizing the matrices [HK] for each K yields the eigenstates with operators

$$
\gamma\_\mathbf{K}^\nu = \sum\_\tau [\mathbf{U}\_\mathbf{K}]^{\nu \tau} c\_\mathbf{K}^\tau,\tag{28}
$$

with transformation matrices [UK] ντ and the corresponding eigenenergies ξ ν <sup>K</sup> such that

$$\mathcal{H}\_{\rm el} = \sum\_{\mathbf{K}} \sum\_{\nu} \xi\_{\mathbf{K}}^{\nu} (\gamma\_{\mathbf{K}}^{\nu})^{\dagger} \gamma\_{\mathbf{K}}^{\nu}. \tag{29}$$

It is possible to implement any kind of supercell with different chemical potentials µ<sup>τ</sup> and hopping tττ<sup>0</sup> in our calculation. To model the specific hole that we describe in the main text, we first designate some sites as part of the hole. These sites have zero hopping probability to all neighbours, leaving the states completely non-dispersive. Then, we increase the chemical potential at all hole sites to move the non-dispersive states above the relevant bands and thus ensure that only the latter contribute to superconductivity.

# A.3 Phonon part of the Hamiltonian with a supercell

Structures with periodic patterning made to alter phonon dispersions have been created in the context of opto-mechanics, where they are known as 'phononic crystals' [\[22\]](#page-15-0), or in the context of sound insulation and engineering, where they are known as 'sonic crystals' or 'acoustic metamaterials" [\[23\]](#page-15-1).)

While the periodically patterned films we propose influence both phononic and electronic structure, the fabrication method allows us to chose which one we influence strongest. In this Letter, we concentrate on the electronic structure, as in general the critical temperature within the BCS framework is not influenced by the momentum dependence of the phonon energies; models that correctly take phonon folding into account go beyond the scope of this paper.

The calculations of the folding of phonons using distributed mass-spring elements is similar to what we describe for electrons, and it has been done in the context described above. For detailed description of the formalism see e.g. Refs [\[37,](#page-16-0) [38,](#page-17-0) [23\]](#page-15-1). (One can also calculate the phonon dispersions in the elastic approximation [\[22\]](#page-15-0).) For the model in the main text we require a hole in the pristine material. The spring constants to the sites that are part of the hole are set to be small compared to the spring constant of the pristine material, to avoid coupling, and the masses to be heavy to suppress movements.

# A.4 The electron-phonon coupling in a system with supercells

We now want to write the new interaction matrix element g νν<sup>0</sup> Kq of our meta-material as a function of the folded electronic structure and interaction matrix element g 0 <sup>q</sup> of the pristine material.

Our starting point is again the interaction Hamiltonian of the form

$$\mathcal{H}\_{\rm el-ph} = \sum\_{\mathbf{r}} \underbrace{D \nabla \cdot \mathbf{u\_r} c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}}}\_{\text{(I)}}.\tag{30}$$

As in the pristine case, we write for the phonon part

$$\begin{split} \mathbf{(I)} &= D \nabla \cdot \mathbf{u\_{r}} \\ &\approx \frac{i}{N} \sum\_{\mathbf{q}} g\_{\mathbf{q}}^{0} e^{i\mathbf{q} \cdot \mathbf{r}} (a\_{\mathbf{q}} + a\_{-\mathbf{q}}^{\dagger}). \end{split} \tag{31}$$

However, we now replace the electron operators of the pristine material with the operators of the new eigenstates. The real space electron operators are then given by

$$c\_{\{\mathbf{r} = \mathbf{R} + \tau\}} = \frac{1}{M} \sum\_{\mathbf{K}} \sum\_{\nu} e^{i\mathbf{K} \cdot \mathbf{R}} [\mathbf{U}\_{\mathbf{K}}^{\dagger}]^{\tau \nu} \gamma\_{\mathbf{K}}^{\nu} \,. \tag{32}$$

This yields an electron part

$$\begin{split} \langle \mathbf{II} \rangle &= c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}} \\ &= \frac{1}{M^{2}} \sum\_{\mathbf{K} \mathbf{K}'} \sum\_{\nu \nu'} \left[ e^{i\mathbf{K} \cdot \mathbf{R}} (\gamma\_{\mathbf{K}}^{\nu})^{\dagger} \mathbf{U}\_{\mathbf{K}}^{\nu \tau} \right] \left[ e^{-i\mathbf{K}' \cdot \mathbf{R}} [\mathbf{U}\_{\mathbf{K}'}^{\dagger}]^{\tau \nu'} \gamma\_{\mathbf{K}'}^{\nu'} \right] \\ &= \frac{1}{M^{2}} \sum\_{\mathbf{K} \mathbf{K}'} \sum\_{\nu \nu'} e^{i(\mathbf{K} - \mathbf{K}') \cdot \mathbf{R}} [\mathbf{U}\_{\mathbf{K}}]^{\nu \tau} [\mathbf{U}\_{\mathbf{K}'}^{\dagger}]^{\tau \nu'} (\gamma\_{\mathbf{K}}^{\nu})^{\dagger} \gamma\_{\mathbf{K}'}^{\nu'}. \end{split} \tag{33}$$

Putting both together and using

$$\frac{1}{M^2} \sum\_{\mathbf{R}} e^{i(\mathbf{q} - \mathbf{K} + \mathbf{K}') \cdot \mathbf{R}} = \sum\_{\mathbf{l}} \delta\_{\mathbf{K}, \mathbf{K}' + \mathbf{Q} + \frac{2\pi}{La} \mathbf{l}},\tag{34}$$

(introducing q = Q + 2π La l) we obtain

$$\begin{split} \mathcal{H}\_{\text{el-ph}} &= \sum\_{\mathbf{r}} D\_{\mathbf{q}} (\nabla \cdot \mathbf{u}\_{\mathbf{r}}) c\_{\mathbf{r}}^{\dagger} c\_{\mathbf{r}} \\ &= \frac{i}{M^{2}N} \sum\_{\mathbf{r}} \sum\_{\mathbf{K} \mathbf{K}'} \sum\_{\mathbf{K} \mathbf{K}'} \sum\_{\nu \nu'} e^{i(\mathbf{K} - \mathbf{K}' - \mathbf{Q}) \cdot \mathbf{R}} [\mathbf{U}\_{\mathbf{K}}]^{\nu \tau} [\mathbf{U}\_{\mathbf{K}'}^{\dagger}]^{\tau \nu'} (\gamma\_{\mathbf{K}}^{\nu})^{\dagger} \gamma\_{\mathbf{K}'}^{\nu'} \\ &\quad \times g\_{\mathbf{q}}^{0} e^{i\mathbf{q} \cdot \mathbf{r}} \left( a\_{\mathbf{q}} + a\_{-\mathbf{q}}^{\dagger} \right) \\ &= \frac{i}{N} \sum\_{\mathbf{K} \mathbf{Q}} \sum\_{\mathbf{l}} \sum\_{\tau} \sum\_{\nu \nu'} g\_{\mathbf{q}}^{0} e^{i\mathbf{q} \cdot \tau} [\mathbf{U}\_{\mathbf{K} + \mathbf{Q}}]^{\nu \tau} [\mathbf{U}\_{\mathbf{K}}]^{\tau \nu'} (\gamma\_{\mathbf{K} + \mathbf{Q}}^{\nu})^{\dagger} \gamma\_{\mathbf{K}}^{\nu'} (a\_{\mathbf{q}} + a\_{-\mathbf{q}}^{\dagger}). \end{split} \tag{35}$$

Note that the sum runs over all K inside the reduced Brillouin zone, and all q vectors in the Brillouin zone of the pristine material. This is equivalent to allowing umklapp scattering, but only within the Brillouin zone of the pristine material, to make a meaningful comparison with the pristine material.

Finally, we rewrite the interaction Hamiltonian as

$$\mathcal{H}\_{\rm el-ph} = \frac{i}{N} \sum\_{\mathbf{K} \mathbf{Q}, 1} \sum\_{\nu \nu'} g\_{\mathbf{K} \mathbf{q}}^{\nu \nu'} (\gamma\_{\mathbf{K}+\mathbf{Q}}^{\nu})^\dagger \gamma\_{\mathbf{K}}^{\nu'} (a\_{\mathbf{q}} + a\_{-\mathbf{q}}^\dagger), \tag{36}$$

with the coupling vertex for the new eigenstates

<span id="page-12-0"></span>
$$g\_{\mathbf{Kq}}^{\nu\nu'} = g\_{\mathbf{q}}^0 \sum\_{\tau} e^{i\mathbf{q}\cdot\boldsymbol{\tau}} [\mathbf{U}\mathbf{K} + \mathbf{Q}]^{\nu\tau} [\mathbf{U}\_{\mathbf{K}}^\dagger]^{\tau\nu'}.\tag{37}$$

Again, the displacement potential proportionality can have a complex, material-specific q dependence, i.e. D does not need to be constant. This stems from the spatial dependence of the deformation potential, and reflects the interaction matrix element in real materials such as MgB2.

# A.4.1 Coupling parameter λ and transition temperature T<sup>c</sup>

We calculate the coupling parameter λ similar to the case of the pristine material, where we used

$$
\lambda^{\text{pristine}} = \frac{1}{N^4} \sum\_{\mathbf{kq}} \frac{2}{\omega\_\mathbf{q} N(0)} |g\_\mathbf{q}^0|^2 \delta(\varepsilon\_\mathbf{k}) \delta(\varepsilon\_\mathbf{k+q}), \tag{38}
$$

but now replacing the interaction matrix element from the pristine material, g 0 q , with the one calculated above, g νν<sup>0</sup> Kq, and summing over momenta q in the 'large' Brioullin zone of the pristine material. As mentioned, one can interpret this as including an amplitude for umklapp scattering in between the new, 'small' Brioullin zones. This yields

<span id="page-13-0"></span>
$$
\lambda^{\rm SC} = \frac{1}{N^4} \sum\_{\mathbf{Kq}\nu\nu'} \frac{2}{\omega\_\mathbf{q} N(0)} |g\_{\mathbf{Kq}}^{\nu\nu'}|^2 \delta(\xi\_\mathbf{K}^\nu) \delta(\xi\_\mathbf{K+q}^{\nu'}), \tag{39}
$$

with g νν<sup>0</sup> Kq defined above. Note that this is now a multi-band superconductor. Throughout this work, we use the coupling parameter λ as a figure of merit. We emphazise that it is directly related to the transition temperature T<sup>c</sup> according to Eqs [19-](#page-9-1)[22.](#page-9-2)

#### A.4.2 Numerical implementation

We use the commercial Matlab package for all calculations, and we will now briefly outline how all calculations can be expressed as combinations of matrix operation and matrix diagonalization. We start with the interaction matrix element from Eq. [\(37\)](#page-12-0) and insert it into Eq. [\(39\)](#page-13-0):

λ SC = 1 N<sup>4</sup> X Kqνν<sup>0</sup> 2 <sup>ω</sup>qN(0)|<sup>g</sup> νν<sup>0</sup> Kq| 2 δ(ξ ν <sup>K</sup>)δ(ξ ν 0 K+q ) = 1 N<sup>4</sup> X Kqνν<sup>0</sup> 2 ωqN(0) g 0 q X τ e iq·τ [UK+q] ντ [U † K] τ ν<sup>0</sup> 2 δ(ξ ν <sup>K</sup>)δ(ξ ν 0 K+q ) = 1 N<sup>4</sup> X Kq 2 <sup>ω</sup>qN(0)|<sup>g</sup> 0 q | 2X νν<sup>0</sup> X τ e iq·τ [UK+q] ντ [U † K] τ ν<sup>0</sup> δ(ξ ν K) 2 δ(ξ ν 0 K+q ) = 1 N<sup>4</sup> X Kq F<sup>q</sup> X νν<sup>0</sup> X τ e iq·τ [UK+q] ντ [U † K] τ ν<sup>0</sup> 2 δ(ξ ν <sup>K</sup>)δ(ξ ν 0 K+q ), (40)

with

$$F\_{\mathbf{q}} = \frac{2}{\omega\_{\mathbf{q}} N(0)} \frac{D\_{\mathbf{q}}^2 \hbar}{2m\omega\_{\mathbf{q}}} = |g\_{\mathbf{q}}^0|^2 \frac{2}{\omega\_{\mathbf{q}} N(0)}.\tag{41}$$

In the last two steps we used the fact that the terms we took outside the sum are positive and independent of νν<sup>0</sup> . We further define λKq such that λ SC = 1/(N2M<sup>2</sup> ) P Kq λKq.

Now we can rewrite this into a form convenient for numerical matrix operations:

λKq = F<sup>q</sup> L2 X νν<sup>0</sup> X τ e iq·τ [UK+q] ντ [U † K] τ ν<sup>0</sup> 2 δ(ξ ν <sup>K</sup>)δ(ξ ν 0 K+q ) = F<sup>q</sup> L2 X νν<sup>0</sup> X ττ<sup>0</sup> [UK+q] ντ [δττ<sup>0</sup>e iq·τ ] ττ<sup>0</sup> [U † K] τ 0ν 0 2 δ(ξ ν <sup>K</sup>)δ(ξ ν 0 K+q ) = F<sup>q</sup> L2 X νν<sup>0</sup> h [UK+q] ∗ [diag(eiq·<sup>τ</sup> )] ∗ [U † K] iνν<sup>0</sup> 2 δ(ξ ν <sup>K</sup>)δ(ξ ν 0 K+q ) = F<sup>q</sup> L2 X νν<sup>0</sup> h [UK+q] ∗ [diag(eiq·<sup>τ</sup> )] ∗ [U † K] iνν<sup>0</sup> 2 ? h δ(ξ ν 0 K+q ) ∗ δ(ξ ν K) i , (42)

where ∗ indicates a matrix multiplication, ? indicates element-wise multiplication, and [diag(eiq·<sup>τ</sup> )] is a L <sup>2</sup> ×L <sup>2</sup> diagonal matrix with diagonal elements (e iq·τ ). The sum over νν<sup>0</sup> is then the sum over matrix elements after taking the absolute square. The multiplication δ(ξ ν 0 K+q ) ∗ δ(ξ ν <sup>K</sup>) is a multiplication of a L <sup>2</sup> × 1 vector with a 1 × L <sup>2</sup> vector yielding a L <sup>2</sup> × L <sup>2</sup> matrix.

For Figure 4a in the main text, we want to concentrate on kinematic effects only. We take care of a normalisation in the following way. First, we always compare to the pristine material at the same filling. Second, for the results shown in Fig. 4a, we normalize by the density of states to concentrate on the kinematic effects only.

# References

- [1] C. Buzea and T. Yamashita, Review of the superconducting properties of MgB2, Superconductor Science and Technology 14(11), R115 (2001), doi[:10.1088/0953-2048/14/11/201.](http://dx.doi.org/10.1088/0953-2048/14/11/201)
- [2] H. Rogalla and P. Kes, 100 Years of Superconductivity, CRC Press, ISBN 9781439849484 (2011).
- [3] F. Marsiglio and J. P. Carbotte, Superconductivity, volume 1: conventional and unconventional superconductors, chap. 3, pp. 201–213, Springer, Berlin Heidelberg, ISBN 9783540732532, doi[:10.1007/978-3-540-73253-2](http://dx.doi.org/10.1007/978-3-540-73253-2) (2008).
- [4] V. Ginzburg and D. Kirzhnits, The question of high-temperature and surface superconductivity, In Soviet Physics Doklady, vol. 12, p. 880 (1968).
- [5] W. E. Pickett, Design for a room-temperature superconductor, Journal of Superconductivity and Novel Magnetism 19(3-5), 291 (2006), doi[:10.1007/s10948-006-0164-9.](http://dx.doi.org/10.1007/s10948-006-0164-9)
- [6] W. E. Pickett, The next breakthrough in phonon-mediated superconductivity, Physica C: Superconductivity and its Applications 468, 126 (2008), doi[:10.1016/j.physc.2007.08.018.](http://dx.doi.org/10.1016/j.physc.2007.08.018)
- [7] G. Karakonstantakis, L. Liu, R. Thomale and S. A. Kivelson, Correlations and renormalization of the electron-phonon coupling in the honeycomb hubbard ladder and superconductivity in polyacene, Phys. Rev. B 88, 224512 (2013), doi[:10.1103/PhysRevB.88.224512.](http://dx.doi.org/10.1103/PhysRevB.88.224512)
- [8] a. E. Grigorescu and C. W. Hagen, Resists for sub-20-nm electron beam lithography with a focus on HSQ: state of the art, Nanotechnology 20(29), 31 (2009), doi[:10.1038/36797.](http://dx.doi.org/10.1038/36797)
- [9] C. Genet and T. W. Ebbesen, Light in tiny holes, Nature 445, 7123 (2007), doi[:10.1038/nature05350.](http://dx.doi.org/10.1038/nature05350)
- [10] G. F. Schneider, S. W. Kowalczyk, V. E. Calado, G. Pandraud, H. W. Zandbergen, L. M. K. Vandersypen and C. Dekker, DNA translocation through graphene nanopores, Nano Letters 10(8), 3163 (2010), doi[:10.1021/nl102069z.](http://dx.doi.org/10.1021/nl102069z)
- [11] M. Corso, Boron Nitride Nanomesh, Science 303(5655), 217 (2004), doi[:10.1007/s11671-](http://dx.doi.org/10.1007/s11671-006-9036-2) [006-9036-2.](http://dx.doi.org/10.1007/s11671-006-9036-2)
- [12] M. P. Allan, S. Berner, M. Corso, T. Greber and J. Osterwalder, Tunable self-assembly of one-dimensional nanostructures with orthogonal directions, Nanoscale Research Letters 2(2), 94 (2007), doi[:10.1007/s11671-006-9036-2.](http://dx.doi.org/10.1007/s11671-006-9036-2)
- [13] B. Hunt, J. D. Sanchez-Yamagishi, A. F. Young, M. Yankowitz, B. J. LeRoy, K. Watanabe, T. Taniguchi, P. Moon, M. Koshino, P. Jarillo-Herrero and R. C. Ashoori, Massive dirac fermions and hofstadter butterfly in a van der Waals heterostructure, Science 340(6139), 1427 (2013), doi[:10.1126/science.1237240.](http://dx.doi.org/10.1126/science.1237240)
- [14] E. K. Eigler and D. K. Schweizer, Positioning single atoms with a scanning tunnelling microscope, doi[:10.1038/344524a0](http://dx.doi.org/10.1038/344524a0) (1990).
- [15] F. E. Kalff, M. P. Rebergen, E. Fahrenfort, J. Girovsky, R. Toskovic, J. L. Lado, J. Fern´andez-Rossier and A. F. Otte, A kilobyte rewritable atomic memory, Nature nanotechnology 18(July), 1 (2016), doi[:10.1038/nnano.2016.131.](http://dx.doi.org/10.1038/nnano.2016.131)
- [16] Z. Nie, A. Petukhova and E. Kumacheva, Properties and emerging applications of selfassembled structures made from inorganic nanoparticles, Nature nanotechnology 5(1), 15 (2010), doi[:10.1038/nnano.2009.453.](http://dx.doi.org/10.1038/nnano.2009.453)
- [17] J. Bardeen, L. N. Cooper and J. R. Schrieffer, Microscopic theory of superconductivity, Physical Review 106(5), 1175 (1957), doi[:10.1103/PhysRev.106.162.](http://dx.doi.org/10.1103/PhysRev.106.162)
- [18] R. Parks, Superconductivity: part 1 and 2, Superconductivity. Taylor & Francis, ISBN 9780824715205 (1969).
- [19] P. B. Allen, Neutron spectroscopy of superconductors, Physical Review B 6(7), 2577 (1972), doi[:10.1103/PhysRevB.6.2577.](http://dx.doi.org/10.1103/PhysRevB.6.2577)
- [20] Z. Yin, S. Savrasov and W. Pickett, Linear response study of strong electronphonon coupling in yttrium under pressure, Physical Review B 74(9), 094519 (2006), doi[:10.1103/PhysRevB.74.094519.](http://dx.doi.org/10.1103/PhysRevB.74.094519)
- [21] P. Coleman, Introduction to Many-Body Physics, Cambridge University Press, doi[:10.1017/CBO9781139020916](http://dx.doi.org/10.1017/CBO9781139020916) (2015).
- <span id="page-15-0"></span>[22] A. H. Safavi-Naeini and O. Painter, Optomechanical crystal devices, In Cavity Optomechanics - Nano- and Micromechanical Resonators Interacting with Light. Springer, doi[:10.1007/978-3-642-55312-7](http://dx.doi.org/10.1007/978-3-642-55312-7) (2014).
- <span id="page-15-1"></span>[23] P. A. Deymier, Acoustic Metamaterials and Phononic Crystals, Springer Series in solidstate science, doi[:10.1007/978-3-642-31232-8](http://dx.doi.org/10.1007/978-3-642-31232-8) (2013).
- [24] C. Lin, B. Wang and K. H. Teo, Optimal boson energy for superconductivity in the Holstein model, Physical Review B - Condensed Matter and Materials Physics 93(22), 1 (2016), doi[:10.1103/PhysRevB.93.224501.](http://dx.doi.org/10.1103/PhysRevB.93.224501)
- [25] J. T. Shen, P. B. Catrysse and S. Fan, Mechanism for designing metallic metamaterials with a high index of refraction, Physical Review Letters 94(19), 1 (2005), doi[:10.1103/PhysRevLett.94.197401.](http://dx.doi.org/10.1103/PhysRevLett.94.197401)
- [26] I. I. Smolyaninov and V. N. Smolyaninova, Is there a metamaterial route to high temperature superconductivity?, Advances in Condensed matter physics 2014(2), 479635 (2013), doi[:10.1155/2014/479635.](http://dx.doi.org/10.1155/2014/479635)
- [27] I. I. Smolyaninov and V. N. Smolyaninova, Theoretical modeling of critical temperature increase in metamaterial superconductors, Physical Review B - Condensed Matter and Materials Physics 93(18), 1 (2016), doi[:10.1103/PhysRevB.93.184510.](http://dx.doi.org/10.1103/PhysRevB.93.184510)
- [28] O. Gunnarsson, Superconductivity in Fullerides, Reviews of Modern Physics 69(2), 33 (1996), doi[:10.1103/RevModPhys.69.575.](http://dx.doi.org/10.1103/RevModPhys.69.575)
- [29] K. Fujita, M. H. Hamidian, I. A. Firmo, S. Mukhopadhyay, C. K. Kim, H. Eisaki, S.-i. Uchida and J. Davis, Spectroscopic imaging STM: atomic-scale visualization of electronic structure and symmetry in underdoped cuprates, vol. Chapter 3 of Springer Series in Solid-State Sciences, Springer Berlin Heidelberg, Berlin, Heidelberg, doi[:10.1007/978-3-](http://dx.doi.org/10.1007/978-3-642-21831-6) [642-21831-6](http://dx.doi.org/10.1007/978-3-642-21831-6) (2012).
- [30] R. Comin and A. Damascelli, Resonant x-ray scattering studies of charge order in cuprates, Annual Review of Condensed Matter Physics 7(1), 369 (2016), doi[:10.1146/annurev-conmatphys-031115-011401.](http://dx.doi.org/10.1146/annurev-conmatphys-031115-011401)
- [31] Y. Maeno, T. Ando, Y. Mori, E. Ohmichi, S. Ikeda, S. NishiZaki and S. Nakatsuji, Enhancement of superconductivity of Sr2RuO4to 3 K by embedded metallic microdomains, Physical Review Letters 81(17), 3765 (1998), doi[:10.1103/PhysRevLett.81.3765.](http://dx.doi.org/10.1103/PhysRevLett.81.3765)
- [32] R. W. Cohen and B. Abeles, Superconductivity in granular aluminum films, Physical Review 168(2), 444 (1968), doi[:10.1103/PhysRev.168.444.](http://dx.doi.org/10.1103/PhysRev.168.444)
- [33] A. K. Geim, I. V. Grigorieva, S. V. Dubonos, J. G. S. Lok, J. C. Maan, A. E. Filippov and F. M. Peeters, Phase transitions in individual sub-micrometre superconductors, Nature 390(November), 259 (1997), doi[:10.1038/36797.](http://dx.doi.org/10.1038/36797)
- [34] O. Madelung, Introduction to Solid-State Theory, Springer Verlag, doi[:10.1007/978-3-](http://dx.doi.org/10.1007/978-3-642-61885-7) [642-61885-7](http://dx.doi.org/10.1007/978-3-642-61885-7) (1978).
- [35] H. Bruus and K. Flensberg, Many-Body Quantum Theory in Condensed Matter Physics: An Introduction, Oxford Graduate Texts (2004).
- [36] P. Phillips, Advanced Solid State Physics , Cambridge University Press, doi[:10.1017/CBO9781139031066](http://dx.doi.org/10.1017/CBO9781139031066) (2012).
- <span id="page-16-0"></span>[37] J. S. Jensen, Phononic band gaps and vibrations in one- and two-dimensional mass-spring structures, Journal of Sound and Vibration 266(5), 1053 (2003), doi[:10.1016/S0022-](http://dx.doi.org/10.1016/S0022-460X(02)01629-2) [460X\(02\)01629-2.](http://dx.doi.org/10.1016/S0022-460X(02)01629-2)

<span id="page-17-0"></span>[38] Z. J. Cao, One and Two-Dimensional Mass Spring Computational Model for Phononic Band Gap Analysis, Ph.D. thesis, University of Waterloo (2009).