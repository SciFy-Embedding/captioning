# arXiv:1210.3705

**Paper ID:** ac39b6874b6b1990815c8a9e1a60c6ec

**PDF Path:** apl/Superconductors/arXiv:1210.3705.pdf

**Processing Status:** complete

**Captions Added:** 7

**Generated:** 2025-06-24T13:44:26.924625

---

# Strain sensitivity and superconducting properties of Nb3Sn from first principles calculations

G. De Marzi,[∗](#page-5-0) L. Morici, L. Muzzi, and A. della Corte

*EURATOM-ENEA Association on Fusion, Via Enrico Fermi 45, 00044 Frascati RM, Italy*

M. Buongiorno Nardelli

*Department of Physics, University of North Texas, 1155 Union Circle 311427, Denton, Texas 76203-5017, USA and*

*CSMD, Oak Ridge National Laboratory, Oak Ridge, TN 37831*

(Dated: February 21, 2022)

Using calculations from first principles based on density functional theory we have studied the strain sensitivity of the high-field superconducting magnet A15 Nb3Sn. The Nb3Sn lattice cell was deformed in the same way as observed experimentally on multi-filamentary, technological wires subject to loads applied along their axes. The phonon dispersion curves and electronic band structures along different high-symmetry directions in the Brillouin zone were calculated, at different levels of applied strain, ε, both on the compressive and the tensile side. Starting from the calculated averaged phonon frequencies and electron-phonon coupling, the superconducting characteristic critical temperature of the material, *Tc*, has been calculated by means of the Allen-Dynes modification of the McMillan formula. As a result, the characteristic bell-shaped *Tc* vs. ε curve, with a maximum at zero intrinsic strain, and with a slight asymmetry between the tensile and compressive sides, has been obtained. These first-principle calculations thus show that the strain sensitivity of Nb3Sn has a microscopic and intrinsic origin, originating from shifts in the Nb3Sn critical surface. In addition, our computations show that variations of superconducting properties of this compound are correlated to stress-induced changes in both the phononic and electronic properties. Finally, the strain function describing the strain sensitivity of Nb3Sn has been extracted from the computed *Tc*(ε) curve, and compared to experimental data from multi-filamentary, composite wires. Both curves show the expected bell-shaped behavior, but the strain sensitivity of the wire is enhanced with respect to the theoretical predictions of the bulk, perfectly binary and stoichiometric Nb3Sn. Understanding the origin of this difference might open potential pathways towards the improvement of the strain tolerance in such systems.

PACS numbers: 74.70.Ad, 74.25.Kc, 74.25.Jb, 68.35.Gy, 63.20.kd, 63.20.dk, 71.15.Mb

#### I. INTRODUCTION

The A15 phase Nb3Sn compound[1](#page-5-1) is currently being used in a variety of large-scale scientific projects employing highfield superconducting magnets (above 10 T)[2](#page-5-2) , including ITER (the International Thermonuclear Experimental Reactor)[3](#page-5-3)[–5](#page-5-4) , the 1 GHz NMR project[6](#page-5-5) , and the CERN LHC Luminosity Upgrade[7](#page-5-6) . In these high-field magnets, the mechanical loads during cooldown (due to different thermal contractions) and operation (due to Lorentz forces) can be very large, and since the superconducting properties of Nb3Sn strongly depend on strain[8–](#page-5-7)[11](#page-5-8), an overall performance degradation can take place. Therefore, for a magnet's sound design it is of fundamental importance to have knowledge of the behavior of the superconducting parameters (namely the critical temperature,T*c*, the upper critical field, B*c*2, and the critical current, I*c*) as a function of strain, ε.

Of particular interest is the uniaxial stress (either in tension or compression), acting along the axial direction of the composite, multi-filamentary wires used in such systems: in a Cable-in-Conduit Conductor (CICC)[12](#page-5-9)[–14](#page-5-10), for example, the Nb3Sn wires are inserted into stainless steel conduits, and compressive stresses due to the different thermal contraction coefficients of the different materials become important[15,](#page-5-11)[16](#page-5-12) . Trasverse load components might also be important[11,](#page-5-8)[17](#page-5-13), but we will focus here only on the uniaxial ones.

Within the framework of the Unified Scaling Law[18](#page-5-14), many authors[18](#page-5-14)[–23](#page-6-0) have proposed modified scaling equations which take into account the uniaxial strain dependence through the so-called strain function, s(ε), but few attempts have been made[19,](#page-5-15)[20](#page-6-1)[,24](#page-6-2) on obtaining a scaling law based on microscopic parameters. For this purpose, a very first step would be to accurately determine the electronic band structures, the phonon dispersion curves and the electron-phonon coupling terms, and study their evolution as a function of applied strain. To this aim, the knowledge of the Nb3Sn lattice cell deformation when a multifilamentary wire is subject to different stress

![](_page_0_Figure_15.jpeg)

**Caption:** Figure 1 presents the structural model of Nb3Sn in both cubic (a) and tetragonal (b) forms. This simplified crystal lattice serves as a basis for applying and observing strain effects, focusing on chain arrangements of Nb atoms and their impact on superconductive behavior. The tetragonal model specifically elucidates susceptibility to uniaxial strain, a key insight for improving Nb3Sn-based components in superconducting technologies.


<span id="page-0-0"></span>FIG. 1: The arrangement of atoms in the cubic (a) and tetragonal (b) phase of A15 Nb3Sn. The Nb atoms in the 2e sites form chains along the [001] direction, whereas those in the 4k sites are in chains along the [100] and [010] directions. To make the distortion clear, in (b), the cell is stretched to the abnormal value of ε = 40%, whereas the Poisson's ratio ν is set to 0.4.

![](_page_1_Figure_0.jpeg)

**Caption:** Figure 1 depicts the atomic configuration of A15 Nb3Sn in its cubic (a) and tetragonally distorted (b) phases. Under an exaggerated uniaxial strain representation (ε = 40%), the Poisson's ratio is set to 0.4, displaying the rearrangement of Nb and Sn atoms. This visualization exemplifies the structural evolution under strain, foundational for studying its implications on superconducting properties.


2

![](_page_1_Figure_1.jpeg)

**Caption:** Figure 2 illustrates the electronic band structure and corresponding density of states (DOS) for Nb3Sn under three strain states: +1.0%, 0.0%, and -1.0%. The Fermi level is set at 0 eV, highlighting that the tetragonal deformation does not significantly modify the electronic structure at the Fermi level. This constancy in electronic behavior across strains is crucial for understanding strain effects predominantly governed by phonon dynamics rather than electron-structural changes.


<span id="page-1-0"></span>FIG. 2: The electronic band structure and density of states for Nb3Sn calculated for three representative strain states: +1.0% (●), zero applied strain (◆), and -1.0% (▴). The Fermi level is set to 0 eV.

components is of basic importance. Recent high resolution Xray diffraction experiments on mechanically loaded samples[25](#page-6-3) have shown in detail how the Nb3Sn lattice cell deforms in the axial and the transverse directions; in particular, it was observed that the stress is completely transferred from the macroscopic level to the individual grains within the composite structure, so that a macroscopic uniaxial load directly corresponds to a stretching of the Nb3Sn lattice cell along the same direction, with the cell contracting in the transverse direction of an amount corresponding to a Poisson's ratio ν equal to 0.38.

The structural and electronic properties of Nb3Sn have been theoretically studied by several groups[26–](#page-6-4)[31](#page-6-5), whereas the full phonon dispersion relations have been calculated by means of a tight-binding method[32](#page-6-6)[,33](#page-6-7) and - more recently - by an *ab initio* pseudopotential approach[34](#page-6-8). In particular, calculations by Tut¨ unc ¨ u¨ *et al.* give evidence of a strong interaction between the electronic states near the Fermi level and several phonon modes (longitudinal acoustic phonons and a group of optical phonon modes with average frequency of 4.5 THz) along the [111] direction.

However, to the best of our knowledge no systematic *ab initio* investigations have been made on studying the evolution of the band structure, phonon dispersion curves and superconducting parameters (electron-phonon mass enhancement parameter, λ, and T*c*) as a function of an applied uniaxial strain. In the present work, this issue has been addressed by employing the plane-wave pseudopotential method, the densityfunctional theory, and a linear-response technique[35](#page-6-9)[,36](#page-6-10), and by using the results by Tut¨ unc ¨ u¨ *et al.* for the undistorted cell as a starting baseline for our calculations.

## II. DETAILS OF CALCULATIONS AND COMPUTATIONAL METHOD

We used density functional theory and density-functional perturbation theory[36](#page-6-10) as implemented in the QUANTUM-ESPRESSO software distribution[37](#page-6-11), within the local-density approximation[38](#page-6-12), a plane-wave expansion up to 40 Ry for the kinetic energy cutoff and ultrasoft pseudopotentials for Nb and Sn[39](#page-6-13). The Brillouin zone has been sampled on a 8×8×8 Monkhorst-Pack (MP) mesh, corresponding to 126 special kpoints within the irreducible part of the Brillouin zone (IBZ). We also have checked more dense grids (up to 16×16×16) but the results did not change considerably.

Lattice dynamical calculations have been performed within the framework of the self-consistent density functional perturbation theory (DFPT)[36](#page-6-10), in which the dynamical matrices are calculated by sampling the IBZ with 8 independent q-points in the tetragonal phase. Dynamical matrices at any wave vectors can be Fourier deconvolved on this mesh, and the phonon dispersion curves along arbitrary symmetry directions can be easily obtained. In order to check the accuracy of the Fourier interpolation, we compared the results of this procedure with direct calculations on selected q-points not present in the grid.

A denser grid of k-points (24×24×24 MP divisions) has been used in order to determine the electron-phonon interac-

3

![](_page_2_Figure_1.jpeg)

**Caption:** Figure 3 shows (a) phonon dispersion curves for Nb3Sn at various strains: +1.0%, 0.0%, and -1.0%, alongside the (b) density of states for phonons at each strain level. The inclusion of experimental curves in the plots validates the theoretical model assumptions of phononic transformations under strain, which critically affect the superconducting behavior. These findings underscore the reliance on phonon dynamics for understanding strain-based modifications in superconducting materials.


<span id="page-2-1"></span>FIG. 3: (a) Phonon dispersion curves of Nb3Sn at three different strain states: +1.0% (●), zero applied strain (◆), and -1.0% (▴); (b) phonon DOS at each calculated strain. For completeness, experimental curves are also reported.

tion parameter λ, calculated as the Brillouin-zone average of the mode-resolved coupling strengths λ<sup>q</sup> *<sup>j</sup>* :

$$
\lambda = \sum\_{\mathbf{q}j} W(\mathbf{q}) \lambda\_{\mathbf{q}j} \tag{1}
$$

where *j* indicates a phonon polarization branch, and W(q) are the weights associated with the phonon wavevectors q, normalized to 1 in the first Brillouin zone.

The lattice parameter of the cubic cell is set to 5.29 *A*˚ [40,](#page-6-14)[41](#page-6-15) . For an uniaxial stress along the *z*-direction (σ*z*) and under the assumption that the system is transversally isotropic, the strain state can be expressed as:

<span id="page-2-0"></span>
$$\begin{aligned} \mathbf{e}\_{\mathbf{x}} &= \mathbf{e}\_{\mathbf{y}} = -\mathbf{v} \frac{\mathbf{o}\_{\mathbf{z}}}{E} \\ \mathbf{e}\_{\mathbf{z}} &= \frac{\mathbf{o}\_{\mathbf{z}}}{E} \end{aligned} \tag{2}$$

which reflects the variation of the lattice parameters of the tetragonally distorted cell. In Eq. [2,](#page-2-0) ν is the Poisson's ratio whereas *E* represents the Young's modulus. In our computations, ν has been set to the value measured in composite wires (ν = 0.4[25](#page-6-3)), and the distortions have been calculated in the strain range ± 1.0%, in steps of 0.2%.

#### III. RESULTS AND DISCUSSION

The Nb3Sn cubic phase belongs to the (*P* 42 *m* 3¯ 2 *n* ) space group and the *O* 3 *h* point-group symmetries, as shown in Fig[.1\(](#page-0-0)a). The Sn atoms are situated on a bcc matrix whereas the faces of the cube are occupied by Nb atoms which form three sets of orthogonal chains along the principal axes. When a uniaxial strain is applied along the c-direction, the lattice is tetragonally distorted, with the Nb-chains in the [001] direction differing from those in the [100] and [010] directions, as shown in Fig. [1\(](#page-0-0)b). The distorted structure has a reduced symmetry *D* 9 4*h* (*P* 42 *m* 2 *m* 2 *c* ).

Starting from the cubic cell, a uniaxial strain has been applied to the cell along the *c*-direction, according to Eq. [\(2\)](#page-2-0). As a result, the deviatoric components of the strain lowered the system's point group symmetries: most of the phonon degeneracies have been removed and a changement in both the electronic and phonon dispersion bands have been induced.

![](_page_3_Figure_0.jpeg)

**Caption:** Figure 4 illustrates the effect of applied uniaxial strain on (a) the superconducting critical temperature, T_c; (b) the electron-phonon coupling constant, λ; and (c) the logarithmically averaged phonon frequency, ω_ln. Each parameter shows a characteristic parabolic behavior with respect to strain, peaking at zero strain. This indicates that the superconducting properties of Nb3Sn are highly sensitive to strain, which is significant for understanding strain effects in practical superconducting applications, particularly in multi-filamentary Nb3Sn wires where strain optimization can enhance critical performance metrics.


<span id="page-3-0"></span>FIG. 4: The behavior of: (a) the superconducting critical temperature, *Tc*; (b) the *el-ph* coupling, λ; and (c) the logarithmically averaged phonon frequency ω*ln* as a function of an applied uniaxial strain. Lines are guide for eye.

#### A. Electronic band structures and phonon dispersion curves

The calculated electronic structure along many highsymmetry directions of the simple-cubic Brillouin zone are displayed in Fig. [2](#page-1-0) for three representative values of the applied strain (zero, 1.0% and -1.0%). The energy bands of the cubic crystal are shown as diamonds, whereas the circles and the triangles represents the 1.0% and -1.0% strain states, respectively. Indeed, the tetragonal deformation does not affect the electronic structure in a severe way: the energy bands of the distorted and undistorted cell are almost unchanged, and there is no evident splitting of the cubic bands at the Fermi level, *EF*. The electronic DOS is also not drastically affected by the tetragonal distortion. In Fig. [2,](#page-1-0) the Fermi level is marked by a dashed horizontal line and is set to 0 eV. It is

![](_page_3_Figure_4.jpeg)

**Caption:** Figure 5 plots the density of states at the Fermi level, N(E_F), as a function of applied strain. The curve is superimposed with λω²_ln, a product function expected to correlate with N(E_F). The good agreement between the two underscores the co-influence of strain on both electronic and phononic aspects of the material's superconductivity, supporting the idea that strain modulations similarly influence electronic density and phonon frequency products.


<span id="page-3-1"></span>FIG. 5: Density of states at Fermi level, *N*(*E<sup>F</sup>* ) calculated as a function of strain. The curve is overlapped to the product function λω<sup>2</sup> *ln*, which in good approximation should be proportional to *N*(*E<sup>F</sup>* ) [42](#page-6-16) .

interesting to notice that *E<sup>F</sup>* falls close to a sharp peak in the electronic DOS[27](#page-6-17), with a value for the density of states of the order of 20 states/eV. This peak is generated by several nearly dispersionless bands crossing the Fermi level in the Γ − *M*, Γ−*R*, and *M* −*R* directions and deriving from the *4d* states of Nb atoms[34](#page-6-8) .

The phonon dispersion curves, calculated along several high-symmetry directions in the Brillouin zone, are plotted for three representative strains in Fig. [3;](#page-2-1) the phonon DOS is depicted in the right panel. There is good agreement with previous published results[34](#page-6-8) at ε *= 0*. For the sake of completeness, some inelastic neutron scattering measurements[43](#page-6-18)[–45](#page-6-19) are also reported in Fig. [3a](#page-2-1), and again a good agreement is obtained.

### B. Derivation of superconducting *Tc* as a function of strain

The modifications of the phonon dispersion curves induced by a tetragonal distortion should have a strong effect on *T<sup>c</sup>* through the strain sensitivities of the averaged phonon frequencies ω*ln*, and of the *el-ph* coupling λ. Therefore, both ω*ln*(ε) and λ(ε) have been explicitly calculated, and *Tc*(ε) has been estimated by means of the Allen-Dynes modification of the McMillan formula[42,](#page-6-16)[46](#page-6-20):

$$T\_c = \frac{\hbar \mathbf{o}\_{ln}}{1.20} e^{-\frac{1.04\left(1+\lambda\right)}{\lambda - \mu^\* \left(1+0.62\lambda\right)}}\tag{3}$$

where *µ* ∗ is the effective Coulomb-repulsion parameter which describes the interaction beetween electrons, and ω*ln* is a weighted logaritmically averaged phonon frequency, defined as:

$$\mathfrak{so}\_{\mathrm{fl}t} = e^{\frac{2}{\lambda} \int\_0^{+\infty} \frac{\mathrm{d}\mathfrak{a}}{\mathrm{a}} \mathrm{a}^2(\mathrm{a}) F(\mathrm{a}) \ln \mathrm{a}} \tag{4}$$

where α 2 (ω)*F*(ω) is the Eliashberg spectral function. We assumed a negligible strain dependence of *µ* ∗ compared to the other parameters, and frozen it as a constant in our computations[19](#page-5-15). Our results are reported in Fig. [4a](#page-3-0)-c. As far as the cubic phase is concerned, our finding are in good agreement with the ones by Tut¨ unc ¨ u¨ *et al.*[34](#page-6-8): a group of six phonon modes at the R-point (whose averaged phonon frequency is approximately 140 cm−<sup>1</sup> ) are found to strongly interact with the *p-d* electronic states near the Fermi level. In these modes only the Nb chains vibrate, the Sn atoms being frozen at their equilibrium positions. The λ<sup>q</sup> *<sup>j</sup>* corresponding to these modes are comprised in the range 0.134−0.197. The overall electronphonon interaction parameter (λ = 1.85) agrees with the experimentally measured value[47](#page-6-21), and - choosing *µ* <sup>∗</sup> = 0.25 - the estimated critical temperature for the strain-free state is *T<sup>c</sup>* = 18.3 K (also in agreement with the highest reported *T<sup>c</sup>* [48](#page-6-22)).

As it can be clearly seen in Figs. [4b](#page-3-0) and [4c](#page-3-0), both ω*ln*(ε) and λ(ε) show a parabolic profile as a function of strain. Very close to the cubic phase (strain-free cell), ω*ln* has a maximum, implying a softening of the logarithmically averaged phonon frequencies when the system undergoes a distortion. The same behavior is found for λ, whose maximum is ∼ 1.85. The strength of the *el-ph* interaction weakens as ∣ε∣ increases. As a result, by varying the axial strain the *T<sup>c</sup>* curve assumes the characteristic bell shape (Fig. [4a](#page-3-0))[49,](#page-6-23)[50](#page-6-24). In addition, the curve shows a slight asymmetry with respect to the maximum, due mainly to an asymmetry in the phononic contribution (ω*ln*). However, a clear confirmation of this would deserve a more detailed analysis, based on an increased density of points on the curve. Qualitatively, the *T<sup>c</sup>* vs. ε curve reproduces the experimental strain sensitivity of the critical current found in A15 superconductors: as it is well known, when a Nb3Sn multifilamentary wire is subjected to a longitudinal strain, its critical current shows a maximum at zero intrinsic strain, and decreases reversibly with the applied load (see for example Ref. [51](#page-6-25) and references therein), with a slight asymmetry[52](#page-6-26) between the compressive and tensile sides. In this sense, these firstprinciple calculations suggest that the origin of such strain sensitivity in Nb3Sn is intrinsic and microscopic in its nature.

Our calculations also show that *N*(*EF*) is influenced by strain. This quantity is related to the *el-ph* coupling constant and to the averaged phonon frequency trhough the following expression[42](#page-6-16):

<span id="page-4-0"></span>
$$N(E\_F) = M < I^2 > \text{co}^2\_{RMS} \lambda \tag{5}$$

in which < *I* <sup>2</sup> > is the average over the Fermi surface of the *el-ph* matrix element squared, ω*RMS* is a weighted RMS phonon frequency and *M* is the average ionic mass. By further assuming that the strain sensitivity of the normalized averaged frequencies ω*RMS* and ω*ln* are the same[54](#page-6-27) and that < *I* <sup>2</sup> > does not depend on any applied strain, it follows that *N*(*EF*,ε) ∝ λω<sup>2</sup> *ln*. The strain dependence of *N*(*EF*), calculated either directly or through Eq. [5](#page-4-0) are consistent with one another, as can be clearly seen in Fig. [5.](#page-3-1) Results of Figs[.4](#page-3-0) and [5](#page-3-1) thus provide evidence that the strain is affecting both the phononic and the electronic properties in the same way. Indeed, existing studies consider only the two extreme cases

![](_page_4_Figure_5.jpeg)

**Caption:** Figure 6 compares theoretical predictions (dashed line) and experimental measurements (line and markers) of the strain function, s(ε), for Nb3Sn wires. Both data sets exhibit a bell-shaped curve; however, discrepancies are noted, particularly in strain sensitivity which is greater in the experimental composite wire due to intrinsic or extrinsic factors. This highlights the complexities involved in replicating theoretical models in real-world conditions, crucial for improving strain tolerance in technological superconducting wires.


<span id="page-4-1"></span>FIG. 6: Direct comparison between the theoretical (dashed line) and experimental (line and markers) strain function[53](#page-6-28). The theoretical *s*(ε) correctly reproduces a bell shape, the mismatch being attributed to those extrinsic effects generally observed in technological wires.

where either the lattice deformations or the electronic properties modifications are considered as a source for the strain sensitivity. Markiewicz[23](#page-6-0)[,24,](#page-6-2)[55](#page-6-29) has attempted to correlate the microscopic full invariant analysis with the Elisahberg-based relations for *T<sup>c</sup>* through strain-induced modifications in the electron-phonon spectrum. In this model, however, the strain induced changes in *N*(*EF*) are only accounted for through a strain-modified frequency dependence of the *el-ph* interaction. In other words, the changes in *N*(*EF*) are not directly calculated, although the model is sufficiently accurate. Furthermore, microscopical theoretical predictions by Taylor and Hampshire[19](#page-5-15) and Markiewicz[55](#page-6-29) have shown that the variation of the superconducting properties of Nb3Sn multifilamentary wires submitted to uniaxial strain are correlated to changes of the phonon spectrum, rather than to the electronic density of states. On the other side, many works[54](#page-6-27)[,56,](#page-6-30)[57](#page-6-31) have linked the superconducting properties of A15 compounds to the variations in the electronic properties and *N*(*EF*).

From the experimental point of view, it is important to highlight new methods for extracting the electron DOS from resistivity data have been explored with the aim to study whether the strain sensitivity is correlated to the electronic modifications[58](#page-6-32) .

However, according to our results, any model aiming to describe the superconducting properties of A15 compounds from microscopic theories should take both contributions into account.

#### C. Comparison with experimental results

Starting from the pioneering work of Ekin[18](#page-5-14), in many models available in literature[18](#page-5-14)[–23](#page-6-0) the strain sensitivity of Nb3Sn is generally parameterized using the strain function, s(ε), defined as[18](#page-5-14)[,59](#page-6-33):

<span id="page-5-16"></span>
$$s(\mathfrak{e}) \doteq \frac{B\_{c2}(\mathfrak{e}, T = 0)}{B\_{c2}(0, T = 0)} = \left[\frac{T\_c(\mathfrak{e})}{T\_c(0)}\right]^w \tag{6}$$

where *Bc*2(ε,*T*) represents the superconducting upper critical field, depending on strain and temperature, and where *w* ≈ 3 for A15 materials[18](#page-5-14) .

The strain function calculated using Eq. [6](#page-5-16) is plotted in Fig. [6](#page-4-1) (dashed line) and compared with the curve extracted from experimental *Icvs*.ε measurements[53](#page-6-28) on a technological, multifilamentary wire (lines and markers in the figure). Although both curves show the expected bell shape, the experimental and theoretical curves are quite different. In particular, the strain sensitivity of the composite wire is enhanced when compared to the theoretical prediction of a bulk, perfectly binary and stoichiometric Nb3Sn system, and this remains valid also if the value of *w* is increased (within physical accepted ranges). This is not surprising, and the difference might have either an intrinsic or extrinsic origin. Among extrinsic phenomena inducing performance degradation with strain in technological wires[60](#page-6-34), filament breakage, reduction in wire's cross-sectional area, stress-induced martensitic formation[53](#page-6-28) , and microcrack/defect formation in the superconductor might play a role. The reversibility of the experimental data plotted in Fig. [6](#page-4-1) implies that some of these phenomena can be ruled out, *e.g.* filament breakage, microcrack and extended defect formation. All others mentioned above, being reversible, can in principle explain the strain sensitivity in Nb3Sn, but their effect over *I<sup>c</sup>* is small and cannot account for the observed behavior[60](#page-6-34). As far as intrinsic mechanisms are considered, our calculations unambiguously show that strain sensitivity in Nb3Sn is associated to lattice and electronic deformations, which result in shifts in the Nb3Sn critical surface.

It should be underlined the fact that in this first principles study we have neglected any sublattice displacements of the Nb atoms[27](#page-6-17) leading to Nb-chains dimerization. This can also have an effect on the theoretical *s*(ε). In our calculations, Nb atoms are frozen in their ideal positions, and the cubic cell is stable with respect to a tetragonal strain[27](#page-6-17). However, if the Nb atoms are allowed to relax, a dimerization of the chains occurs, and the undistorted cubic structure become unstable with respect to a spontaneous sublattice distortion. However, such distortions are small (the ratio between the major and minor axis of the tetragonal cell spans from 0.9938 to 0.9964[27](#page-6-17)) and therefore their effect on *s*(ε) is expected to be negligible. Also, our system is perfectly binary, whereas in technological wires Ti or Ta is inserted as the ternary element, with the aim to improve the pinning efficiency and therefore *Jc*(*B*). Moreover, due to compositional inhomogeneities, a distribution of the superconducting properties is observed (*T<sup>c</sup>* depends on the atomic Sn content, having a maximum at Sn% = 0.25%); see for example Refs. [61](#page-6-35) and [62](#page-6-36) and reference therein. Considering all these aspects, it is clear that the polycrystalline Nb3Sn formed by a reaction heat treatment inside a composite, multifilamentary system has a microscopic structure that inevitably deviates from that of an ideal Nb3Sn lattice cell. Therefore, differences between the theoretical computations of *s*(ε) and the experimental degradation in wires can be expected, which might possibly suggest paths towards the improvement of the strain tolerance in technological wires. For example, at 0.5% compressive strain the strain sensitivity could in principle be reduced to ∼ 0.07%, thus helping in the design of those devices where the levels of strain experienced by Nb3Sn are sufficiently large, as is the case of high-field superconducting magnets.

#### Acknowledgments

We thank Federico Quagliata and Pietro D'Angelo for setting up the environment and the CRESCO parallel cluster at ENEA C.R. Frascati. M.B.N. wishes to acknowledge partial support from the Office of Basic Energy Sciences, U.S. Department of Energy at Oak Ridge National Laboratory under Contract No. DE-AC05-00OR22725 with UT-Battelle, LLC.

<span id="page-5-0"></span>∗ [demarzi@enea.it](mailto:demarzi@enea.it)

- <span id="page-5-1"></span><sup>1</sup> B. T. Matthias, Phys. Rev. 92, 874 (1953).
- <span id="page-5-2"></span><sup>2</sup> T. Miyazaki, T. Hase, and T. Miyatake, *Handbook of Superconducting Materials* (Bristol: IOP Publishing, 2003), pp. 639–72.
- <span id="page-5-3"></span><sup>3</sup> A. Vostner and E. Salpietro, Supercond. Sci. Technol. 19, S90 (2006).
- <sup>4</sup> N. Mitchell, A. Devred, P. Libeyre, B. Lim, S. F., and the ITER Magnet Division, IEEE Trans. on Appl. Supercond. 22, 4200809 (2012).
- <span id="page-5-4"></span><sup>5</sup> A. Devred, I. Backbier, D. Bessette, G. Bevillard, M. Gardner, M. Jewell, N. Mitchell, I. Pong, and A. Vostner, IEEE Trans. on Appl. Supercond. 22, 4804909 (2012).
- <span id="page-5-5"></span><sup>6</sup> H. Wada and T. Kiyoshi, IEEE Trans. Appl. Supercond. 12, 715 (2002).
- <span id="page-5-6"></span><sup>7</sup> L. Bottura, G. de Rijk, L. Rossi, and E. Todesco, IEEE Trans. on Appl. Supercond. 22, 4002008 (2012).
- <span id="page-5-7"></span><sup>8</sup> G. Rupp, J. Appl. Phys. 48, 3858 (1977).
- 9 J. Ekin, IEEE Trans. Magn. 15, 197 (1979).
- <sup>10</sup> B. ten Haken, Ph.D. Thesis, Technical University of Twente, Enschede, The Netherlands (1994).
- <span id="page-5-8"></span><sup>11</sup> J. Ekin, J. Appl. Phys. 62, 4829 (1987).
- <span id="page-5-9"></span><sup>12</sup> M. O. Hoenig, A. G. Montgomery, and S. J. Waldman, IEEE Trans. on Mag. 15, 792 (1979).
- <sup>13</sup> P. Bruzzone, IEEE Trans. Appl. Supercond. 16, 839 (2006).
- <span id="page-5-10"></span><sup>14</sup> M. Spadoni, IEEE Trans. on Magn. 30, 1699 (1994).
- <span id="page-5-11"></span><sup>15</sup> A. Nijhuis and Y. Ilyin, Supercond. Sci. Technol. 19, 945 (2006).
- <span id="page-5-12"></span><sup>16</sup> N. Mitchell, Supercond. Sci. Technol. 18, S396 (2005).
- <span id="page-5-13"></span><sup>17</sup> G. Mondonico, B. Seeber, A. Ferreira, B. Bordini, L. Oberli, L. Bottura, A. Ballarino, R. Flukiger, and C. Senatore, Supercond. ¨ Sci. Technol. 25, 115002 (2012).
- <span id="page-5-14"></span><sup>18</sup> J. W. Ekin, Cryogenics 20, 611 (1980).
- <span id="page-5-15"></span><sup>19</sup> D. M. Taylor and D. P. Hampshire, Supercond. Sci. Technol. 18,

S241 (2005).

- <span id="page-6-1"></span><sup>20</sup> S. Oh and K. Kim, J. Appl. Phys. 99, 0330909 (2006).
- <sup>21</sup> A. Godeke, B. ten Katen, H. H. J. ten Kate, and D. C. Larbalestier, Supercond. Sci. Technol. 19, R100 (2006).
- <sup>22</sup> D. Arbelaez, A. Godeke, and S. Prestemon, Supercond. Sci. Technol. 22 (2009).
- <span id="page-6-0"></span><sup>23</sup> W. D. Markiewicz, Cryogenics 46, 864 (2006).
- <span id="page-6-2"></span><sup>24</sup> W. D. Markiewicz, Cryogenics 44, 767 (2004).
- <span id="page-6-3"></span><sup>25</sup> L. Muzzi, V. Corato, A. della Corte, G. D. Marzi, T. Spina, J. Daniels, M. D. Michiel, F. Buta, G. Mondonico, B. Seeber, et al., Supercond. Sci. Technol. 25, 054006 (2012).
- <span id="page-6-4"></span><sup>26</sup> B. M. Klein and Z. Lu, Physica B: Condensed Matter 296, 120 (2001).
- <span id="page-6-17"></span><sup>27</sup> B. Sadigh and V. Ozolin¸s, Phys. Rev. B ˇ 57, 2793 (1998).
- <sup>28</sup> Z. W. Lu and B. M. Klein, Phys. Rev. Lett. 79, 1361 (1997).
- <sup>29</sup> L. F. Mattheiss and W. Weber, Phys. Rev. B 25, 2248 (1982).
- <sup>30</sup> B. M. Klein, L. L. Boyer, D. A. Papaconstantopoulos, and L. F. Mattheiss, Phys. Rev. B 18, 6411 (1978).
- <span id="page-6-5"></span><sup>31</sup> L. F. Mattheiss, Phys. Rev. B 12, 2161 (1975).
- <span id="page-6-6"></span><sup>32</sup> W. Weber, Physica B+C 126, 217 (1984), ISSN 0378-4363.
- <span id="page-6-7"></span><sup>33</sup> W. Weber, *Electronic Structure of Complex Systems* (Plenum Press, New York, 1984), vol. 113, p. 345.
- <span id="page-6-8"></span><sup>34</sup> H. M. Tut¨ unc ¨ u, G. P. Srivastava, S. Ba ¨ gc ı, and S. Duman, Phys. ˘ Rev. B 74, 212506 (2006).
- <span id="page-6-9"></span><sup>35</sup> S. Baroni, P. Giannozzi, and A. Testa, Phys. Rev. Lett. 58, 1861 (1987).
- <span id="page-6-10"></span><sup>36</sup> S. Baroni, S. de Gironcoli, A. Dal Corso, and P. Giannozzi, Rev. Mod. Phys. 73, 515 (2001).
- <span id="page-6-11"></span><sup>37</sup> QUANTUM-ESPRESSO is a community project for high-quality quantum-simulation software, based on density-functional theory, and coordinated by Paolo Giannozzi. See http://www.quantumespresso.org and http://www.pwscf.org.
- <span id="page-6-12"></span><sup>38</sup> N. Troullier and J. L. Martins, Phys. Rev. B 43, 1993 (1991).
- <span id="page-6-13"></span><sup>39</sup> We used the pseudopotentials Nb.pw91-nsp-van.UPF and Sn.pw91-n-van.UPF from the http://www.quantum-espresso.org distribution.
- <span id="page-6-14"></span><sup>40</sup> R. G. Maier, Z. Naturforsch. Teil A 24, 1033 (1969).
- <span id="page-6-15"></span><sup>41</sup> V. Guritanu, W. Goldacker, F. Bouquet, Y. Wang, R. Lortz, G. Goll, and A. Junod, Phys. Rev. B 70, 184526 (2004).
- <span id="page-6-16"></span><sup>42</sup> W. L. McMillan, Phys. Rev. 167, 331 (1968).
- <span id="page-6-18"></span><sup>43</sup> L. Pintschovius, H. Takei, and N. Toyota, Phys. Rev. Lett. 54, 1260 (1985).
- <sup>44</sup> J. D. Axe and G. Shirane, Phys. Rev. B 8, 1965 (1973).
- <span id="page-6-19"></span><sup>45</sup> G. Shirane and J. D. Axe, Phys. Rev. B 18, 3742 (1978).
- <span id="page-6-20"></span><sup>46</sup> P. B. Allen and R. C. Dynes, Phys. Rev. B 12, 905 (1975).
- <span id="page-6-21"></span><sup>47</sup> E. L. Wolf, J. Zasadzinski, G. B. Arnold, D. F. Moore, J. M. Rowell, and M. R. Beasley, Phys. Rev. B 22, 1214 (1980).
- <span id="page-6-22"></span><sup>48</sup> J. Hanak, K. Strater, and R. Cullen, RCA Review 25, 342 (1964).
- <span id="page-6-23"></span><sup>49</sup> R. Flukiger, R. Isernhage, W. Goldhacker, and W. Specking, Adv. ¨ Cryo. Eng. (Materials) 30, 851 (1984).
- <span id="page-6-24"></span><sup>50</sup> J. W. Ekin, *Experimental Techniques for Low-Temperature Measurements* (New York: Oxford University Press, 2007).
- <span id="page-6-25"></span><sup>51</sup> A. Godeke, Ph.D. Thesis, Technical University of Twente, Enschede, The Netherlands (2005).
- <span id="page-6-26"></span><sup>52</sup> R. Flukiger, D. Uglietti, V. Ab ¨ acherli, and B. Seeber, Supercond. ¨ Sci. Technol. 18, S416 (2005).
- <span id="page-6-28"></span><sup>53</sup> G. D. Marzi, V. Corato, L. Muzzi, A. della Corte, G. Mondonico, B. Seeber, and C. Senatore, Supercond. Sci. Technol. 25, 025015 (2012).
- <span id="page-6-27"></span><sup>54</sup> K. C. Lim, J. D. Thompson, and G. W. Webb, Phys. Rev. B 27, 2781 (1983).
- <span id="page-6-29"></span><sup>55</sup> W. D. Markiewicz, Cryogenics 44, 895 (2004).
- <span id="page-6-30"></span><sup>56</sup> W. Weber and I. Goldberg (????).
- <span id="page-6-31"></span><sup>57</sup> B. M. Klein, L. L. Boyer, D. A. Papaconstantopoulos, and L. F. Mattheiss, Phys. Rev. B 18, 6411 (1978).
- <span id="page-6-32"></span><sup>58</sup> M. G. T. Mentink, M. M. J. Dhalle, D. R. Dietderich, A. Godeke, W. Goldacker, F. Hellman, and H. H. J. ten Kate, AIP Conference Proceedings 1435, 225 (2012).
- <span id="page-6-33"></span><sup>59</sup> D. O. Welch, Adv. Cryo. Eng. 26, 48 (1980).
- <span id="page-6-34"></span><sup>60</sup> J. Ekin, IEEE Trans. Magn. 13, 127 (1977).
- <span id="page-6-35"></span><sup>61</sup> C. Senatore, V. Abacherli, M. Cantoni, and R. Fl ¨ ukiger, Super- ¨ cond. Sci. Technol. 20, S217 (2007).
- <span id="page-6-36"></span><sup>62</sup> A. Godeke, Supercond. Sci. Technol. 19, R68 (2006).