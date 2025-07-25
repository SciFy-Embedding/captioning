# arXiv:1608.00175

**Paper ID:** 97bda32fc59754d6dfdfd6b737da38a6

**PDF Path:** apl/Superconductors/arXiv:1608.00175.pdf

**Processing Status:** complete

**Captions Added:** 17

**Generated:** 2025-06-24T13:44:27.118259

---

# Theoretical estimates of maximum fields in superconducting resonant radio frequency cavities: Stability theory, disorder, and laminates

Danilo B. Liarte,<sup>1</sup> Sam Posen,<sup>2</sup> Mark K. Transtrum,<sup>3</sup> Gianluigi Catelani,<sup>4</sup> Matthias Liepe,<sup>5</sup> and James P. Sethna<sup>1</sup>

<sup>1</sup>Laboratory of Atomic and Solid State Physics, Clark Hall,

Cornell University, Ithaca, New York 14853-2501

<sup>2</sup>Fermi National Accelerator Laboratory, Batavia, IL 60510, USA.

<sup>3</sup>Department of Physics and Astronomy, Brigham Young University, Provo, Utah 84602, USA

<sup>4</sup>Forschungszentrum J¨ulich, Peter Gr¨unberg Institut (PGI-2), 52425 J¨ulich, Germany

<sup>5</sup>LEPP, Physics Department, Newman Laboratory, Cornell University

Theoretical limits to the performance of superconductors in high magnetic fields parallel to their surfaces are of key relevance to current and future accelerating cavities, especially those made of new higher-Tc materials such as Nb3Sn, NbN, and MgB2. Indeed, beyond the so-called superheating field Hsh, flux will spontaneously penetrate even a perfect superconducting surface and ruin the performance. We present intuitive arguments and simple estimates for Hsh, and combine them with our previous rigorous calculations, which we summarize. We briefly discuss experimental measurements of the superheating field, comparing to our estimates. We explore the effects of materials anisotropy and the danger of disorder in nucleating vortex entry. Will we need to control surface orientation in the layered compound MgB2? Can we estimate theoretically whether dirt and defects make these new materials fundamentally more challenging to optimize than niobium? Finally, we discuss and analyze recent proposals to use thin superconducting layers or laminates to enhance the performance of superconducting cavities. Flux entering a laminate can lead to so-called pancake vortices; we consider the physics of the dislocation motion and potential re-annihilation or stabilization of these vortices after their entry.

#### I. INTRODUCTION

To transfer energy to beams of charged particles, accelerators frequently use superconducting radio-frequency (SRF) cavities, devices that are capable of sustaining large amplitude electromagnetic fields with relatively small input power. The energy gain of a beam traversing a cavity is determined by the electric field amplitude along its path—a larger amplitude can reduce the number of cavities required to reach a given energy. This is especially important in high energy accelerators, which call for as many as tens of thousands of cavities [\[1\]](#page-19-0). It is therefore of interest to understand the mechanisms that fundamentally limit the accelerating electric field. For stateof-the-art SRF cavities that have been carefully prepared to prevent non-fundamental degradation processes such as field emission [\[2,](#page-20-0) [3\]](#page-20-1) and multipacting [\[4\]](#page-20-2), studies show that the limit is not the electric field, but rather the interaction of the magnetic field with the superconducting material of the cavity walls. The fundamental limit to acceleration in SRF cavities is the superheating field Hsh, introduced in Section [\(II\)](#page-2-0).

This article will cover ideas, methods, and results revolving around the superheating field and its dependence on the superconductor – materials properties, anisotropy, defects and disorder, and laminates. The ideas and methods are primarily gleaned from the broader condensed matter community. In Section [\(II\)](#page-2-0) we review computations of Hsh for clean systems using field theories from the 1950's derived for pure superconductors near their transition temperature [\[5\]](#page-20-3); in Section [\(IV A\)](#page-7-0) we draw from more sophisticated theories from the 1960's to calculate Hsh at all temperatures [\[6\]](#page-20-4), and discuss the future need to use these historical theories to incorporate effects of strong coupling and electronic structure [\[7\]](#page-20-5) in new materials. In Section [\(IV B\)](#page-8-0) we review the use of these methods to address the electronic anisotropy of some of the new materials. In Section [\(IV C](#page-10-0) we introduce an illustrative calculation of the effects of disorder using tools and methods developed in the 60's for disordered systems [\[8,](#page-20-6) [9\]](#page-20-7) and nucleation theory [\[10,](#page-20-8) [11\]](#page-20-9), providing reassurance that new materials will likely not be far more sensitive to flaws and dirt. Finally, in Section [\(V](#page-13-0) we investigate the properties of superconducting laminates, by drawing from work from the 90's on the dynamics of 'pancake vortices' in certain layered hightemperature superconductors [\[12\]](#page-20-10) (particularly BSCCO, Bi2Sr2Ca<sup>n</sup>−<sup>1</sup>CunO2n+4+x).

We frankly have two goals for this article. As discussed above, we wish to provide an introduction for the accelerator community into tools and methods from the broader condensed matter community that can help interpret current experimental challenges and guide plans for future research in optimizing materials properties for SRF cavities. But conversely, we want to provide a window for the broader condensed matter theory community into the remarkable frontiers of field, frequency, and materials preparation being explored by the SRF community. We invite their participation in melding 21st century materials-by-design tools from electronic structure theory with 20th century field theories of superconductivity, bridging the scales to address current technological challenges in the accelerator field. (Full disclosure: this article was supported in part by the Center for Bright Beams, an NSF Science and Technology Center whose mission is precisely to bring the accelerator community

# A. Basic facts about superconductors: type I and II, Hc, Hc1, and Hc<sup>2</sup>

Normal conducting metals, such as copper, are not viable as radio-frequency cavities for long-pulse highgradient applications. Due to their high surface resistance, these cavities dissipate too much power on the walls, which can result in melting, among other structural problems, if they are not sufficiently cooled. When subject to high accelerating fields, copper cavities are limited to short-pulse applications. In contrast, superconducting radio-frequency cavities have a much lower surface resistance, which implies low dissipation on the walls and high quality factors (of about 1010, compared to 10<sup>4</sup> for copper) [\[13\]](#page-20-11). Taking into account the refrigerator power to keep the cavity in the superconductor state, SRF cavities are considerably more economical than copper cavities, and present huge benefits, especially for long-pulse applications. At high magnetic fields, however, high-temperature superconductor cavities can dissipate as much power as copper due to the nucleation and motion of vortices.

At low enough temperature and applied magnetic field (which for now we assume to be constant in time), superconductors exhibit the Meissner effect: magnetic fields are expelled from the interior of the superconductor, exponentially decaying from the interface surface. Larger applied magnetic field can destroy this Meissner state in two ways, depending on the type of superconductor. In type-I superconductors, an abrupt phase transition takes place at the thermodynamic critical field Hc, above which the superconductor is in the normal state. In type-II superconductors, the situation is slightly more complicated. Magnetic flux penetration starts, via vortex nucleation, at a lower magnetic field Hc<sup>1</sup> < Hc. Hc<sup>1</sup> is called the lower critical field. The transition to the normal phase takes place at the upper-critical field Hc<sup>2</sup> (Hc<sup>2</sup> > Hc). In the intermediate range, Hc<sup>1</sup> < H < Hc2, the system is in the vortex lattice state [1](#page-1-0) .

#### B. The superheating field

For these cavities during operation, the external magnetic field is parallel to the superconductor surface. In many applications, the threshold field for flux penetration onto the superconductor is not set by H<sup>c</sup> or Hc<sup>1</sup> (for type-I and type-II superconductors, respectively); it is set by the metastability limit of the Meissner state, i.e. by the superheating field [\[14–](#page-20-12)[26\]](#page-20-13). The Meissner state is metastable at H<sup>c</sup> < H < Hsh for type-I superconductors, and at Hc<sup>1</sup> < H < Hsh for type-II superconductors. The onset of instability of the Meissner state is related to the vanishing of a surface energy barrier that prevents field penetration onto the superconductor even when H > H<sup>c</sup> or H > Hc1.

The metastable Meissner state is analogous to the state of superheated water (perhaps explaining the name "superheating field"). Liquid water in a glass can be superheated in a microwave to a temperature above the liquidgas transition temperature, but still remain in the liquid state due to the surface tension barrier at the liquid-gas interface, causing small vapor bubbles to contract rather than grow. Surface tension in water is analogous, for instance, to the surface tension due to the energy barrier preventing vortex nucleation in type-II superconductors. Unlike the case of water, as we argue in Section [\(I D\)](#page-2-1), thermal nucleation of vortices occurs at relatively long time scales, suggesting that the Meissner state can be sustained in RF applications for fields as large as the superheating field. However, this scenario can considerably change when one considers the effects of disorder in the superconductor. Section [\(IV C\)](#page-10-0) discusses disorderinduced nucleation of vortices.

The superheating field is associated with spinodal curves where the local stability of the Meissner state is broken. This is a more precise definition that is useful for both type-I and type-II superconductors. We shall discuss calculations of the superheating field in Section [\(II\)](#page-2-0). Our calculations there will be assuming an external field that is constant in time and ignore thermal fluctuations. We here discuss these approximations.

#### C. Why GHz is slow

Calculations of the superheating field for DC applied magnetic fields will be accurate for RF applications when the microscopic relaxation times are smaller than the time scales that are associated with changes in the fields inside the cavity. Time scales for the latter are of order of nanoseconds [\[13\]](#page-20-11). A version of time dependent Ginzburg-Landau theory given by Gor'kov and Eliashberg predicts the characteristic relaxation time near Tc: τGL = π ~/[8 k (T<sup>c</sup> − T)], where ~ is the Planck constant divided by 2π and k is the Boltzmann constant [\[5\]](#page-20-3). For T<sup>c</sup> − T = 1K, one obtains τGL ∼ 10−<sup>3</sup> ns for oscillating fields parallel to the sample surfaces. Using ∆ ∼ k Tc, where ∆ is the superconductor gap, we find τGL ∼ ∆−<sup>1</sup> at low temperatures, which is similar to the scaling of collective modes in unconventional superfluids (see e.g. Section 23.5 of Ref. [\[27\]](#page-20-14)). However, note that Gor'kov and Eliashberg theory is applicable to gapless superconductors, filled with magnetic impurities and sufficient pairbreaking strength. For superconductors with a clean gap,

<span id="page-1-0"></span><sup>1</sup> At higher magnetic fields (> Hc2), surface superconductivity can persist up to a third critical field, Hc3. This critical field should not be mistaken by the superheating field, below which the system displays bulk superconductivity and field expulsion.

the relaxation time is expected to be larger than τGL, and to scale with the inelastic phonon-scattering time τE, which, near T<sup>c</sup> is of the order of ∼ 10−<sup>8</sup> s in Al and ∼ 10−<sup>11</sup> s for Pb [\[5\]](#page-20-3), due to its larger critical temperature [2](#page-2-2) . Yoo et al. measured an ultra fast electron-phonon relaxation time of 360 fs for niobium [\[28\]](#page-20-15). So, at GHz frequencies we may ignore the time dependence in studying the stability.

#### <span id="page-2-1"></span>D. Why thermal fluctuations are small

One key question for our purposes is whether thermal fluctuations can help activate vortices over the surface barrier. Thermal fluctuations in most superconductors (apart from the high-T<sup>c</sup> cuprate superconductors) are very small. This is due to the same approximation that makes the BCS theory of superconductors so successful. BCS theory is a mean-field theory of interacting Cooper pairs, which becomes exact when each Cooper pair interacts with an infinite number of neighbors (thus seeing the mean behavior of the system). Each Cooper pair is of radius roughly the coherence length ξ, so BCS theory will be valid when the density of Cooper pairs times ξ 3 is large. Simple estimates show that there are about 10<sup>6</sup> centers of Cooper pairs within the region occupied by each pair state; a scenario where the pairs strongly overlap in space, and each pair only feels the average occupancy of the other pair states [\[35\]](#page-20-16). Thermal fluctuations of vortices will be unimportant so long as the condensation energy density—the amount of energy F that is necessary to destroy superconductivity over a unit volume times ξ 3 , is large compared to kBT. Table [\(I\)](#page-3-0) gives the characteristic temperature Tth = F ξ<sup>3</sup>/k<sup>B</sup> where fluctuations will become important, for niobium and also three candidate materials being explored for next generation accelerating cavities. Only for NbN is this characteristic temperature remotely comparable to Tc.

We can gain further insight from an analytic calculation of Ev/(k<sup>B</sup> T), where E<sup>v</sup> is the energy per unit length of a vortex line integrated over a coherence length ξ. Using results from BCS theory, the zerotemperature thermodynamical critical field is given by <sup>H</sup>c(0) = 2 <sup>√</sup> π p N (0)∆, where N (0) = m k<sup>F</sup> /(2 π <sup>2</sup> ~ 2 ) is the density of states at the Fermi energy, ∆ is the superconductor gap at zero temperature, and k<sup>F</sup> is the Fermi wave number. Also, ∆ ≈ 1.76 k<sup>B</sup> Tc, and the coherence length ξ<sup>0</sup> = ~v<sup>F</sup> /(π∆), where v<sup>F</sup> is the Fermi velocity. Thus,

$$\frac{E\_v}{k\_B T} \sim \frac{H\_c^{\cdot 2} \xi^3}{k\_B T} \approx \frac{1.4}{t} \left(\frac{\varepsilon\_F}{\Delta}\right)^2,\tag{1}$$

where ε<sup>F</sup> = ~ <sup>2</sup> k<sup>F</sup> 2 /(2 m) is the Fermi energy, and t = T /Tc. Since the gap is much smaller than the Fermi energy, we can neglect thermal nucleation of vortices; unlike the case of superheated water, the effects of thermal fluctuations is very small. More generally, we expect that τmic τcav τt.n.v. within the Meissner metastable state, where τmic, τcav, and τt.n.v. correspond to time scales associated with microscopic degrees of freedom, the variation of the cavity fields, and thermal nucleation of vortices, respectively.

The negligible effects of thermal fluctuations tells us that estimating the limiting superheating field of a perfectly clean surface will not be analogous to bubble formation for superheated water. Instead, we shall use linear stability theory in Section [\(II B\)](#page-4-0) to estimate the field at which the uniform Meissner state becomes energetically unstable to an infinitesimal perturbation in the space of magnetic fields and superconducting order. A variant of critical droplet theory will appear in Section [\(IV C\)](#page-10-0), where we estimate the effects of flaws and disorder in nucleating vortex penetration.

## <span id="page-2-0"></span>II. BASIC THEORY OF THE SUPERHEATING FIELD

The superheating field Hsh is set by the competition between magnetic pressure (imposed by the external magnetic field), the energy cost to destroy superconductivity, and the attractive force due to the zero-current boundary condition at the interface. In Ginzburg-Landau theory, the ratio κ = λ/ξ of the penetration depth λ to the coherence length ξ determines many properties of superconductors. In particular, κ < 1/ √ 2 and κ > 1/ √ 2 are associated with type-I and type-II superconductivity, respectively. In the flux-line lattice of type-II superconductors, both the vortex supercurrent and magnetic field are confined to a tube of radius λ. The superconductivity is destroyed (the density of superconducting electrons vanishes) over a smaller vortex core of radius ξ. Within GL theory, Hsh(T)/Hc(T) depends on materials properties only through the parameter κ, which is independent of temperature. A careful calculation using linear stability analysis [\[14\]](#page-20-12) shows that Hsh plateaus at about 0.75H<sup>c</sup> in the large κ limit, and diverges as κ −1/2 for κ 1.

#### A. Simple arguments for the superheating field

We now give simple arguments and pictures to estimate the superheating field of superconductors (see e.g. [\[36\]](#page-20-17)). The main idea is to compute the work necessary to push magnetic field onto the superconductor through an energy barrier set by the magnetic energy, and compare the result with the condensation energy. It is worth noting that there are important qualitative differences between these simple arguments and the actual linear sta-

<span id="page-2-2"></span><sup>2</sup> A simple estimate given in Sec. 10.3 of Ref. [\[5\]](#page-20-3), assuming a Debye phonon spectrum and free-electron Fermi surface, gives τ<sup>E</sup> scaling as T<sup>c</sup> −3 .

| Material λ[nm] ξ[nm] |     |     |           |    |       |      |      | κ Tc[K] Hc1[T] Hc[T] Hsh[T] F[J/m3 | ] F ξ3<br>/kB<br>[K] |
|----------------------|-----|-----|-----------|----|-------|------|------|------------------------------------|----------------------|
| Nb                   | 40  | 27  | 1.5       | 9  | 0.13  | 0.21 | 0.25 | 17547                              | 25009.0              |
| Nb3Sn                | 111 | 4.2 | 26.4      | 18 | 0.042 | 0.5  | 0.42 | 99472                              | 533.6                |
| NbN                  | 375 |     | 2.9 129.3 | 16 | 0.006 | 0.21 | 0.17 | 17547                              | 31.0                 |
| MgB2                 | 185 | 4.9 | 37.8      | 40 | 0.017 | 0.26 | 0.21 | 26897                              | 229.1                |

<span id="page-3-0"></span>TABLE I. Representative material parameters for niobium, the traditional superconducting material used in SRF cavities, as well as candidate SRF materials that have the potential to reduce cooling costs due to their higher Tc. The coherence length ξ is calculated using equations in Ref. [\[29\]](#page-20-18). The penetration depth λ is calculated from Eq. 3.131 in Ref. [\[5\]](#page-20-3). The ratio κ = λ/ξ is called the Ginzburg-Landau parameter, and determines many properties of superconductors. A residual resistivity ratio of 100 was assumed for niobium. For MgB2, the values of λ and ξ are experimental values given in the reference. For calculations, <sup>H</sup><sup>c</sup> <sup>=</sup> <sup>φ</sup>0/[µ0(2<sup>√</sup> 2πξλ)] is used [\[5\]](#page-20-3). Hc<sup>1</sup> for Nb is found from fit to numerically computed data in Ref. [\[30\]](#page-20-19) and [\[31\]](#page-20-20). Hc<sup>1</sup> for strongly type II materials is found from Eq. 5.18 in Ref. [\[5\]](#page-20-3). Hsh is calculated using Hsh ' Hc(0.75 + 0.54 κ −1/2 ) [\[14\]](#page-20-12). The condensation energy density F is given by µ0H<sup>2</sup> <sup>c</sup> /2 [\[5\]](#page-20-3). Nb data is extracted from Ref. [\[32\]](#page-20-21), Nb3Sn data from Ref. [\[30\]](#page-20-19), NbN data from Ref. [\[33\]](#page-20-22), and MgB<sup>2</sup> data from Ref. [\[34\]](#page-20-23).

bility analysis of the GL free energy. We will return to these issues when we discuss the effects of anisotropy in Sec. [\(IV B\)](#page-8-0), and discuss them further in the full publication [\[36\]](#page-20-17).

![](_page_3_Figure_4.jpeg)

**Caption:** Illustrates the orientation of magnetic fields between vacuum and superconductors (SC), highlighting the influence of geometry on magnetic flux. The diagram contrasts simple SC block and a toroidal structure, emphasizing differing magneto-dynamic responses due to altered field alignment. Understanding such orientations assists in clarifying SC behavior under magnetic loads and designing devices with improved field penetration properties.


<span id="page-3-1"></span>FIG. 1. (On the left) Illustration of a superconductor occupying the half-space x > 0, and subject to an applied magnetic field H that is parallel to the z axis. "SC" stands for superconductor. (On the right) Approximate shape of a superconducting RF cavity in the regions of high magnetic fields. As in the flat case, the magnetic field that is generated by the accelerating beam (and excited by an external RF source, driving the operating/accelerating mode) is parallel to the interior surface of the cavity.

Consider a superconductor occupying the half-space x > 0, and subject to an applied magnetic field H that is parallel to its surface, along the direction z. We illustrate this geometry on the left side of Fig. [\(1\)](#page-3-1), where "SC" stands for superconductor. Note that the superconductor region extends to infinite in the positive and negative y and z directions, and in the positive x direction; there are no 'corners' in this geometry [3](#page-3-2) .

Let us start with the argument for the superheating field of a type-I superconductor. For small external magnetic fields, the order parameter does not vanish at the vacuum-superconductor interface. However, if we push a slab of magnetic field onto the superconductor (just enough to make the order parameter vanish at the interface), we will destroy superconductivity over a length scale of order ξ. The work per unit area that is necessary to push magnetic energy onto the superconductor is set by the magnetic pressure and the penetration length; it is given approximately by [Hsh/(4π)]Hsh λ in cgs units. To estimate the superheating field, we compare this work with the condensation energy per unit area [H<sup>c</sup> 2 /(8π)] ξ, resulting:

<span id="page-3-3"></span>
$$\frac{H\_{\rm sh}}{H\_c} \approx 2^{-1/2} \,\kappa^{-1/2} \,. \tag{2}$$

Equation [\(2\)](#page-3-3) should be compared with the small-κ limit of the exact result using Ginzburg-Landau theory [\[14\]](#page-20-12): Hsh/H<sup>c</sup> ≈ 2 <sup>−</sup>1/<sup>4</sup> κ −1/2 .

In type-II superconductors, field penetration occurs via vortex nucleation, and the superheating field is set by the magnetic pressure that is necessary to push a vortex through a surface barrier onto the superconductor[4](#page-3-4) . There are two steps to this penetration. First, the core of the superconducting vortex (of radius ∼ ξ) must penetrate into the surface, at a cost given by the core volume

<span id="page-3-2"></span><sup>3</sup> The absence of corners is an important limiting factor in our approach, for corners typically facilitate field penetration in real

samples of arbitrary shapes. Modern RF cavities have an approximate cylindrical shape in the region of high magnetic fields (see right side of Fig. [\(1\)](#page-3-1)), with no corners, so such geometric considerations become unimportant.

<span id="page-3-4"></span><sup>4</sup> Note that this argument is not related to Yogi's 'vortex line nucleation' [\[37,](#page-20-24) [38\]](#page-20-25) estimate of Hsh. The latter, developed to analyze impressive experimental data, was qualitatively incorrect [\[14\]](#page-20-12). In particular, its estimate for the metastable limit Hsh for large κ went below Hc1, which makes no sense. This misled the SRF field for years into ignoring the potential importance of higher κ materials.

times the condensation energy. Second, this newly penetrated vortex must fight past an attractive force toward the surface due to the boundary conditions at the surface, which is usually estimated [\[26\]](#page-20-13) by the attraction to an 'image vortex'. Below we discuss the superheating field estimated from the initial penetration of the vortex. (Bean and Livingston's original estimate [\[26\]](#page-20-13) of the superheating field starts (somewhat arbitrarily) at a distance x = ξ after this initial penetration, and focuses on the effects of the attractive longer-range force.)

Figure [\(2\)](#page-4-1) illustrates the penetration of a vortex core (red disk) onto a superconductor occupying the halfspace x > 0. The magnetic work per unit length to push the vortex core onto the superconductor is given approximately by the condensation energy (per unit length):

<span id="page-4-2"></span>
$$\frac{H\_{\rm sh}}{4\pi} \frac{\Phi\_0}{\pi \lambda^2} 4\lambda \xi \approx \frac{H\_c}{8\pi} \pi \xi^2,\tag{3}$$

where Hsh/(4 π) is the magnetic pressure, Φ<sup>0</sup> is the fluxoid quantum, πλ<sup>2</sup> is the vortex area in the xy plane, 4 λ ξ is approximately the area that is associated with the region of field penetration (area of the orange box in Fig. [\(2\)](#page-4-1); it is the amount of the area of the vortex that penetrates the superconductor when a vortex core is pushed inside), and πξ<sup>2</sup> is the area of the vortex core. Using Φ<sup>0</sup> = 2 <sup>√</sup> 2 πH<sup>c</sup> λ ξ in Eq. [\(3\)](#page-4-2):

$$\frac{H\_{\rm sh}}{H\_c} \approx \frac{\sqrt{2}}{32} \pi \approx 0.14,\tag{4}$$

independent of κ.

![](_page_4_Figure_6.jpeg)

**Caption:** Illustration of circular shapes showing relationships between superconducting cross-sections defined by penetration depth (λ) and coherence length (ξ). These diagrams are essential for understanding the magnetic field penetration and vortex core configurations within superconductors, pivotal in advancing the calculation models for superconducting materials and analyzing their field-dependent behaviors.


<span id="page-4-1"></span>FIG. 2. Illustrating the penetration of a vortex core into a type-II superconductor (from [\[36\]](#page-20-17)). We estimate the superheating field from the work necessary to push a vortex core a distance x ∼ ξ into the superconductor. The vortex then must fight past an attractive force to a depth x ∼ λ to destroy the Meissner state.

How does this estimate compare with the field estimated from the attractive force, and with the true answer? The true answer, given below in Section [\(II B\)](#page-4-0), is about five times larger: Hsh/H<sup>c</sup> ≈ 0.75. Bean and Livingston's estimate of the superheating field due to the attractive force to the image vortex is Hsh/H<sup>c</sup> = 0.71, of the same form as our estimate 0.14 but larger and closer to the true estimate. We present the calculation of the

field necessary to introduce the core primarily due to its simplicity, and also because it motivates our analysis of anisotropic superconductors in Section [\(IV B\)](#page-8-0).

One should think of these two contributions as being sequential rather than serial: first the core must penetrate, and then the vortex must fight the longer-range attraction to enter the bulk. (It is interesting and convenient that these two fields are of the same scale.) The GL calculation in Section [\(II B\)](#page-4-0) of course incorporates both the initial core penetration and the longer range attractive force, together with cooperative effects of multiple vortices entering at the same time.

Note that, while the field needed to push the vortex core into the superconductor is roughly comparable to that needed to push the vortex past the attractive longrange potential, the two contributions contribute very differently to the total energy barrier to flux penetration. Energy is force times distance: the two forces are comparable but the Bean-Livingston force acts on a scale longer by a factor κ = λ/ξ than our core nucleation, and will dominate the barrier height. Finally, note that in practice the dominant mechanisms for vortex nucleation that set the superheating field will not involve straight vortices penetrating all along their lengths (as in our calculation above) or, even more impressively, arrays of straight vortices cooperatively pushing their way through the surface barrier (Section [\(II B\)](#page-4-0) below). We expect that disorder and flaws (discussed in Section [\(IV C\)](#page-10-0)) will lead to localized intrusions of single vortex loops into the material (Fig. [\(8\)](#page-11-0)).

## <span id="page-4-0"></span>B. Linear stability calculation of the superheating field

In this section we have seen that the superheating field arises in a bulk superconductor due to the competing effects of magnetic pressure and the destruction of superconductivity. Using relatively simple arguments, we derived the qualitative dependence of this field on κ. We now describe a more rigorous calculation of the superheating field using a linear stability analysis. Linear stability analysis is commonly used in a variety of pattern formation problems[\[39](#page-20-26)[–44\]](#page-20-27). For type II superconductors, the transition from the Meissner state to the mixed state is triggered by fluctuations of a critical wavelength that spontaneously break the transverse symmetry of the bulk sample, which when coupled to the inhomogeneous depth dependence of the Meissner state, make the superheating transition a challenging application of this method. We here describe this calculation using the Ginzburg-Landau theory for concreteness, although the basic procedure could be extended to other theories as we discuss below. Our presentation follows closely the procedure described in [\[14\]](#page-20-12), however, the calculation has a long history in the literature[\[18](#page-20-28)[–25\]](#page-20-29).

The Ginzburg-Landau free energy for a superconductor occupying the half space x > 0 in terms of the magnitude of the superconducting order parameter f and the gaugeinvariant vector potential q is given by

$$\mathcal{F}[f, \mathbf{q}] = \int\_{x>0} d^3 r \left\{ \xi^2 (\nabla f)^2 + \frac{1}{2} (1 - f^2)^2 + f^2 \mathbf{q}^2 \right.$$

$$+ \left( \mathbf{H}\_a - \lambda \nabla \times \mathbf{q} \right)^2 \right\}, \tag{5}$$

where H<sup>a</sup> is the applied magnetic field (in units of √ 2Hc).

We take the applied field to be oriented along the zaxis H<sup>a</sup> = (0, 0, Ha), and the order parameter f = f(x) to depend only on the distance from the superconductors surface. Assuming that the order parameter is real and parameterizing the vector potential as q = (0, q(x), 0) fixes the gauge. The Ginzburg-Landau equations that extremize F with respect to f and q are

$$
\xi^2 f'' - q^2 f + f - f^3 = 0, \qquad \lambda^2 q'' - f^2 q = 0, \qquad (6)
$$

and with our choices H = λq<sup>0</sup> , where primes denote derivatives with respect to x. With appropriate boundary conditions[\[5,](#page-20-3) [14\]](#page-20-12) these equations can be solved numerically to characterize the Meissner state.

For a given solution (f, q) we next consider the second variation of F associated with small perturbations f → f + δf and q → q + δq given by

<span id="page-5-0"></span>
$$\delta^2 \mathcal{F} = \int\_{x>0} d^3 r \left\{ \xi^2 (\nabla \delta f)^2 + 4f \delta f \mathbf{q} \cdot \delta \mathbf{q} + f^2 \delta \mathbf{q}^2 \right. \tag{7}$$

$$(3f^2 + \mathbf{q}^2 - 1) \delta f^2 + \lambda^2 (\nabla \times \delta \mathbf{q})^2 \right\}. \tag{7}$$

If the expression in Eq. [\(7\)](#page-5-0) is positive for all possible perturbations, then the solution is (meta) stable. Since the solution (f, δq) depends only on the distance from the boundary (and is therefore translationally invariant along the y and z directions), we can expand the perturbation in Fourier modes parallel to the surface. As shown in Ref. [\[18\]](#page-20-28), we can restrict our attention to perturbations independent of z and write

<span id="page-5-1"></span>
$$\begin{aligned} \delta f(x, y) &= \delta \tilde{f}(x) \cos ky, \\ \delta \mathbf{q}(x, y) &= (\delta \tilde{q}\_x \sin ky, \delta \tilde{q}\_y \cos ky, 0), \end{aligned} \tag{8}$$

where k is the wave-number of the Fourier mode. The remaining Fourier components (corresponding to replacing cos → sin and vice-versa in Eq. [\(8\)](#page-5-1)) are redundant as they decouple from those given in Eq. [\(8\)](#page-5-1) and satisfy the same differential equations derived below.

After substituting into the expression [\(7\)](#page-5-0) for the second variation and integrating by parts, we arrive at

$$\delta^2 \mathcal{F} = \int\_0^\infty dx \left( \delta \vec{f} \cdot \delta \vec{q}\_y \cdot \delta \vec{q}\_x \right) \begin{pmatrix} -\xi^2 \frac{d^2}{dx^2} + q^2 + 3f^2 + \xi^2 k^2 - 1 & 2fq & 0\\ 2fq & -\lambda^2 \frac{d^2}{dx^2} + f^2 & -\lambda^2 k \frac{d}{dx} \\ 0 & \lambda^2 k \frac{d}{dx} & f^2 + \lambda^2 k^2 \end{pmatrix} \begin{pmatrix} \delta \vec{f} \\ \delta \vec{q}\_y \\ \delta \vec{q}\_x \end{pmatrix}. \tag{9}$$

This matrix operator is self-adjoint, and the second variation will be positive definite if its eigenvalues are all positive. In the eigenvalue equations for this operator, the function δq˜<sup>x</sup> can be solved for algebraically. The resulting differential equations for δ ˜f and δq˜<sup>y</sup> are

<span id="page-5-2"></span>
$$-\xi^2 \delta \bar{f}'' + (3f^2 + q^2 - 1 + \xi^2 k^2) \delta \bar{f} + 2fq \delta \bar{q}\_y = E \delta \bar{f},\tag{10}$$

and

<span id="page-5-3"></span>
$$-\lambda^2 \frac{d}{dx} \left[ \frac{f^2 - E}{f^2 + \lambda^2 k^2 - E} \delta \tilde{q}\_y' \right] + f^2 \delta \tilde{q}\_y + 2fq \delta \tilde{f} = E \delta \tilde{q}\_y,\tag{11}$$

where E is the stability eigenvalue. Note that by decomposing in Fourier modes, the two-dimensional problem is transformed into a one-dimensional eigenvalue problem. Numerically, it can be solved by the same methods as the Ginzburg-Landau equations[\[14\]](#page-20-12).

The stability eigenvalue will depend on the solution of the Ginzburg-Landau equations, i.e., the applied magnetic field Ha, and the Fourier mode k under consideration. The superheating field is found by varying both the applied magnetic field and Fourier mode until the smallest eigenvalue first becomes negative. The wave-number of the destabilizing fluctuations are therefore found simultaneously with Hsh and denoted by kc. Values of Hsh and k<sup>c</sup> were calculated in Ginzburg-Landau theory for a wide range of κ in references[\[14,](#page-20-12) [25\]](#page-20-29) along with analytic estimates. The results are summarized in Figure [\(3\)](#page-6-0).

For small κ, the critical fluctuation occurs with wavenumber k<sup>c</sup> = 0 while for large κ, k<sup>c</sup> > 0. Interestingly, the transition to nonzero k<sup>c</sup> occurs at some critical κ<sup>c</sup> that is distinct from the type-I/type-II boundary (κ = 1/ √ 2). Estimates of κ<sup>c</sup> vary in the literature from 0.5[\[18\]](#page-20-28) to 1.13(±0.05)[\[23\]](#page-20-30). Estimates of κ<sup>c</sup> from solving Eqs. [\(10\)](#page-5-2)and [\(11\)](#page-5-3) range from 1.10[\[22\]](#page-20-31) to 1.1495[\[14\]](#page-20-12) (our high-accuracy result).

The linear stability approach described in this section could be extended to other geometries as was done for the case of a superconducting film separated from a bulk su-

![](_page_6_Figure_0.jpeg)

**Caption:** The plot depicts numerically derived versus analytically approximated values of the superheating field in Ginzburg-Landau theory. The comparison across diverse values of Ginzburg-Landau parameter kappa reveals significant alignment between the different approaches, emphasizing reliability in extrapolating GL theory to practical scenarios in type-II superconductors.


<span id="page-6-0"></span>FIG. 3. Superheating Field in Ginzburg-Landau Theory (from [\[14\]](#page-20-12)). (a) A numerical estimate of Hsh in Ginzburg-Landau theory over many orders of magnitude of κ was found in reference[\[14\]](#page-20-12) (black solid line), along with a large-κ expansion (red dashed line). A Pad´e approximation for small κ was derived in reference[\[25\]](#page-20-29)(blue dotted-dashed line). (b) The linear stability calculation also yields the wavenumber of the destabilizing fluctuation k<sup>c</sup> (black solid line). This first becomes nonzero at κ<sup>c</sup> ≈ 1.1495 where it empirically behaves like 1.2 √ κ − κ<sup>c</sup> (blue dotted-dashed line). Large-κ estimates for k<sup>c</sup> were also derived in reference [\[14\]](#page-20-12) (red dashed line).

perconductor by a thin insulating layer in reference [\[45\]](#page-20-32). More complicated theories of superconductivity can also be solved using our methods by replacing the Ginzburg-Landau free energy with the appropriate analog, such as the Eilenberger formalism described in more detail in Section [\(IV A\)](#page-7-0).

#### III. EXPERIMENTS

#### A. High Power pulsed RF experiments

Some of the earliest measurements showing Hsh > H<sup>c</sup> for niobium were reported by Renard and Rocher based on DC magnetization measurements. Yogi et al. performed a more systematic study at RF frequencies on samples of Sn, In, Pb, and alloys, in order to cover a range of κ values [\[37\]](#page-20-24). Analysis of their data resulted in the vortex line nucleation model discussed in footnote [4.](#page-3-4) Noting that measurements of the RF critical field have shown inconsistency, Campisi used a very high power RF source at SLAC to very quickly ramp up the fields in cavities [\[46\]](#page-20-33). The goal of these high power RF measurements is to reduce the influence of defects by outpacing the thermal effects they cause. Campisi performed high power RF measurements on Nb, Nb3Sn, and Pb cavities. Hays and Padamsee performed similar measurements on these materials at Cornell [\[47\]](#page-20-34). The niobium results are reproduced in Fig.[\(4\)](#page-6-1), showing fairly reasonable agreement with the expected superheating field close to T<sup>c</sup> [5](#page-6-2) , but then diverging at lower temperatures.

![](_page_6_Figure_7.jpeg)

**Caption:** Exhibits the alterations in maximum quenching and free energy of RF fields in niobium superconductors with baking treatments, revealing notable reductions in field strengths with baking. The variances between baked and unbaked samples highlight potential enhancements in Nb surface treatments for optimizing superconducting RF cavity performance and their quench limits.


<span id="page-6-1"></span>FIG. 4. Pulsed measurement of the maximum field of superconducting niobium cavities from Valles (symbols), compared with estimates of the theoretical maximum possible superheating field (colored ranges). All measurements show good agreement with Hsh at high temperatures. The cavity baked to removed HFQS degradation (red squares) also shows good agreement at low temperatures. DC flux penetration measurements (green triangles) show good agreement with Hsh as well.

After these experiments were performed, new preparation techniques were developed for niobium cavities, including a recipe involving electropolishing and a bake at 120 C. This recipe was found to avoid the "high field Q-slope" (HFQS) degradation mechanism that occurs in niobium cavities at peak fields of approximately 100 mT [\[48,](#page-20-35) [49\]](#page-20-36). Experiments by Valles show that pulsed measurements of unbaked niobium produced curves that diverged from the expected Hsh near the expected onset field of HFQS. However, after the bake was performed, the data agreed very well [\[50\]](#page-20-37). The Hc<sup>1</sup> and Hsh curves plotted in the figure were calculated from niobium material parameters that were extracted from measurements of R<sup>s</sup> vs T and f vs T via the SRIMP Matthis-Bardeen

<span id="page-6-2"></span><sup>5</sup> T<sup>c</sup> assumed to be 9.2 K for Valles' data.

code [\[51,](#page-20-38) [52\]](#page-20-39). The baked curve has a lower Hsh due to the change in the mean free path after the bake, which in turn affects κ.

# B. DC flux penetration measurements by N. Valles.

Valles also performed measurements of the superheating field of unbaked niobium using a DC probe to avoid the effects of HFQS. Using a superconducting solenoid, he applied a DC field to the exterior of a niobium cavity operating at low fields. A sudden decrease in the quality factor of the cavity indicated that flux from the magnet had penetrated to the interior cavity surface. The penetration field extracted from measurements of the applied field agreed well with the expected superheating field for unbaked niobium, as shown in Fig. [\(4\)](#page-6-1) [\[50\]](#page-20-37).

# IV. BEYOND GINZBURG-LANDAU: EILENBERGER, ANISOTROPY, AND DISORDER

The isotropic Ginzburg-Landau analysis of Section [\(II\)](#page-2-0) is a trustworthy estimate for the superheating field only for ideal surfaces of single-band superconductors with cubic symmetry near the superconducting transition temperature Tc. In this section we pursue three topics that introduce new physics to this calculation. First, superconducting RF cavities are usually run at temperatures significantly lower than Tc; niobium cavities, with T<sup>c</sup> ∼ 9K are usually run at T = 2–4K in working accelerators. In Section [\(IV A\)](#page-7-0) we review calculations of the superheating fields that use Eilenberger theory, which is valid at lower temperatures, presenting the analytic results [\[15\]](#page-20-40) at large κ. Our estimates suggest that these Eilenberger corrections to GL are quantitatively important at operating temperatures, but not large. Second, many of the potential new superconductors have rather anisotropic crystal structures and electronic properties; if the superheating field has significant anisotropy, this could motivate single-crystal or controlled growth conditions to control surface orientations in cavities. In Section [\(IV B\)](#page-8-0) we review calculations [\[36\]](#page-20-17) which show that this anisotropy will be small near Tc; we also discuss conflicting results for the anisotropy of multi-band superconductors (like MgB2) at low temperatures. Third, in Section [\(IV C\)](#page-10-0) we estimate the effects of disorder and flaws in these materials, presenting both qualitative and simple quantitative estimates of the effects of defects and dirt in locally lowering the barrier to magnetic flux penetration and thus lowering the effective superheating field.

# <span id="page-7-0"></span>A. Eilenberger theory for lower temperatures

The Ginzburg-Landau approach to superconductivity is generally accurate near the critical temperature Tc, but usually the accuracy of its prediction worsen as temperature is lowered below Tc. A basic example of its failure is given by the temperature dependence of the order parameter ∆: according to GL theory, ∆(T) behaves as p 1 − T /Tc, in agreement near T<sup>c</sup> with the microscopic BCS theory. The latter, however, predicts that at low temperatures the order parameter is temperatureindependent up to exponentially small corrections. For our purposes, the limited validity of GL theory implies that the dependence of the superheating field on κ discussed in Sec. [\(II\)](#page-2-0) cannot be assumed to be quantitatively accurate at the low temperatures at which RF cavities are usually operated. This motivates us to consider a more general approach, valid at arbitrary temperature.

For low-T<sup>c</sup> superconductors, the coherence length ξ<sup>0</sup> = ~v<sup>F</sup> /2∆<sup>0</sup> is much longer than the Fermi wavelength; here ξ<sup>0</sup> is the zero-temperature coherence length for a clean superconductor with zero-temperature order parameter ∆<sup>0</sup> and Fermi velocity v<sup>F</sup> . Thanks to the separation in length scales (or equivalently, the separation in energy scales between ∆<sup>0</sup> and the much larger Fermi energy), these superconductors can be modeled using the so-called quasiclassical approach, reviewed for example in Refs. [\[53,](#page-20-41) [54\]](#page-20-42). This powerful approach is quite flexible, permitting in principle to include effects such as Fermi surface anisotropy and impurity scattering (we will comment on the latter at the end of this section). This come at the price of having to calculate various Green's functions from which physical quantities such as the order parameter and the current can be obtained. Such calculations are usually much more involved that those of the GL approach.

It was shown by Eilenberger [\[6\]](#page-20-4) that one can arrive at an expression for the thermodynamic potential as functional of order parameter ∆(r) and vector potential A(r), similar to the GL functional, once the quasiclassical equations for the Green's function have been solved. While a general solution is not possible, for the case of a clean superconductor with spherical Fermi surface we developed in Ref. [\[15\]](#page-20-40) a perturbative approach valid for large κ. Then the thermodynamic potential Ω is

$$\begin{split} \Omega &= \nu \int d^3r \left\{ \frac{1}{3} \left( \nabla \times \mathbf{A} - \mathbf{H}\_a \right)^2 + \Delta^2 \log \left( \frac{T}{T\_c} \right) \\ &+ \int (dn) \left[ \frac{\Delta^2}{\omega\_n} - 2 \left( \sqrt{\Omega\_n^2 + \Delta^2} - \omega\_n \right) \right. \\ &+ \frac{1}{\kappa\_0^2} \frac{\sqrt{\Omega\_n^2 + \Delta^2}}{4\Omega\_n^2} \left( n \cdot \nabla s^{(0)} \right)^2 \right] \right\}. \tag{12} \end{split}$$

In this expression ν is the density of states at the Fermi energy, lengths are in units of the zero-temperature pen-
etration depth λ0,

$$\frac{1}{\lambda\_0} = \frac{8\pi}{3} \left(\frac{2\pi\xi\_0}{\Phi\_0}\right)^2 \nu \Delta\_0^2,\tag{13}$$

the vector potential is in units of Φ0/2πξ<sup>0</sup> with Φ<sup>0</sup> the magnetic flux quantum, κ<sup>0</sup> = λ0/ξ0, and n is the unit vector on the Fermi surface. We also use the short-hand notations

$$\int (dn) = 2\pi T \sum\_{n} \int \frac{dn}{4\pi}, \qquad \Omega\_n = \omega\_n - in \cdot \mathcal{A} \,, \tag{14}$$

with ω<sup>n</sup> = 2πT(n + 1/2), n = 0, 1, 2, . . ., the fermionic Matsubara frequencies, and

$$s^{(0)} = \frac{2\Delta}{\sqrt{\Omega\_n^2 + \Delta^2}} \,. \tag{15}$$

The thermodynamic potential in Eq. [\(12\)](#page-7-0) reduces to the GL one near T<sup>c</sup> [6](#page-8-0) and it can be used to find the superheating field at arbitrary temperature in the regime κ<sup>0</sup> 1. The calculation of Hsh proceeds in the same manner as in the GL approach, by studying the stability against small perturbation of the local minima of Ω. This study was performed in Ref. [\[15\]](#page-20-0) at leading order in κ<sup>0</sup> → +∞. The ratio Hsh/H<sup>c</sup> between superheating and critical field can be calculated analytically at T = T<sup>c</sup> and T = 0:

$$\frac{H\_{\rm sh}^{\infty}}{H\_c}(T\_c) \simeq 0.745\,, \qquad \frac{H\_{\rm sh}^{\infty}}{H\_c}(0) \simeq 0.840\,, \qquad (16)$$

where we use the ∞ symbol in the superscript to indicate that these are leading-order results. Interestingly, the zero-temperature ratio is almost 13 % larger than the near-T<sup>c</sup> one, indicating that naive extrapolation to low-temperatures of the GL result underestimates the superheating field. At arbitrary temperature, the H<sup>∞</sup> sh /H<sup>c</sup> ratio can be found numerically and is shown in Fig. [\(5\)](#page-8-1). Note the non-monotonic dependence of H<sup>∞</sup> sh /H<sup>c</sup> on temperature, which leads the superheating field to acquire its largest value H<sup>∞</sup> sh ' 0.843Hc(0) at T ' 0.04Tc.

It should be noted that while the Meissner state remains metastable up to Hsh, a clean superconductor can become gapless at a lower field H<sup>g</sup> [\[55\]](#page-20-1); for example at T = 0 we have H<sup>g</sup> ' 0.816H<sup>c</sup> < Hsh. The field H<sup>g</sup> is relevant to applications such as superconducting cavities because as the applied field approaches Hg, AC losses rapidly increase. Indeed, in the presence of a gap the AC losses are in general exponentially suppressed, but this "protection" from losses is absent in the gapless state.

![](0__page_8_Figure_11.jpeg)

**Caption:** Depicts the temperature dependence of the ratio between superheating (Hsh) and critical (Hc) fields, showing a decreasing trend as temperature approaches Tc. The plot captures significant insights into temperature effects on superheating capacity within superconductors, informing on limitations faced near operational thresholds, particularly for high-field applications in particle accelerators.


<span id="page-8-1"></span>FIG. 5. Temperature dependence of the ratio H<sup>∞</sup> sh /H<sup>c</sup> (from [\[15\]](#page-20-0)). Note the non-monotonic behavior at low temperatures.

The above results are restricted to the leading order in 1/κ0, which makes it possible to neglect the contributions from the last term in square brackets in Eq. [\(12\)](#page-7-0). At next to leading order, that term must be taken into account and leads to an expression for the superheating field of the form [7](#page-8-2)

$$\frac{H\_{\rm sh}}{H\_c}(T) \simeq \frac{H\_{\rm sh}^{\infty}}{H\_c}(T) + \frac{h(T)}{\sqrt{\kappa\_0}}.\tag{17}$$

This formula, with a weakly temperature-dependent dimensionless coefficient h(T), has the same inverse square root dependence on κ<sup>0</sup> as the GL expression [\[14\]](#page-20-2).

In closing this section, let us comment briefly on the effect of impurity scattering. Both non-magnetic and magnetic impurities were considered in Ref. [\[55\]](#page-20-1) in the limit κ → ∞. At sufficient strength of the non-magnetic impurities scattering rate, there are some qualitative changes: the non-monotonicity of Hsh(T) is suppressed, and more importantly the gap remains open up to Hsh. However, quantitatively the value of Hsh/H<sup>c</sup> is changed by at most a few percent. In contrast, adding magnetic impurities strongly decreases Hsh, similar to the well-known suppression of T<sup>c</sup> due to the pair-breaking effect of such impurities.

### B. Anisotropic superconductors

Layered superconductors can display highly anisotropic critical fields. The upper-critical field of BSCCO,[8](#page-8-3) for instance, can vary by two orders of

<span id="page-8-0"></span><sup>6</sup> In considering the limit T → T<sup>c</sup> in Ref. [\[15\]](#page-20-0), a prefactor was missed in Eq. (29) and consequently Eq. (31), which should read respectively: κGL = 2πTp p 2/3ζ κ<sup>0</sup> ≈ 1.50κ<sup>0</sup> and ξ(T) = 2/3[∆0/∆(T)]ξ<sup>0</sup> in the notation of that work.

<span id="page-8-2"></span><sup>7</sup> This formula can be obtained by extending to next-to-leading order the calculations of Ref. [\[15\]](#page-20-0) (G. Catelani, unpublished).

<span id="page-8-3"></span><sup>8</sup> The cuprate superconductors have d-wave order parameters, and hence have an anisotropic gap that vanishes along certain directions. Thus, as discussed for gapless superconductivity in Section [\(IV A\)](#page-7-1), these likely will not be useful for sustained operations at GHz frequencies.

magnitude depending on the angle between the crystal anisotropy axis c and the applied magnetic field [\[5\]](#page-20-3). Near zero temperature, the upper critical field of magnesium diboride is about six times larger for c ⊥ B than for c k B (see e.g. [\[56,](#page-20-4) [57\]](#page-20-5)). Here we review Ref. [\[36\]](#page-20-6), which investigates the effects of crystal anisotropy on the superheating field of superconductors, motivated partly by the opportunity of controlling surface orientation in order to achieve higher accelerating fields inside the cavity.

Near the critical temperature, for the anisotropy axis c aligned with one of the Cartesian directions, the anisotropic formulation of Ginzburg-Landau theory [\[58–](#page-20-7) [61\]](#page-20-8) provides a clean approach to study the anisotropy of the superheating field. We can use a change of coordinates and rescaling of the vector potential to turn the anisotropic GL free energy onto isotropic form, and then use previous results from Ref. [\[14\]](#page-20-2) to calculate the superheating field anisotropy of several materials. We find that:

$$H\_{\rm sh}^{\rm ani} = \begin{cases} H\_{\rm sh}(\kappa\_{\parallel}), & \text{for } \mathbf{c} \parallel z, \\\ H\_{\rm sh}(\gamma \kappa\_{\parallel}), & \text{for } \mathbf{c} \parallel x \text{ or } y, \end{cases} \tag{18}$$

where the superheating field on the right hand side is the solution of the linear stability analysis for isotropic Fermi surfaces, which we discussed in Section [\(II B\)](#page-4-0), using κ = κ<sup>k</sup> and κ = κ<sup>⊥</sup> = γκ<sup>k</sup> for c parallel and perpendicular to z, respectively. Within GL theory, γ = p mc/m<sup>a</sup> = λc/λ<sup>a</sup> = ξa/ξc, with m<sup>i</sup> , λ<sup>i</sup> and ξ<sup>i</sup> representing the effective mass, penetration depth, and coherence length along the i-th direction, respectively. Since Hsh ≈ 0.75 H<sup>c</sup> goes to a constant for large κ, we find that the superheating field is nearly isotropic for most high-κ unconventional superconductors. On the other hand, Hsh ≈ 0.84 H<sup>c</sup> κ −1/2 for small-κ type-I superconductors, resulting in an anisotropy of about γ 1/2 when κkγ is small. Figure [\(6\)](#page-9-0) displays a phase diagram in terms of κ<sup>k</sup> and γ, showing the region where GL theory predicts type-I (left of the blue line), type-II (right of dark red line) and mixed (in between dark red and blue lines) superconductivity, and the regions where each asymptotic solution is expected. Note, in particular, that Hsh k /Hsh <sup>⊥</sup> ≈ 1 for MgB2. This result is valid only very near Tc, where the anisotropies in λ and ξ are equivalent. In the next paragraph we will use results from a two-gap BCS theory to estimate the superheating field anisotropy of MgB<sup>2</sup> at lower temperatures.

Theoretical and experimental studies indicate that the assumption λc/λ<sup>a</sup> = ξa/ξ<sup>c</sup> (vortex and vortex core have identical shapes within GL theory) is strongly violated for low-temperature MgB2, thus suggesting the use of two parameters to describe crystal anisotropy, namely γ<sup>λ</sup> = λc/λ<sup>a</sup> and γ<sup>ξ</sup> = ξa/ξc. Also, γ<sup>λ</sup> and γ<sup>ξ</sup> exhibit different temperature dependences, with γ<sup>λ</sup> decreasing and γ<sup>ξ</sup> increasing for decreasing temperature, respectively. Calculations from Ref. [\[56\]](#page-20-4) using a two-gap BCS model suggest that γ<sup>λ</sup> and γ<sup>ξ</sup> become equal only at Tc; near

![](0__page_9_Figure_6.jpeg)

**Caption:** This phase diagram maps the regions of Type I, Type II, and Mixed superconducting behavior in terms of anisotropic parameters based on mass and Ginzburg-Landau equations. The boundaries delineating Type I, Mixed, and Type II regions reveal how superconducting properties transition with material anisotropy, providing insight into optimizing alloy compositions like MgB2 and BSCCO for specific applications involving superconducting states.


<span id="page-9-0"></span>FIG. 6. Phase diagram of anisotropic superconductors in terms of mass anisotropy (γ = p mc/ma) and GL (λa/ξa) parameters (from [\[36\]](#page-20-6)). The superconductor is of type-I to the left of the blue line, of type-II to the right of the dark red line, and mixed in between (in the mixed phase, the SC is of type-I for c k z and of type-II for c ⊥ z). The blue and yellow regions correspond to the asymptotic solutions Hsh k /Hsh <sup>⊥</sup> ≈ γ 1/2 and Hsh k /Hsh <sup>⊥</sup> ≈ 1, respectively (within 10% accuracy). Note that the superheating field of MgB<sup>2</sup> is nearly isotropic near T = Tc.

zero temperature, γ<sup>ξ</sup> ≈ 6 whereas γ<sup>λ</sup> ≈ 1, agreeing with some [\[57,](#page-20-5) [62,](#page-20-9) [63\]](#page-20-10), but not all (See Ref. [\[56\]](#page-20-4) and references therein) experimental estimates.

We can use our simple estimates of Section [\(II A\)](#page-2-0) to make a qualitative prediction for the resulting anisotropy H c⊥y sh /H<sup>c</sup>k<sup>y</sup> sh in the superheating field, when γ<sup>ξ</sup> 6= γ<sup>λ</sup> deviates from the single-band GL prediction. Now the anisotropic shape of the vortex and vortex core plays an important role (see Fig. [\(7\)](#page-10-0)a). When c is in the xy plane, as in Fig. [\(7\)](#page-10-0)b, for instance, the superheating field is estimated from the work performed to push the black-dashed "box" into the superconductor, which can considerably vary from c k y (left) to c k x (right). This leads to an estimate H c⊥y sh /H<sup>c</sup>k<sup>y</sup> sh ≈ γξ/γλ. A second estimate generalizes the Bean and Livingston argument of the longerrange vortex attraction to incorporate anisotropy, and leads to a slightly different result: H ckx sh /H<sup>c</sup>⊥<sup>x</sup> sh ≈ γξ/γλ. Yet a third calculation, which we term "Extended GL", yields an almost isotropic result, and is based on a direct linear stability analysis of the anisotropic GL free energy (see Eq. (7) of Ref. [\[36\]](#page-20-6)) assuming unconstrained λ's and ξ's. Table [II](#page-10-1) summarizes our estimates of Hsh for the three geometries, using experimental values for H<sup>c</sup> and κ for MgB2. Note that we correct numerical discrepancies of our first estimates in the second row of the table: "1st (corrected)". The last column shows the maximum superheating field anisotropy according to each method. Most of the values of Hsh are as low as Hsh ≈ 0.24T for Nb [\[13\]](#page-20-11). We discuss the origin of these disparate predictions further in Ref. [\[36\]](#page-20-6).

Our GL arguments for the superheating field anisotropy can be trusted near Tc: at large κ the super-

![](0__page_10_Figure_1.jpeg)

**Caption:** Depicts a gravitating vortex (blue) with core (red) contrasted against superconducting surface configurations, demonstrating the anisotropies within low-temperature MgB2 under external fields. The scale is exaggerated for clarity, showing distinctions vital for interpreting superheating field anisotropies in multi-band superconductors like MgB2, impacting their practical synthesis and functionality.


<span id="page-10-0"></span>FIG. 7. From Ref. [\[36\]](#page-20-6). (a) Illustrating vortex (blue disk) and vortex core (red disk) of zero-temperature MgB<sup>2</sup> in the ac plane, with the external magnetic field parallel to the normal of the plane of the figure. We have drawn ξ<sup>a</sup> about 30 times larger with respect to λa, so that the core becomes discernible; in the correct scale, the vortex core occupies the tiny black region in the middle of the figure. Notice that vortex and vortex core have identical shapes within GL theory. (b) To estimate the superheating field, we calculate the work to push a vortex core into the superconductor, thus destroying the Meissner state. The very different area of the black dashed boxes for c k y (left) and c k x lead to substantial anisotropy of the superheating field for low-temperature MgB2.

| Approach                  | Hsh<br>( Tesla ) |                   |      | Max. Anis. |
|---------------------------|------------------|-------------------|------|------------|
|                           |                  | c k x c k y c k z |      |            |
| 1st estimate              |                  | 0.04 0.006 0.04   |      | ∼ 6        |
| 1st (corrected)           | 0.2              | 0.03              | 0.2  | ∼ 6        |
| 2nd estimate (B & L) 1.13 |                  | 0.18              | 0.18 | ∼ 6        |
| "Extended GL"             | 0.21             | 0.22              | 0.22 | ∼ 1        |

<span id="page-10-1"></span>TABLE II. Estimates of the superheating field and maximum anisotropy of low-temperature MgB<sup>2</sup> for three geometries.

heating field anisotropy is not a reason to control surface orientation. Our arguments at lower temperature and for multi-band superconductors are more speculative. The vortex core shape will surely change for x ∼ ξ due to the boundary conditions at the surface; the anisotropy in the long-range attraction in multi-band materials may be different from that of a simple anisotropic GL approach. It will be important to apply linear stability analysis to more sophisticated theories, such as multi-gap BCS or strong-coupling Eliashberg theory, especially in the face of the conflicting results shown in Table [II.](#page-10-1)

## C. Disorder and vortex nucleation.

Niobium RF cavities are routinely operated in the metastable regime, at fields Hc<sup>1</sup> < H < Hsh above the field Hc<sup>1</sup> where vortices in equilibrium would penetrate into the superconductor (and dissipate roughly the same energy as in a normal metal). Table [\(I\)](#page-3-0) in Section [\(I D\)](#page-2-1) gives Hc<sup>1</sup> and Hsh for other candidate materials. For niobium this metastable regime gives us an important factor of ∼ 1.6 in field. Running in the metastable regime is crucial for utility with the higher temperature superconductors, whose Hc<sup>1</sup> equilibrium fields are much lower than the operating fields for current Nb cavities (Table [\(I\)](#page-3-0)).

It took many years of experimentation to raise operating fields of the niobium cavities to approach near to their fundamental limits. Will the new, more complex materials be fundamentally more challenging to optimize? Our preliminary experimental cavities using Nb3Sn appear already to be operating above Hc<sup>1</sup> [\[64\]](#page-20-12), but are not yet delivering anywhere near to the theoretically predicted superheating field. Just as we have been exploring the fundamental theoretical limits to the fields for ideal surfaces, in this section we explore the fundamental theoretical challenges in minimizing the effects of dirt, flaws, and defects in lowering the barriers to vortex entry.

What kind of flaw or disorder fluctuation would be needed to allow vortices to enter at fields substantially lower than the superheating field? How big a damage region is needed to bypass the surface barrier to vortex entry? Damage will significantly affect the superconducting properties if the flaw or fluctuation has a characteristic length of order the coherence length ξ. Since the proposed candidate materials for next generation SRF cavities have shorter coherence lengths than niobium (Table [\(I\)](#page-3-0)), this potentially could imply that these new materials are more susceptible to defects and dirt.

Figure [\(8\)](#page-11-0) shows a cartoon of a vortex loop entering a superconductor. Based on the discussion at the end of Section [\(II A\)](#page-2-0) and the caption of Fig. [\(8\)](#page-11-0), at external fields far from Hc1 and Hsh, the energy of the vortex loop will grow in the absence of disorder until it reaches a critical radius Rc, at which point the energy will again decrease. This critical radius and the needed damage zone will get smaller as the field H grows, vanishing at H = Hsh.

The energy per unit length of the vortex loop will have two contributions – a curvature energy and an attractive

![](0__page_11_Figure_0.jpeg)

**Caption:** Illustrates a schematic depicting a vortex flux tube nucleating into a superconductor from a semi-cylindrical radius R. The nucleation barrier for vortex penetration describes the energetics within the superconducting cavity context, offering insights into defect-driven vortex dynamics and potential enhancements in applied magnetic fields.


<span id="page-11-0"></span>FIG. 8. Flux tube nucleation allowing the penetration of a single vortex core into the superconductor occupying the half space x > 0. The nucleation barrier at zero disorder can be estimated by computing the energy of this loop, plausibly a semicircular loop of radius R, and subtracting the magnetic work done by the pressure due to the external field H, where Hc1 < H < Hsh. This figure illustrates the nucleation barrier for low fields near Hc1; at higher fields the radius becomes comparable to ξ. The boundary conditions at the surface of the superconductor lead to an attractive force on the vortex as discussed in Section [\(II A\)](#page-2-0), in addition to the curvature energy of the vortex loop (ignored here). We can estimate the disorder needed to nucleate at a field H by calculating the damage needed to lower this barrier to zero as the radius R grows.

energy between the vortex and the surface. The latter can be estimated from the attraction of a straight vortex to the 'image vortex' needed to set the correct boundary conditions at the surface. This potential barrier (the major component of the superheating field) was estimated by Bean and Livingston for high κ type-II superconductors [\[26\]](#page-20-13). The unitless Gibbs free energy per unit length 4 π G/( √ 2 H<sup>c</sup> Φ0) of a straight vortex flux line a depth x inside a superconductor with external field H can be written in the (London) large-κ limit as [\[12\]](#page-20-14):

$$G = \frac{\Phi\_0}{4\pi} \left( H(e^{-x/\lambda} - 1) - \frac{1}{2} \frac{\Phi\_0}{2\pi\lambda^2} K\_0(2x/\lambda) + H\_{c1} \right) \tag{19}$$

$$\frac{4\,\pi\,G}{\sqrt{2}\,H\_c\,\Phi\_0} = g(x) = h\left(e^{-x/\lambda} - 1\right) - \frac{K\_0(2\,x/\lambda)}{2\,\kappa} + \frac{\ln\kappa}{2\,\kappa},\tag{20}$$

where h = H/( √ 2Hc), and K<sup>ν</sup> denotes the modified Bessel function of imaginary argument [\[65\]](#page-20-15), and for now κ and λ are the Ginzburg-Landau parameter and penetration length of the pure material, respectively. The first term is a magnetic pressure, the second term is the interaction with the 'image vortex' that imposes the correct boundary conditions, and the third term is the energy per unit length of a vortex deep in the superconductor. We can estimate Rc(H) by setting the derivative dG/dx = 0 in Eq. [\(19\)](#page-11-1) and expanding the Bessel function for small arguments, leading to Rc(H) ∼ ξHsh/H. At the lowest field for vortex penetration Hc<sup>1</sup> this expansion is unreliable; however, since Hc<sup>1</sup> = Hsh log(κ)/κ [\[12\]](#page-20-14), the resulting estimate Rc(Hc1) ≈ ξκ/ log(κ) = λ/ log(κ) is still quite good, as shown in Figure [\(9a](#page-12-0)). But the new materials of interest have lower critical fields Hc<sup>1</sup> too small to be useful; we must run at fields H comparable to Hsh. Near Hsh, R<sup>c</sup> ≈ ξ, again as shown in Fig. [\(9\)](#page-12-0). For disorder or defects to remove this energy barrier, they will thus necessarily have to strongly affect a region of volume ∼ ξR<sup>2</sup> <sup>c</sup> ∼ ξ 3 .

To make this more quantitative, one needs to identify and model the dominant mechanism for vortex nucleation. If the characteristic defect size is large compared to ξ (e.g., nucleation on grain boundaries or inclusions of competing phases), one must model and control these individual defects. Clean grain boundaries are usually atomistically sharp (much thinner than ξ) and hence do not significantly decrease the local superconducting properties; indeed, studies of hot spots in large grain niobium cavities show no correlation with grain boundaries [\[66\]](#page-21-0), and using single crystals to avoid grain boundaries has not improved performance [\[67,](#page-21-1) [68\]](#page-21-2). But in more complex materials, grain boundaries could be more disordered, thicker, or contaminated by impurities, and a grain boundary or grain boundary intersection with the correct orientation with respect to the surface could provide a route to entry. The effect of surface roughness on Bean and Livingston's surface barrier has been studied in Ref. [\[69\]](#page-21-3). Kubo has used the London model to investigate the effects of nano-scale surface topography on the superheating field [\[70\]](#page-21-4). Perhaps most dangerous could be inclusions of metallic or poorly superconducting second phases, or irregularities in the surface morphology.

If the characteristic defect size is small compared to ξ, and if the defects are uncorrelated in position, then the fluctuations in regions of order ξ 3 can be quantitatively estimated to linear order using the central limit theorem. This leads to Gaussian random fluctuations in the superconducting properties. For example, for alloys and doped crystals there are natural concentration fluctuations that will locally change the superconducting transition temperature, coherence length, condensation energy, and other properties. This is the traditional theoretical framework for field-theoretic calculations of the effects of disorder.

<span id="page-11-2"></span><span id="page-11-1"></span>Let us hypothesize a system where the critical temperature is decreased due to disorder. In the context of Ginzburg-Landau theory for a homogeneous system, a change in the critical temperature yields a change in the coefficient α = α(x) of ψ 2 , where ψ is the superconductor order parameter [\[5\]](#page-20-3). The probability of a fluctuation in α(x) away from its pure value α<sup>0</sup> would be proportional

![](0__page_12_Figure_0.jpeg)

**Caption:** This figure consists of two panels depicting the superheating field adjusted for Gaussian disorder across varying magnetic fields. Panel (a) examines the difference between g(x) and the normalized coherence length, ξ0, across a range of fields, showing critical points at superheating (Hsh) and lower critical fields (Hcl). Panel (b) presents the reduced coherence (α/a0) against x/ξ0, indicating fluctuations across different magnetic intensities. Such graphs elucidate the microscopic structural responses of superconductors like Nb3Sn to magnetically induced stress, potentially aiding in predicting performance under practical operational limits.


<span id="page-12-0"></span>FIG. 9. (a) Unitless Gibbs free energy (Eq. [\(20\)](#page-11-2)) to push a straight vortex line from a depth ξ to depth x into a superconductor like Nb3Sn with κ = 26.4, for several values of the magnetic field. The superheating field can be estimated in the large κ limit from the condition G 0 (ξ) = 0, characterizing the vanishing of the surface energy barrier at x = ξ; Bean and Livingston's estimate gives hsh = 1/2 so Hsh = Hc/ √ 2 ≈ 0.71Hc, comparable to the correct large-κ limit. Note that the peak in the barrier is at x<sup>c</sup> ∼ ξHsh/H; near Hc<sup>1</sup> it is roughly λ ≈ κξ/ log κ = λ/ log κ ≈ λ, but in the interesting region near Hsh it is near the coherence length ξ. (b) The spatially-dependent critical temperature shift α = α(x), needed to flatten the energy barrier and allow for the penetration of vortices, in our particular model with κ for Nb3Sn. This is shown for several values of H in the interval [Hc1, Hsh). Here H = 0.6H<sup>c</sup> would duplicate the maximum possible superheating field for niobium.

to

<span id="page-12-1"></span>
$$\, \Pi \{ \alpha(\mathbf{x}) \} \propto \exp \left( - \int (\alpha(\mathbf{x})/\alpha\_0 - 1)^2 / (2\sigma^2) d^3 \mathbf{x} \right), \tag{21}$$

where σ is a material-dependent constant that encapsulates the likelihood that the dirt in the material will cause a given fractional change α/α<sup>0</sup> in the critical temperature. The constant σ will become larger either if there are bigger concentration fluctuations or if the material is particularly sensitive to dirt. In principle, we should now calculate the most probable three-dimensional profile α(x) needed to flatten the energy barrier and allow vortices in at a lowered field H < Hsh, and then use Π{α(x)} in Eq. [\(21\)](#page-12-1) to estimate the probability per unit surface area P(H/Hsh) = Π{α(x)} of vortex penetration.

Rather than doing this full variational calculation, we build on the Bean-Livingston model of Equation [\(19\)](#page-11-1). In GL theory, the characteristic lengths scale as λ ∼ α −1 and ξ ∼ α −1/2 . Hence we distinguish λ0, ξ<sup>0</sup> and κ<sup>0</sup> for the pure material from λ(x) = λ0/a, ξ(x) = ξ0/ √ a and κ(x) = λ(x)/ξ(x) = κ0/ √ a for the damaged region, where a = α/α0.

![](0__page_12_Figure_8.jpeg)

**Caption:** This figure presents a log-log plot showing the probability of vortex nucleation, P(H/Hsh), as a function of reduced magnetic field, H/Hsh, for three superconductors: Nb3Sn, NbN, and MgB2. The probability decreases with increasing field strength, with the curves representing the different superconductors illustrating their respective susceptibility to vortex penetration under varied conditions of Gaussian disorder. The significance lies in understanding the reliability and capability of these materials to resist vortex nucleation under applied magnetic fields, critical for optimizing superconducting materials for SRF cavities.


<span id="page-12-3"></span>FIG. 10. Relative logarithmic reliability P = −(2σ 2 /ξ<sup>3</sup> <sup>0</sup>) log(P(H/Hsh)) of vortex nucleation, in a simple model of Gaussian random disorder, for the κ values of the three candidate superconductors. Solid curves are P3<sup>D</sup> for a semicircular vortex barrier model (Fig. [\(8\)](#page-11-0), Eq. [\(26\)](#page-13-0)); dashed curves are (d/ξ0)P2<sup>D</sup> for pancake vortex nucleation in a 2D superconducting layer of thickness d (Section [\(V\)](#page-13-1)).

What is the minimum amount of dirt that is necessary to reduce the superheating field to a certain value? For instance, how much dirt would it take to reduce Hsh for Nb3Sn (estimated at 0.42 T in Table [\(I\)](#page-3-0)) to H = 0.25 T (Hsh for niobium), a factor H/Hsh ∼ 0.6? One would need enough dirt 'flatten' the surface barrier between[9](#page-12-2) ξ<sup>0</sup> and Rc(H) ≈ ξ0Hsh/H = 5ξ0/3 along the x direction (thus allowing for vortex penetration), as shown in the dashed line of Fig. [\(9a](#page-12-0)). In general, we are interested in finding an x-dependent parameter α = α(x) that flattens the energy barrier from x = ξ<sup>0</sup> to x = x<sup>f</sup> , where x<sup>f</sup> > ξ0, and is defined by G(x<sup>f</sup> ) = G(ξ0). The solution for α(x) is then found from the equation G(x) = G(ξ0) for ξ<sup>0</sup> < x < x<sup>f</sup> , and α(x) = α<sup>0</sup> for x > x<sup>f</sup> , where in the left and right hand sides we use {λ(x), ξ(x)} and {λ0, ξ0} in Eq. [\(19\)](#page-11-1), respectively.

Note that we are making a rough approximation here. The magnetic fields and supercurrents surrounding the vortex line will see a spatially varying critical temperature α(x) whenever it is far from the surface, and properly measuring its energy and thus the surface attraction

<span id="page-12-2"></span><sup>9</sup> Bean and Livingston measure the barrier starting at x = ξ, below which London theory is unreliable.

should include the resulting shift in energy. The depths x of importance to us are of order the coherence length ξ, and thus these long distance fields and currents are largely cancelled by the image vortex a distance 2x away. The vortex will see a depth-dependent disorder, but its energy will be qualitatively well described by our model in the region H ∼ Hsh.

Figure [\(9\)](#page-12-0) shows α/α<sup>0</sup> as a function of x/ξ for several values of H in the interval [Hc1, Hsh). Apart from an overall constant given by the normalization of the Gaussian, the negative logarithm of the probability of this fluctuation as a function of the lowered entry field H is

$$-\log(P(H/H\_{\rm sh})) = \Pi\{\alpha(\mathbf{x})\}\tag{22}$$

$$=\int (\alpha(\mathbf{x})/\alpha\_0 - 1)^2 / (2\sigma^2) d^3 \mathbf{x} \quad (23)$$

$$=\frac{\xi\_0^3}{2\sigma^2} \int\_{\epsilon^3} (\alpha(\xi\_0 \mathbf{u})/\alpha\_0 - 1)^2 d^3 \mathbf{u} \quad (24)$$

$$=\frac{\xi\_0^3}{2\sigma^2}\mathcal{P}(H/H\_{\rm sh}),\tag{25}$$

a measure of the relative logarithmic reliability of the superconductor to disorder-induced nucleation. Here we pull out the volume ξ 3 <sup>0</sup> of the damage zone by changing variables to u = x/ξ0. In a three-dimensional system with a semicircular vortex nucleation approximation, we can use our Bean-Livingston style methods to approximate this as a one-dimensional integral

<span id="page-13-0"></span>
$$\mathcal{P}\_{3D}(H/H\_{\rm sh}) = \int \pi u(\alpha(\xi\_0 u)/\alpha\_0 - 1)^2 du. \tag{26}$$

For a vortex pancake nucleation event for a thin SIS film of thickness d (discussed in Section [\(V\)](#page-13-1)), we find

$$\mathcal{P}\_{2D}(H/H\_{\rm sh}) = (d/\xi\_0) \int (\alpha(\xi\_0 u)/\alpha\_0 - 1)^2 du \qquad (27)$$

(see Fig [\(10\)](#page-12-3)).

Clearly, the relative reliability decreases rapidly as H approaches Hsh, by many orders of magnitude in this model calculation. The high-κ calculation of Bean and Livingston cannot be simply extrapolated to niobium, but there is no reason to doubt that a similar sensitivity of the barrier to H/Hsh is expected. Nonetheless, niobium cavities are used in planned applications at 0.7Hsh [\[1,](#page-19-0) [71\]](#page-21-5), suggesting realistic values of disorder are tolerable in niobium. Indeed, the dependence of the barrier on H/Hsh is much stronger than its dependence on κ or ξ. This suggests, examining Figure [\(10\)](#page-12-3), that the factor of five to ten change in ξ<sup>0</sup> with the new superconductors may not be so dangerous. The resulting two to three orders of magnitude smaller volume for the critical damage zone at fixed field, it would seem, could be remedied by working not at 0.8Hsh but at perhaps 0.6Hsh (Figure [\(10\)](#page-12-3)). Manufacturing high-quality cavities from these new materials may be challenging. What our calculation can provide is reassurance that these materials should not be avoided because of their shorter coherence lengths.

# <span id="page-13-1"></span>V. LAMINATES AND VORTEX PENETRATION

In recent years, much effort in superconducting RF has been devoted to exploring single or multiple thin films – laminated structures hopefully tunable to optimize performance. This section is devoted to exploring possible advantages to such laminates. The work in this section relies heavily on extensive discussions and consultation with Alex Gurevich, whose work prompted most of the calculations presented.

In practical terms, two of the candidate materials (Nb3Sn and NbN) can be grown by deposition on Nb surfaces, so fabricating a surface layer onto a Nb cavity leverages existing expertise. Gurevich points out [\[72\]](#page-21-6) that thermal conductivities of new candidate materials are often small; since the heat generated by the surface residual resistance at the surface must be conducted through the cavity, keeping the thickness of these new materials small can improve performance. (For Nb3Sn, recent surface resistances have been small enough, at least at low fields, that thickness may not be an issue.) Gurevich has also proposed [\[73\]](#page-21-7) separating one or more superconducting layers by insulating layers (a SIS geometry). Calculations show [\[45\]](#page-20-16) that laminates do not substantially improve the theoretical maximum superheating field in AC applications beyond that of pure materials (or thick layers) for the film-insulator-bulk structure,[10](#page-13-2) though adding a thin S 0 layer on the bulk S superconductor may lead to an enhancement of the energy barrier [\[77,](#page-21-8) [78\]](#page-21-9).

Gurevich has suggested that the SIS geometry may have a different advantage – reducing the impact of flux penetration. Our calculations in Section [\(V C\)](#page-15-0) suggest that SIS films with thickness d small compared to the London penetration depth λ will be more susceptible to vortex penetration than bulk films; the damage zone needed for vortex nucleation at fields below pure Hsh can be thinner by the fraction λ/d, presumably making them much more likely. Also, one would naively expect it to be harder to grow low-defect two-layer laminates than depositing a single layer or preparing a pure surface. Layers thick compared to the penetration depth would presum-

<span id="page-13-2"></span><sup>10</sup> A free-standing superconducting layer (or a layer surrounded by insulators) with thickness small compared to the magnetic penetration depth λ can have an enormous superheating field (since it can remain superconducting without paying most of the cost of expelling the flux). In the accelerator community, there is widespread focus on raising this 'Hc1' for the superconducting film [\[74,](#page-21-10) [75\]](#page-21-11) – defined, somewhat unphysically [\[76\]](#page-21-12) as the minimum field needed for a vortex to be stable parallel to and inside the film. But such an in-film stable vortex configuration demands magnetic flux on both sides of the film. In a GHz AC application, pushing the flux through the film twice per cycle generates unacceptable heating [\[76\]](#page-21-12). Besides, any such parallel vortex would be precariously unstable to formation of two vortex pancakes. A thin superconducting layer with a large magnetic penetration depth atop a lower-Hsh layer with a small penetration depth can have modestly higher superheating fields, due to the way the bottom layer modifies the magnetic field penetration.

ably behave similarly to a bulk material; vortices deeper than λ do not 'feel' the surface except insofar as other vortices penetrating the surface push them deeper.

The dynamics after flux penetration will be substantially different for the SIS geometry than for a simple 3D superconducting surface. In either case, a flaw may nucleate one to several vortex entries when the field increases in one direction; some or all may be 'pulled back' as the field shifts to the opposite direction. If the nucleation center flaws are rare and the vortices do not build up over time, they need not cause local heating enough to cause a quench. But since the RF cavities operate at GHz frequencies, and each flaw could (or should) generate multiple vortices per cycle, potentially billions of vortices could be introduced by a single flaw if they can escape re-annihilation.

In three dimensions, a vortex penetrating at a point (y, z) on the surface will grow in the z direction pointing along the field as it penetrates a depth of order x ∼ λ (Fig. [\(11\)](#page-14-0) left). If multiple vortices enter, they may push and entangle one another; as they interact with disorder in the material they may exhibit avalanches [\[79,](#page-21-13) [80\]](#page-21-14). During the field reversal, the points where the vortices exit the material will be forced together along the z direction (shrinking in length), and new vortices with opposite winding number will nucleate (potentially annihilating some or all of the old vortices). Even if this process is incomplete, leaving some tangle of vortex loops, it may enter a kind of limit cycle. Indeed, many periodically stressed disordered dynamical systems can enter into limit cycles at low levels of stress, with a transition to 'turbulent' aperiodic behavior at a critical threshold (colliding colloids in reversing low-Reynolds number flows [\[81\]](#page-21-15), plasticity in vortex structures of superconductors [\[82](#page-21-16)[–84\]](#page-21-17), etc). It is possible that the quench of RF cavities explores precisely this kind of dynamical phase transition, separating a local hot spot from an invading front of vortices. Apart from these brief speculations, we will not discuss three-dimensional dislocation dynamics further in this work; the remainder of this section will focus on the SIS geometry.

In the two-dimensional SIS geometry, a vortex penetration event may end with the vortex trapped in the insulating layer, leaving two 2D vortices penetrating the outer superconducting film (Fig. [\(11\)](#page-14-0) right. See also footnote [10.](#page-13-2)) Such 2D vortices, called pancake vortices, have been studied in great detail [\[12\]](#page-20-14) in the context of high temperature cuprate superconductors, some of which are well described as nearly decoupled 2D superconducting sheets. A vortex pair nucleated by a defect at (y, z) on the surface will separate along the z direction as the field increases, be buffeted by thermal fluctuations, dirt, defects, and other vortices as they separate, and then be pulled back along the z direction as the field reverses. (Some of the other vortices will be emitted by the same defect, once the initial pair departs and the resulting long-range suppression of nucleation drops, see Section [V D.](#page-17-0)) In this part we shall explore Gurevich's

suggestion that, even after billions of cycles, this annihilation should be effective at avoiding vortex escape (presumably preventing a buildup of vortices which otherwise would lead to a quench).

In Section [\(V A\)](#page-14-1), we introduce an "impact parameter," the amount of lateral vortex separation between a vortex-antivortex pair that can be tolerated during a cycle while still expecting them to annihilate, in Section [\(V B\)](#page-14-2), we examine the expected lateral meandering distance expected from pancake vortices in an RF cycle, in Section [\(V C\)](#page-15-0), we examine the expected meandering due to disorder, and in Section [\(V D\)](#page-17-0), we briefly consider the effect of vortex-vortex interaction and the situation of two nearby defects.

![](0__page_14_Figure_7.jpeg)

**Caption:** This schematic differentiates between vortex behavior in bulk superconductors and in superconducting films separated by insulators, forming pancake vortices. It illustrates the concept of vortex semiloops and pancake vortex cores, highlighting how structural modifications in superconductors, like film thickness or insulator layers, impact vortex dynamics and pinning mechanisms, instrumental in designing materials to refine superconductor performance under variable external fields.


<span id="page-14-0"></span>FIG. 11. Vortices in a bulk superconductor for semiloops (left). Vortices in thin superconducting films separated by insulators form pancakes.

### <span id="page-14-1"></span>A. Impact Parameter

How far ∆x perpendicular to the field must a vortex pair migrate before their mutual attraction ceases to be strong enough to annihilate them at the end of a cycle? Figure [\(12\)](#page-15-1) shows the trajectories for a pancake pair as they return at the end of a cycle, separated by different distances x perpendicular to the external magnetic field, using the vortex interaction formulation from Ref. [\[85\]](#page-21-18). There is a separatrix between trajectories which collide and trajectories which miss each other. We will call the value of ∆x at this separatrix the impact parameter, ximp. For perhaps credible parameters d = 30 nm, λ = 100 nm, µ0Hsh = 0.4 T, ximp ∼ 20 nm. Simulations were used to evaluate ximp as a function of field, and the results are plotted in Fig. [\(13\)](#page-15-2).

## <span id="page-14-2"></span>B. Thermal Meandering

The motion due to thermal fluctuations can be estimated using the Einstein equation,

$$
\left = 2Dt\tag{28}
$$

![](0__page_15_Figure_0.jpeg)

**Caption:** Shows trajectories of vortex-antivortex pancaked pairs in RF superconducting films, analyzing annihilation thresholds for varied perpendicular separations. The impact parameter ximp dictates maximal lateral separation allowing annihilation, with implications on decay and lifetime of vortices, essential in the design for SRF applications where vortex mitigation is critical.


<span id="page-15-1"></span>FIG. 12. For initial separations smaller than the impact parameter ximp, the pancakes annihilate (inner solid, dash, dot), but larger than ximp, they can wander away (dash-dot, outer solid). α = 0.2 shown

![](0__page_15_Figure_2.jpeg)

**Caption:** This graph compares the lateral separation limits in vortex-antivortex pairs to maximal meandering induced by thermal motions and disorder. For superconductors specimens like Nb3Sn, it quantitatively outlines factors limiting vortex pair annihilation, emphasizing the improbability of significant vortex escape events. This evaluation is essential for ensuring the integrity and performance reliability of next-generation SRF cavity materials.


<span id="page-15-2"></span>FIG. 13. This figure compares ximp, the maximum lateral separation resulting in impact of a vortex pair, to p hx 2 thermali and p hx 2 disorderi, the expected meandering distances due to thermal and disorder effects, for realistic parameters given in the text. Note that the former remains a factor of at least ten larger than the latter, suggesting that vortex escape by these mechanisms is a 10σ event. Thus neither thermal motion nor disorder is dangerous, according to our estimates, to prevent nucleated pancake vortices from annihilating with extremely high probability at the end of every cycle.

where xthermal is the displacement in time t due to thermal motion and D is a diffusion constant. For one RF cycle at frequency f, t = f −1 . D ≈ µkBT, where k<sup>B</sup> is Boltzmann's constant, T is the temperature and µ is the mobility of the vortex, given by Bardeen Stephen as ρn/(Hc<sup>2</sup> φ<sup>0</sup> d), where ρ<sup>n</sup> is the normal state resistivity, Hc<sup>2</sup> is the upper critical field and φ<sup>0</sup> is the flux quantum. Solving, the wandering due to thermal motion is given by:

$$
\sqrt{\langle x\_{thermal}^2 \rangle} = \sqrt{\frac{2 \, k\_B \, T \, \rho\_n}{H\_{c2} \, df \, \phi\_0}} \tag{29}
$$

For realistic parameters T = 2 K, ρ<sup>n</sup> = 100 nΩm; µ0Hc<sup>2</sup> = 30 T, f = 1.3 GHz, d = 30 nm, xthermal = 1.5 nm, as shown in Fig. [\(13\)](#page-15-2). From these results, we can calculate the approximate expected rate of production of vortices that fail to annihilate. One expects that the distribution of final separations will be Gaussian with standard deviation xthermal, suggesting that the number of vortices which do not annihilate will be given by the tail of the Gaussian. For example, at H = 0.8Hsh, ximp is about 22 nm from Fig. [\(13\)](#page-15-2), or about 15 standard deviations, making it extremely unlikely for vortices to escape due to thermal meandering alone.

### <span id="page-15-0"></span>C. Disorder Meandering

![](0__page_15_Figure_8.jpeg)

**Caption:** The diagram demonstrates the role of surface disorders in altering the path of vortices through screening currents, affecting the performance of superconducting films in an SIS' framework. This illustration underscores the influence of surface imperfections on superconducting behavior, elaborating how such disorders may lead to greater vortex meandering and energy dissipation, critical in evaluating materials for SRF cavity applications.


<span id="page-15-3"></span>FIG. 14. Surface disorder may cause pancake vortices to meander away from a nucleation site and build up in a film over many RF cycles.

To calculate the wandering due to surface disorder, illustrated in Fig. [\(14\)](#page-15-3), we consider a single-cell f=1.3 GHz niobium SRF cavity with an SIS structure using d = 30 nm thick Nb3Sn layers. Assume that the topmost S layer has a normally-distributed random array of defects over its surface. For our geometry, we divide the L×L (where L ∼10 cm) surface area of the cavity into N a×a regions of order the pancake vortex size, where N = L <sup>2</sup>/a<sup>2</sup> . We represent the effect of these defects as lowering the local value of B<sup>c</sup> in a given region. Therefore these defects will nucleate vortex penetration, and they will attract pancake vortices in the film. At the worst of the defects, the expected value for H<sup>c</sup> is αHc,nominal, where α is a constant between 0 and 1. At this defect, vortices penetrate at approximately H = αHsh. We represent the surface of the cavity with a distribution H of values for the reduction in the square of Hc. For simplicity of analysis, and
since the defects are normally distributed, we will use the notation generally used in propagation of random errors.

<span id="page-16-3"></span>
$$\mathcal{H} = H\_{c,nominal}^2 (1 - |(0 \pm \sigma)|) \tag{30}$$

Here is σ is the variance of the normally distributed H<sup>2</sup> c reduction. We use absolute values because there should be no H<sup>c</sup> values higher than the nominal value.

We can find σ using our restriction that the expected value for H<sup>c</sup> at the worst defect is αHc,nominal. To do this, we need to work with a normal distribution of the H<sup>2</sup> c reduction in our N regions N (0, σ) = [1/(σ √ 2π)] exp[x <sup>2</sup>/(2 σ 2 )]. First, we need to find φ, the value of x above which lies just one half of one of our N regions (one half because the absolute value effectively doubles the number of samples in our integration).

<span id="page-16-0"></span>
$$\int\_{\phi}^{\infty} \frac{1}{\sigma \sqrt{2\pi}} e^{-x^2/(2\sigma^2)} \mathrm{d}x = \frac{1}{2} \left[ 1 - \mathrm{erf} \left( \frac{\phi}{\sqrt{2}\sigma} \right) \right] = \frac{1}{2N} \tag{31}$$

Next, we set the expected value of the distribution in this region to be 1-α 2 . This sets the expected value of H<sup>c</sup> to be αH<sup>c</sup> for the most extreme defect (we also have to normalize for there being only one defect in our sample size). This defines σ.

<span id="page-16-1"></span>
$$\int\_{\phi}^{\infty} \frac{x}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x}{\sigma}\right)^2} dx = \frac{\sigma}{\sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{\phi}{\sigma}\right)^2} = \frac{1-\alpha^2}{2N} \quad (32)$$

To obtain an analytical expression, instead of solving equations [\(31\)](#page-16-0) and [\(32\)](#page-16-1), we can approximate. First, estimate that φ ≈ α:

$$\int\_{\alpha}^{\infty} \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{\pi}{\sigma}\right)^2} \mathrm{d}x = \frac{1}{2N} \tag{33}$$

We can also use a linear approximation for the Gaussian, so that the integral can be evaluated by calculating the area of the triangle shown in Fig. [\(15\)](#page-16-2). The equation of the line evaluated at y = 0 determines the length ∆x in the figure:

$$0 = \frac{-\sigma\alpha}{\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{\alpha}{\sigma}\right)^2}(\Delta x) + \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{\alpha}{\sigma}\right)^2} \tag{34}$$

This gives ∆x = σ <sup>2</sup>/α. Evaluating the integral,

$$\int\_{\alpha}^{\infty} \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x}{\sigma}\right)^2} dx = \frac{\sigma}{2\alpha \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{\alpha}{\sigma}\right)^2} = \frac{1}{2N} \quad (35)$$

Rearranging,

$$\frac{\alpha^2}{2\sigma^2} = \ln \frac{\sigma N}{\alpha \sqrt{2\pi}}\tag{36}$$

![](1__page_16_Figure_16.jpeg)

**Caption:** This figure illustrates a Gaussian distribution, where the x-axis represents the width parameter, and α signifies a standard deviation or scale factor. This visualization is likely used to represent some form of statistical distribution or uncertainty that is pertinent to the experiment's parameters, like disorder effects on superconducting properties, though further context is required for a full interpretation. The red triangle and Δx potentially indicate a specific range relevant to experimental conditions. Understanding statistical variation aids in the quantitative analysis of phenomena such as thermal fluctuations or material defects.


<span id="page-16-2"></span>FIG. 15. Approximations to find an analytical expression for σ.

N is very large, so ln σN α √ 2π ≈ ln N. Solving,

$$
\sigma = \frac{\alpha}{\sqrt{2\ln N}}\tag{37}
$$

Now let us consider the behavior of the cavity at a field just above the expected vortex penetration field of the worst defect. Let ˆx point into the film and ˆz be aligned with the magnetic field. In this case, pairs of pancake vortices will form at the defect, move apart in y due to the force exerted by the increasing magnetic field, and then move in the opposite direction in y as the magnetic field direction reverses. In the mean time, they will have meandered some distance in x. If the meandering distance is very small compared to the impact parameter (approximately 50 nm), the pancake vortex pairs are most likely to meet again and annihilate. However, if the meandering distance is significant compared to the impact parameter, vortex pairs are likely to get lost, accumulating over the film.

For this calculation, we will not do a full simulation. Rather, we will consider a path along y of a vortex pancake moving from the defect at y = 0 to the extremum y = ymax, without any movement in x. We will, however, integrate the force in x that the pancake would experience along its path, and calculate what the expected meandering distance would be from this.

The vortex will have two regions on either side of it in x at a given time, one with Hc,1, and one with Hc,2. The centers of these two regions are separated by a distance a. The x-component of the force experienced by the vortex can be approximated from the gradient in condensation energy resulting from the difference in the local Hc. Magnetic energy density is given by H<sup>2</sup>/2µ0. We convert this to energy by multiplying by the volume of a region, a <sup>2</sup>d.

<span id="page-16-4"></span>
$$F\_x = \frac{H\_{c,1}^2 - H\_{c,2}^2}{2\mu\_0 a} a^2 d\tag{38}$$

The motion of the pancake is described by Bardeen-Stephen drag:

<span id="page-17-0"></span>
$$
\eta v\_x = F\_x / d \tag{39}
$$

Here v<sup>x</sup> is the x-component of the velocity of the pancake and η is the Bardeen-Stephen drag coefficient, defined by η = Hc2φ0/ρn, where Hc<sup>2</sup> is the upper critical field of the film, φ<sup>0</sup> is the flux quantum, and ρ<sup>n</sup> is the normal resistivity of the film.

The meandering distance ∆x within a region is given by ∆x = v∆t, where ∆t is the time spent in that region. For simplicity, we approximate that the vortex moves with uniform speed in y, spending equal time in each of the regions that it travels through, such that ∆t = 1 2f a ymax .

Using Eq. Eq. [\(30\)](#page-16-3) and Eq. [\(38\)](#page-16-4), we can describe a distribution F<sup>x</sup> of forces that the pancake would experience. We observe that taking the difference between the absolute values from Eq. [\(30\)](#page-16-3) produces a normal distribution:

<span id="page-17-1"></span>
$$\mathcal{F}\_x = \frac{adH\_{c,nominal}^2}{2\mu\_0} (0 \pm \sigma) \tag{40}$$

Between two regions that are adjacent in x, the pancake will experience a single value of the distribution of force values. After it travels a distance a, it will encounter a new pair of regions and therefore a new force value. We can use ∆x = v∆t and Eq. [\(39\)](#page-17-0) to find p h∆x <sup>2</sup>i, the RMS wandering distance traveled by the vortex over a distance a:

$$
\sqrt{\langle \Delta x^2 \rangle} = \sqrt{\langle F^2 \rangle} \frac{\Delta t}{d\eta} \tag{41}
$$

Multiply by the square root of the number of steps in a period Nsteps = 2ymax a to find the total RMS wandering distance p h∆X<sup>2</sup>i. Use Eq. [\(40\)](#page-17-1).

$$\sqrt{\langle \Delta X^2 \rangle} = \frac{ad\sigma H\_{c,nominal}^2}{2\mu\_0} \frac{\Delta t}{d\eta} \sqrt{\frac{2y\_{max}}{a}} \qquad (42)$$

Since ∆t = a v<sup>y</sup> , from Eq. [\(39\)](#page-17-0) and using F<sup>y</sup> = φ0H µ<sup>0</sup> , we obtain:

<span id="page-17-2"></span>
$$\frac{\Delta t}{\eta} = \frac{ad}{F\_y} = \frac{ad\mu\_0}{\phi\_0 \Delta H} \tag{43}$$

Here ∆H is the difference in magnetic field across the film. If we are looking just above the penetration field, ∆H ≈ αHshd/λ. Eq. [\(43\)](#page-17-2) can also be used to find <sup>y</sup>max a .

$$\frac{y\_{\max}}{a} = \frac{1}{2f\Delta t} = \frac{\phi\_0 \alpha H\_{\text{sh}}}{2f\eta a \lambda \mu\_0} \tag{44}$$

Use the definition of η.

$$\frac{y\_{max}}{a} = \frac{\rho\_n \alpha H\_{\rm sh}}{2fH\_{c2}a\lambda\mu\_0} \tag{45}$$

Now substitute.

$$\sqrt{\langle \Delta X^2 \rangle} = \frac{1}{2\sqrt{\ln N}} \frac{a^2 \lambda H\_{c,nominal}^2}{\phi\_0 H\_{sh}} \sqrt{\frac{\rho\_n \alpha H\_s h}{f H\_{c2} a \lambda \mu\_0}} \quad (46)$$

We then use Hc,nominal = φ<sup>0</sup> 2 √ <sup>2</sup>πλξ .

$$\sqrt{\langle \Delta X^2 \rangle} = \frac{\xi}{4\sqrt{2}\pi\sqrt{\ln N}} \frac{H\_{c,nominal}}{H\_{\rm sh}} \sqrt{\frac{\rho\_n \alpha H\_{\rm sh}}{fH\_{c2}a\lambda\mu\_0}} \tag{47}$$

We can set our region size a = ξ ≈ 4 nm. Hsh/Hc,nominal for Nb3Sn is approximately 0.75. N is approximately 10<sup>15</sup>, so <sup>1</sup> 4 √ 2π √ ln N is approximately 10−<sup>2</sup> . We also use Hc2=30 T, ρn=10 nΩm, and Hc,nominal = 530 mT. For α = 0.8, these factors give p h∆X<sup>2</sup>i ≈ 2.1 nm. Setting this as a standard deviation for vortex meandering and using ximp ∼22 nm from above gives a result of 11 standard deviations, again making it extremely unlikely for vortices to escape due to disorder meandering.

Varying a and α over a reasonable range also gives a small value for the meandering distance, as seen in Fig. [\(16\)](#page-18-0).

In addition to the impact parameter and thermal meandering distance, Fig. [\(13\)](#page-15-0) plots the meandering distance due to disorder using the analytical formulation above. Also plotted in the figure is a simulation of vortex pair creation and annihilation in the presence of disorder. In addition to forces from the applied magnetic field and from the randomly distributed array of circular defects with radius a, [11](#page-17-3) the simulation considers the forces of one vortex on the other, and finds the maximum lateral separation between the pair during a cycle.

## D. Interacting vortices, interacting defects

We have presented analyses of thermal meandering and meandering of a vortex-antivortex pair due to disorder, and so far, nothing has caused a large buildup of vortices in the film. We have also performed a preliminary analysis of the nucleation of many vortices at a defect over the course of an RF cycle, all of which interact with

<span id="page-17-3"></span><sup>11</sup> In the simulation, the defects pin vortices with force that increases linearly from the defect edge where it is zero, to the center of the defect, where it is a maximum. The maximum is set such that moving the vortex from the center of the defect to the edge would require work equal to the condensation energy of the volume of the vortex core.

![](1__page_18_Figure_0.jpeg)

**Caption:** This figure explores pancake vortex meandering within a Nb3Sn superconducting film against two parameters. In subfigure (a), the mean displacement (sqrt(AVERAGE[X^2])) is shown increasing with region size a, highlighting disorder effects at a fixed damping constant α = 0.8. Subfigure (b) reflects the influence of variation in α for a constant region size (4 nm). The results illustrate how disorder can facilitate vortex deviation, crucial for predicting the magnitude of magnetic field distortions in superconducting materials.


<span id="page-18-0"></span>FIG. 16. The effect of varying a for fixed α = 0.8 (a) and varying α for fixed a = 4 nm (b) on the mean meandering distance of a pancake vortex due to disorder.

each other. While an attractive force exists between vortices and antivortices, vortices exert a repulsive force on other vortices, and antivortices exert a repulsive force on other antivortices. These repulsive forces between similar vortex types appear to result in substantially larger lateral movement, which could lead to higher rates of non-annihilation. An example video is included as supplemental material [\[86\]](#page-21-0); three frames of which are shown in Fig. [\(17\)](#page-18-1). Here the film thickness is 30 nm, parameters are appropriate for Nb3Sn material, and the disorder is modeled as 60 pinning sites randomly spread over 0.4 square microns that exert a force of Fmax k<sup>i</sup> ρ, where Fmax = B<sup>2</sup> <sup>c</sup> µ<sup>0</sup> ξ <sup>2</sup> d rpin/ 2, k<sup>i</sup> is a value between 0 and 1 that is randomly chosen for each pinning site, and ρ = rsep/rpin for rsep < rpin and 0 for rsep > rpin, where rsep is the distance between the vortex and the center of the pinning site and rpin is the pinning site radius, (here chosen to be 2 nm). Note that the maximum horizontal meandering due to disorder and interactions is roughly eight nanometers, only a factor of three less than the impact parameter ximp, suggesting that interactions may

be much more dangerous for multiple defects over many cycles. However, further work on interactions is clearly needed; it is possible that realistic parameters for the disorder strength could decrease the meandering, and it is possible that the distribution of maximum meandering distances near a defect over multiple cycles could be sub-Gaussian. On the other hand, it is likely that two defects that both nucleate pancake vortices and are close enough together that the vortices interact could be dangerous even in the absence of disorder.

![](1__page_18_Figure_5.jpeg)

**Caption:** This panel series, labeled (a), (b), and (c), provides a comparative visualization of parameters influencing pancake vortex dynamics in superconducting films with an SIS' layer structure. These graphs depict changes in mean parallel and perpendicular field strength due to interaction with disorder and other factors in successive RF cycles, indicating significant insights into the lateral meandering affected by thermal and disorder influences. The assessment of these factors helps in comprehending how structural elements within superconductors impact vortex behavior, a key factor in designing effective superconducting film architectures.


<span id="page-18-1"></span>FIG. 17. Frames from a preliminary simulation of pancake vortices (red) and antivortices (blue) being generated at a defect (green) in a thin film in an AC field. The repulsion between similar vortices causes a lateral spread. Note the horizontal and vertical scales differ by over a factor of 100. The the horizontal lines are actually circular representations of the disorder potential; the vortices drift sideways much less than they traverse vertically. (a) Near zero field, (b) Near a local maximum field, and (c) Near a return to zero field, showing the cycle-to-cycle variation.

## VI. CONCLUSIONS

This article attempts to make a case for thoughtful, broad efforts to identify and estimate fundamental physical limits of materials in parallel with experimental efforts. Our estimates for the superheating field of pure materials [\[14,](#page-20-0) [15\]](#page-20-1) had a significant impact in the SRF community, we understand, not because it promised that Nb3Sn could be run at twice the field of Nb. Rather, we pointed out that the 'vortex line nucleation' model was incorrect (footnote [4](#page-3-0) on page [4\)](#page-3-0)). This model, created by and initially trusted in the SRF community, had provided discouraging estimates for high κ materials, misleading the field for some years. We also note that our controlled, theoretically grounded calculations allow one to explore questions like materials anisotropy that cannot be gleaned reliably from phenomenological models (or, indeed, from our own qualitative arguments, Section [\(IV B\)](#page-8-0)).

We understand that many in the SRF community are skeptical of the use of bulk (or thick films) of new materials with higher κ, even though the theoretical estimates suggest significantly improved Hsh as well as lower cooling requirements. We too were concerned until recently that the smaller coherence lengths might make the metastable state more susceptible to vortex penetration. But we believe that our calculation of the effects of disorder within a particular model clearly indicates that the reliability of the new materials increases so rapidly away from Hsh that the effects of lower coherence length should not be a major concern. One must always make choices of where to focus resources (laminates versus bulk materials, coated copper versus niobium [\[87\]](#page-21-1); an interesting review has been recently published in this special issue [\[74\]](#page-21-2)) – but informed choices may involve consultation with experts outside the SRF field.

We are also guardedly pessimistic about the utility of thin laminates in increasing performance. (a) We are concerned that experimentalists continue to be inspired by the very high parallel fields sustainable by isolated thin films [\[74,](#page-21-2) Section 9] (footnote [10](#page-13-0) on page [14\)](#page-13-0), even though these high fields are irrelevant in GHz applications [\[76\]](#page-21-3) and likely also practically inaccessible in any AC application. (b) Dangerous vortices in thin films will not typically reside parallel and inside the films, but penetrate in and out of the film via pancake vortices, whose motion dissipates heat. The maximum fields reachable without flux penetration for ideal thin films are rather similar to those of the bulk material [\[45\]](#page-20-2). The flux penetration needed to reach higher fields produces heating per cycle comparable to that for copper cavities [\[76\]](#page-21-3). (c) We explore extensively the suggestion [\[73\]](#page-21-4) that the insulating layers in laminates may act to trap flux from defects, keeping flux from entering the bulk. Here the key issue is whether the nucleated pancake vortices escape from the vicinity of their parent flaw, or annihilate with their partners at the end of each cycle. Modeling these as sources for pancake vortices, we agree that neither thermal diffusion nor disorder are dangerous, but that interactions between vortices, and between vortices generated at separate defects, might allow for escape and unacceptable heating – warranting experimental caution and further theoretical study.

Finally, a word to our colleagues outside the accelerator community. This paper is a collaboration of SRF experts from the accelerator community (Posen, Liepe) and condensed matter theorists, and we draw heavily on conversations with both theorists and experimentalists inside the community (Hasan Padamsee, Alex Gurevich, Nicholas Valles, Takayuki Kubo, Kenji Saito). These domain-specific experts have enormous experience in the challenges and issues relevant for the field; we were told that superheating fields, higher κ materials, anisotropy, disorder, and laminates were the key questions, and have been guided into studying these in the correct limits and focusing on the right issues. We can testify that this teamwork has led to both excellent condensed-matter physics and efficient, targeted research to improve SRF performance.

## ACKNOWLEDGMENT

Our work on the superheating field was urged upon us by Hasan Padamsee, who recognized both the importance of firm estimates of the theoretical maximum performance for niobium cavities, and the confusion in the SRF field at that time about potential new materials. We also thank Alex Gurevich for extensive consultation and collaboration, particularly on the new work on laminates in Section [\(V\)](#page-13-1). Much of that section was inspired by our conversations with him and tests his independently derived thoughts and conclusions on the subject. DBL and JPS were supported by NSF DMR-1312160, Sam Posen is supported by the United States Department of Energy, Offices of High Energy Physics. Fermilab is operated by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the United States Department of Energy. GC acknowledges partial support by the EU under REA Grant Agreement No. CIG-618258. ML was supported by NSF Award No. PHY-1416318, and DOE Award No. DE-SC0008431. This work was supported by the U.S. National Science Foundation under Award OIA-1549132, the Center for Bright Beams.

- [1] T. Behnke, J. E. Brau, B. Foster, J. Fuster, M. Harrison, J. M. Paterson, M. Peskin, M. Stanitzki, N. Walker, and
- H. Yamamoto, The International Linear Collider, Tech.

Rep. (Technical Design Report, 2013).

- [2] P. Bernard, D. Bloess, and T. Flynn, [Proceedings of the](https://accelconf.web.cern.ch/accelconf/SRF91/papers/srf91e12.pdf?origin=publication{_}detail) [Third European Particle Accelerator Conference , 1269](https://accelconf.web.cern.ch/accelconf/SRF91/papers/srf91e12.pdf?origin=publication{_}detail) [\(1992\).](https://accelconf.web.cern.ch/accelconf/SRF91/papers/srf91e12.pdf?origin=publication{_}detail)
- [3] P. Kneisel, B. Lewis, and L. Turlington, [Proceedings of](http://www.jlab.org/div{_}dept/admin/publications/papers/93/PR93-58.pdf) [the Sixth Workshop on RF Superconductivity \(1993\).](http://www.jlab.org/div{_}dept/admin/publications/papers/93/PR93-58.pdf)
- [4] D. Proch and U. Klein, in Proceedings of the Conference for Future Possibilities for Electron Accelerators (1979) pp. N1–17.
- [5] M. Tinkham, Introduction to Superconductivity, 2nd ed. (McGraw-Hill, New York, 1996).
- [6] G. Eilenberger, Z. Phys. 214, 195 (1968).
- [7] G. Eliashberg, Sov. Phys.-JETP (Engl. Transl.);(United States) 11, 696 (1960).
- [8] I. Lifshitz, [Advances in Physics](http://dx.doi.org/10.1080/00018736400101061) 13, 483 (1964), [http://dx.doi.org/10.1080/00018736400101061.](http://arxiv.org/abs/http://dx.doi.org/10.1080/00018736400101061)
- [9] J. Zittartz and J. S. Langer, Phys. Rev. 148[, 741 \(1966\).](http://dx.doi.org/10.1103/PhysRev.148.741)
- [10] J. S. Langer, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.21.973) 21, 973 (1968).
- [11] B. C. Daniels and J. P. Sethna, [Phys. Rev. E](http://dx.doi.org/10.1103/PhysRevE.83.041924) 83, 041924 [\(2011\).](http://dx.doi.org/10.1103/PhysRevE.83.041924)
- [12] R. P. Huebener, Magnetic Flux Structures in Superconductors (Springer-Verlag, Berlin, 2001).
- [13] H. Padamsee, RF Superconductivity: Sicence, Technology, and Applications (WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim, 2009).
- <span id="page-20-0"></span>[14] M. K. Transtrum, G. Catelani, and J. P. Sethna, [Phys.](http://dx.doi.org/10.1103/PhysRevB.83.094505) Rev. B 83[, 094505 \(2011\).](http://dx.doi.org/10.1103/PhysRevB.83.094505)
- <span id="page-20-1"></span>[15] G. Catelani and J. P. Sethna, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.78.224509) 78, 224509 [\(2008\).](http://dx.doi.org/10.1103/PhysRevB.78.224509)
- [16] M. P. Garfunkel and B. Serin, Phys. Rev. 85[, 834 \(1952\).](http://dx.doi.org/10.1103/PhysRev.85.834)
- [17] V. L. Ginzburg, Soviet Phys.—JETP 7, 78 (1958).
- [18] L. Kramer, Phys. Rev. 170[, 475 \(1968\).](http://dx.doi.org/10.1103/PhysRev.170.475)
- [19] P. de Gennes, [Solid State Communications](http://dx.doi.org/ http://dx.doi.org/10.1016/0038-1098(65)90387-X) 3, 127 (1965).
- [20] V. P. Galaiko, Sov. Phys. JETP 23, 475 (1966).
- [21] L. Kramer, [Zeitschrift f¨ur Physik](http://dx.doi.org/10.1007/BF01395939) 259, 333 (1973).
- [22] H. J. Fink and A. G. Presson, Phys. Rev. 182[, 498 \(1969\).](http://dx.doi.org/10.1103/PhysRev.182.498)
- [23] P. Christiansen, [Solid State Communications](http://dx.doi.org/ http://dx.doi.org/10.1016/0038-1098(69)90652-8) 7, 727 [\(1969\).](http://dx.doi.org/ http://dx.doi.org/10.1016/0038-1098(69)90652-8)
- [24] S. J. Chapman, [SIAM Journal on Applied Mathematics](http://dx.doi.org/10.1137/S0036139993254760) 55[, 1233 \(1995\).](http://dx.doi.org/10.1137/S0036139993254760)
- [25] A. J. Dolgert, S. J. Di Bartolo, and A. T. Dorsey, [Phys.](http://dx.doi.org/10.1103/PhysRevB.53.5650) Rev. B 53[, 5650 \(1996\).](http://dx.doi.org/10.1103/PhysRevB.53.5650)
- [26] C. P. Bean and J. D. Livingston, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.12.14) 12, 14 [\(1964\).](http://dx.doi.org/10.1103/PhysRevLett.12.14)
- [27] J. B. Ketterson and S. N. Song, Superconductivity (Cambridge University Press, Cambridge, 1999).
- [28] K. M. Yoo, X. M. Zhao, M. Siddique, R. R. Alfano, D. P. Osterman, M. Radparvar, and J. Cunniff, "Rate of electron—phonon relaxation in niobium," in [Ultra](http://dx.doi.org/ 10.1007/978-3-642-84269-6_109)[fast Phenomena VII: Proceedings of the 7th International](http://dx.doi.org/ 10.1007/978-3-642-84269-6_109) [Conference, Monterey, CA, May 14–17, 1990](http://dx.doi.org/ 10.1007/978-3-642-84269-6_109) , edited by C. B. Harris, E. P. Ippen, G. A. Mourou, and A. H. Zewail (Springer Berlin Heidelberg, Berlin, Heidelberg, 1990) pp. 357–359.
- [29] T. P. Orlando, E. J. McNiff, S. Foner, and M. R. Beasley, [Physical Review B](http://dx.doi.org/ 10.1103/PhysRevB.19.4545) 19, 4545 (1979).
- [30] M. Hein, [High-Temperature-Superconductor Thin Films](http://books.google.com/books?id=5MIDpYI1z20C{&}pgis=1) [at Microwave Frequencies](http://books.google.com/books?id=5MIDpYI1z20C{&}pgis=1) (Springer, New York, 1999).
- [31] J. Harden and V. Arp, Cryogenics 3[, 105 \(1963\).](http://dx.doi.org/10.1016/0011-2275(63)90029-8)
- [32] B. Maxfield and W. McLean, [Physical Review](http://dx.doi.org/10.1103/PhysRev.139.A1515) 139, [A1515 \(1965\).](http://dx.doi.org/10.1103/PhysRev.139.A1515)
- [33] D. Oates, A. Anderson, C. Chin, J. Derov, G. Dresselhaus, and M. Dresselhaus, [Physical Review B](http://dx.doi.org/10.1103/PhysRevB.43.7655) 43, 7655 [\(1991\).](http://dx.doi.org/10.1103/PhysRevB.43.7655)
- [34] Y. Wang, T. Plackowski, and A. Junod, [Physica C: Su](http://dx.doi.org/10.1016/S0921-4534(01)00617-7)[perconductivity](http://dx.doi.org/10.1016/S0921-4534(01)00617-7) 355, 179 (2001).
- [35] J. R. Schrieffer, Theory of Superconductivity (Westview Press, 1999).
- [36] D. B. Liarte, M. K. Transtrum, and J. P. Sethna, [Phys.](http://dx.doi.org/10.1103/PhysRevB.94.144504) Rev. B 94[, 144504 \(2016\).](http://dx.doi.org/10.1103/PhysRevB.94.144504)
- [37] T. Yogi, G. J. Dick, and J. E. Mercereau, [Phys. Rev.](http://dx.doi.org/10.1103/PhysRevLett.39.826) Lett. 39[, 826 \(1977\).](http://dx.doi.org/10.1103/PhysRevLett.39.826)
- [38] K. Saito, in Proceedings of the 11th International Conference on RF Superconductivity, Travemunde (2004).
- [39] J. Langer, Reviews of Modern Physics 52, 1 (1980).
- [40] R. C. Brower, D. A. Kessler, J. Koplik, and H. Levine, Physical review letters 51, 1111 (1983).
- [41] A. Sharma and R. Khanna, Physical Review Letters 81, 3463 (1998).
- [42] G. I. Sivashinsky, Annual Review of Fluid Mechanics 15, 179 (1983).
- [43] E. Bodenschatz, W. Pesch, and G. Ahlers, Annual review of fluid mechanics 32, 709 (2000).
- [44] E. Ben-Jacob, N. Goldenfeld, J. Langer, and G. Sch¨on, Physical Review A 29, 330 (1984).
- <span id="page-20-2"></span>[45] S. Posen, M. K. Transtrum, G. Catelani, M. U. Liepe, and J. P. Sethna, [Phys. Rev. Applied](http://dx.doi.org/10.1103/PhysRevApplied.4.044019) 4, 044019 (2015).
- [46] I. Campisi, [IEEE Trans. Magn.](http://dx.doi.org/10.1109/TMAG.1985.1063638) 21, 134 (1985).
- [47] T. Hays and H. Padamsee, in Proc. Eighth Work. RF Supercond. (Padova, 1997) pp. 789–794.
- [48] B. Visentin, TESLA Meeting, DESY 98-05 (1998).
- [49] P. Kneisel, [Proceedings of the Ninth Workshop on RF](http://accelconf.web.cern.ch/accelconf/SRF99/papers/tup044.pdf) [Superconductivity , 328 \(1999\).](http://accelconf.web.cern.ch/accelconf/SRF99/papers/tup044.pdf)
- [50] S. Posen, N. Valles, and M. Liepe, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.115.047001) 115, [047001 \(2015\).](http://dx.doi.org/10.1103/PhysRevLett.115.047001)
- [51] J. Halbritter, Intern. Note KFZ Karlsruhe 3/70-6 (1970).
- [52] J. Halbritter, [Zeitschrift fr Phys.](http://dx.doi.org/10.1007/BF01409429) 238, 466 (1970).
- [53] V. Chandrasekhar, in Superconductivity, edited by K. H. Bennemann and J. B. Ketterson (Springer, Berlin, 2008) Chap. 8.
- [54] J. Rammer, Quantum Field Theory of Non-equilibrium States (Cambridge University Press, 2007).
- [55] F. P.-J. Lin and A. Gurevich, Physical Review B 85, 054513 (2012).
- [56] V. Kogan and S. Budko, [Physica C: Superconductivity](http://dx.doi.org/ http://dx.doi.org/10.1016/S0921-4534(02)02293-1) 385[, 131 \(2003\).](http://dx.doi.org/ http://dx.doi.org/10.1016/S0921-4534(02)02293-1)
- [57] S. L. Bud'ko and P. C. Canfield, [Physica C: Supercon](http://dx.doi.org/http://dx.doi.org/10.1016/j.physc.2015.02.024)[ductivity and its Applications](http://dx.doi.org/http://dx.doi.org/10.1016/j.physc.2015.02.024) 514, 142 (2015), superconducting Materials: Conventional, Unconventional and Undetermined.
- [58] V. L. Ginzburg, Zh. Eksper. Teor. Fiz. 23, 236 (1952).
- [59] C. Caroli, P. G. de Gennes, and J. Matricon, Physik Kondensierten Materie 1, 176 (1966).
- [60] L. P. Gor'kov and T. K. Melik-Barkhudarov, Zh. Eksper. Teor. Fiz. 45, 1493 (1964).
- [61] D. R. Tilley, [Proceedings of the Physical Society](http://stacks.iop.org/0370-1328/86/i=2/a=305) 86, 289 [\(1965\).](http://stacks.iop.org/0370-1328/86/i=2/a=305)
- [62] R. Cubitt, S. Levett, S. L. Bud'ko, N. E. Anderson, and P. C. Canfield, [Phys. Rev. Lett.](http://dx.doi.org/ 10.1103/PhysRevLett.90.157002) 90, 157002 (2003).
- [63] R. Cubitt, M. R. Eskildsen, C. D. Dewhurst, J. Jun, S. M. Kazakov, and J. Karpinski, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.91.047002) 91, 047002 [\(2003\).](http://dx.doi.org/10.1103/PhysRevLett.91.047002)
- [64] S. Posen, M. Liepe, and D. L. Hall, [Ap](http://dx.doi.org/http://dx.doi.org/10.1063/1.4913247)[plied Physics Letters](http://dx.doi.org/http://dx.doi.org/10.1063/1.4913247) 106, 082601 (2015), [http://dx.doi.org/10.1063/1.4913247.](http://dx.doi.org/http://dx.doi.org/10.1063/1.4913247)
- [65] I. S. Gradschteyn and I. M. Ryzhic, Table of integrals, series and products (Academic Press, 2007).
- [66] G. Eremeev and H. Padamsee, in Proceedings of EPAC 2006, Edinburgh, Scotland (2006).
- [67] P. Kneisel, G. R. Myneni, G. Ciovati, J. Sekutowicz, and T. Carneiro, in Proceedings of 2005 Particle Accelerator Conference, Knoxville, Tennesseee (2005).
- [68] P. Kneisel, G. R. Myneni, G. Ciovati, J. Sekutowicz, and T. Carneiro, in Proceedings of the 12th International Workshop on RF Superconductivity, Cornell University, Ithaca, New York, USA (200).
- [69] F. Bass, V. Freilikher, B. Shapiro, and M. Shvartser, [Physica C: Superconductivity](http://dx.doi.org/http://dx.doi.org/10.1016/0921-4534(96)00134-7) 260, 231 (1996).
- [70] T. Kubo, [Progress of Theoretical and Experimental](http://dx.doi.org/10.1093/ptep/ptv082) Physics 2015 [\(2015\), 10.1093/ptep/ptv082.](http://dx.doi.org/10.1093/ptep/ptv082)
- [71] R. L. Geng, in Proceedings of SRF2011, Chicago, IL USA (2011).
- [72] A. Gurevich, ["On the theoretical RF field limits of mul](http://arxiv.org/abs/1309.5626)[tilayer coating structures of superconducting resonator](http://arxiv.org/abs/1309.5626) [cavities,"](http://arxiv.org/abs/1309.5626) (2013).
- <span id="page-21-4"></span>[73] A. Gurevich, [Appl. Phys. Lett.](http://dx.doi.org/10.1063/1.2162264) 88, 012511 (2006).
- <span id="page-21-2"></span>[74] A.-M. Valente-Feliciano, [Superconductor Science and](http://stacks.iop.org/0953-2048/29/i=11/a=113002) Technology 29[, 113002 \(2016\).](http://stacks.iop.org/0953-2048/29/i=11/a=113002)
- [75] G. Lamura, M. Aurino, A. Andreone, and J.-C. Vill´egier, [Journal of Applied Physics](http://dx.doi.org/http://dx.doi.org/10.1063/1.3211321) 106, 053903 (2009), [http://dx.doi.org/10.1063/1.3211321.](http://dx.doi.org/http://dx.doi.org/10.1063/1.3211321)
- <span id="page-21-3"></span>[76] S. Posen, G. Catelani, M. U. Liepe, J. P. Sethna, and M. K. Transtrum, "Theoretical Field Limits for Multi-Layer Superconductors," (2013), [arXiv:1309.3239](http://arxiv.org/abs/1309.3239) [\[physics.acc-ph\].](http://arxiv.org/abs/1309.3239)
- [77] T. Kubo, ["Multilayer coating for high gradients,"](http://arxiv.org/abs/1607.01495) (2016), [arXiv:1607.01495.](http://arxiv.org/abs/1607.01495)
- [78] M. Checchin, A. Grassellino, M. Martinello, S. Posen, A. Romanenko, and J. F. Zasadzinski, Proceedings of the Seventh International Particle Accelerator Conference , 2254 (2016).
- [79] I. Aranson, A. Gurevich, and V. Vinokur, [Phys. Rev.](http://dx.doi.org/10.1103/PhysRevLett.87.067003) Lett. 87[, 067003 \(2001\).](http://dx.doi.org/10.1103/PhysRevLett.87.067003)
- [80] I. S. Aranson, A. Gurevich, M. S. Welling, R. J. Wijngaarden, V. K. Vlasko-Vlasov, V. M. Vinokur, and U. Welp, [Phys. Rev. Lett.](http://dx.doi.org/ 10.1103/PhysRevLett.94.037002) 94, 037002 (2005).
- [81] L. Cort´e, P. M. Chaikin, J. P. Gollub, and D. J. Pine, Nature Physics 4, 420 (2008).
- [82] D. P´erez Daroca, G. Pasquini, G. S. Lozano, and V. Bekeris, Phys. Rev. B 84[, 012508 \(2011\).](http://dx.doi.org/10.1103/PhysRevB.84.012508)
- [83] A. Motohashi and S. Okuma, [Journal of Physics: Con](http://stacks.iop.org/1742-6596/302/i=1/a=012029)ference Series 302[, 012029 \(2011\).](http://stacks.iop.org/1742-6596/302/i=1/a=012029)
- [84] S. Okuma, Y. Tsugawa, and A. Motohashi, [Phys. Rev.](http://dx.doi.org/10.1103/PhysRevB.83.012503) B 83[, 012503 \(2011\).](http://dx.doi.org/10.1103/PhysRevB.83.012503)
- [85] J. Clem, [Physical Review B](http://prb.aps.org/abstract/PRB/v43/i10/p7837{_}1) 43, 7837 (1991).
- <span id="page-21-0"></span>[86] S. Posen, M. Liepe, and J. P. Sethna, "Pancake vortex dynamics in an AC field," [http://www.lassp.cornell.](http://www.lassp.cornell.edu/sethna/Superconductivity/VortexDynamics/pancakevideo.gif) [edu/sethna/Superconductivity/VortexDynamics/](http://www.lassp.cornell.edu/sethna/Superconductivity/VortexDynamics/pancakevideo.gif) [pancakevideo.gif](http://www.lassp.cornell.edu/sethna/Superconductivity/VortexDynamics/pancakevideo.gif) (2016).
- <span id="page-21-1"></span>[87] R. Russo, L. Catani, A. Cianchi, D. DiGiovenale, J. Lorkiewicz, S. Tazzari, C. Granata, P. Ventrella, G. Lamura, and A. Andreone, [IEEE Transactions on](http://dx.doi.org/10.1109/TASC.2009.2019205) [Applied Superconductivity](http://dx.doi.org/10.1109/TASC.2009.2019205) 19, 1394 (2009).