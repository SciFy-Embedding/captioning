# arXiv:1606.00430

**Paper ID:** 8d1ff18e329a0c02f3d75b20778591ed

**PDF Path:** apl/Superconductors/arXiv:1606.00430.pdf

**Processing Status:** complete

**Captions Added:** 0

**Generated:** 2025-06-24T13:44:27.112229

---

# A systematic study of the superconducting critical temperature in two and three dimensional tight-binding models: a possible scenario for superconducting H3S ?

Thiago X. R. Souza

Department of Physics, University of Alberta, Edmonton, AB, Canada T6G 2E1,

and

Departamento de Fisica, Universidade Federal de Sergipe, 49100-000 Sao Cristovao, SE, Brazil

F. Marsiglio

Department of Physics, University of Alberta, Edmonton, AB, Canada T6G 2E1 (Dated: December 23, 2021)

Ever since BCS theory was first formulated it was recognized that a large electronic density of states at the Fermi level was beneficial to enhancing Tc. The A15 compounds and the high temperature cuprate materials both have had an enormous amount of effort devoted to studying the possibility that such peaks play an important role in the high critical temperatures existing in these compounds. Here we provide a systematic study of the effect of these peaks on the superconducting transition temperature for a variety of tight-binding models of simple structures, both in two and three dimensions. In three dimensions large enhancements in T<sup>c</sup> can occur, due to van Hove singularities that result in divergences in the density of states. Furthermore, even in more realistic structures, where the van Hove singularity disappears, large enhancements in T<sup>c</sup> continue due to the presence of 'robust' peaks in the densities of states. Such a peak, recently identified in the bcc structure of H3S, is likely the result of such a van Hove singularity. In certain regimes, anomalies in the isotope coefficient are also expected.

#### I. INTRODUCTION

The weak coupling Bardeen-Cooper-Schrieffer (BCS)[1](#page-14-0) expression for the superconducting transition temperature T<sup>c</sup> is

<span id="page-0-0"></span>
$$T\_c \sim \omega\_D e^{-1/[g(\epsilon\_F)V]} \tag{1}$$

(we set ¯h = k<sup>B</sup> = 1) where ω<sup>D</sup> is the typical (Debye) phonon frequency, V is the attractive interaction strength, and g(ǫ<sup>F</sup> ) is the electron density of states at the Fermi level. This simple expression makes clear that a high value of the density of states at the Fermi energy is desirable for high Tc, and has served to motivate a directed search for high T<sup>c</sup> materials for more than half a century. Some understanding of the impact on T<sup>c</sup> has come historically from a study of the A15 compounds, where experiments suggested that various 'anomalous' superconducting properties in these compounds could be explained by peaks (or in some cases valleys) in the electron density of states near the Fermi level.

Indeed, as early as 1967 Labb´e et al.[2](#page-14-1) suggested that sharp peaks in the electronic density of states could explain the high T<sup>c</sup> and low isotope effects in some A15 compounds. They adopted a density of states with a square-root singularity, reminiscent of the result obtained in one dimension. Since that time, Nettel and Thomas[3](#page-14-2) and Horsch and Rietschel[4](#page-14-3) developed this model further in the context of Eliashberg theory, again with an eye towards explaining the high critical temperatures of some of the A15 compounds. Follow-up work by Lie and Carbotte[5](#page-14-4) , Ho et al.[6](#page-14-5) , Pickett[7](#page-14-6) and Mitrovi´c and Carbotte[8](#page-14-7) served to establish the importance of peaked structures in the electronic density of states near the Fermi level for the critical temperatures in the A15 superconductors.

In the mid 1980's the possibility of enhancing the superconducting critical temperature through a twodimensional structure was advanced by Hirsch and Scalapino,[10](#page-14-8) and these authors also used Monte Carlo simulations and high-order perturbation corrections to support their claims. They found enhanced superconductivity when the Fermi level was near a singularity, particularly in the weak coupling regime. These ideas were further developed with the discovery of high temperature superconductivity in 1986, and several papers[11](#page-14-9)[,12](#page-14-10) subsequently explored some of the consequences of a twodimensional van Hove singularity for superconductivity. Rather than recount a detailed history of the various calculations, we refer the reader to review papers, a comprehensive one in 1997,[13](#page-14-11) and a more recent review[14](#page-14-12) focussed on the A15 compounds. While the early work focussed on a square-root singularity, most of the work in the last 30 years has almost exclusively utilized a density of states with a logarithmic divergence, motivated by the two-dimensional tight-binding model. A notable exception is the extended saddle point singularity pointed out through density functional theory calculations in 1991[15](#page-14-13) and observed through ARPES measurements and modelled in 1993.[16](#page-14-14) The extended saddle point results in a one-dimensional-like square-root singularity in the electronic density of states.

The theoretical description of these various scenarios has focussed on the situation where the Fermi energy lies close to the singularity in the density of states. In this pa-

2

per we wish to do two things. First, we will extend these calculations to all electron densities, in the vicinity of the singularity, and well away from it. Our results will be numerical, and will account self-consistently for changes in the chemical potential as the electron density and coupling strength of the pairing interaction varies. These calculations will be performed for a two-dimensional tightbinding model on a square lattice, where a logarithmic singularity in the electronic density of states always exists.

Secondly, we will extend these calculations to three dimensions. Of course van Hove anomalies also exist in three dimensions. Jelitto[17](#page-14-15) showed long ago that for the body-centred cubic (bcc) and face-centred cubic (fcc) lattice structures these anomalies result in singularities in the density of states as well. As this important result appears to be under-appreciated, we review some of his results in the Appendix. Finally, we note that the density of states for the bcc lattice, with a non-negligible nextnearest-neighbor (NNN) hopping amplitude, renders a density of states with a significant and 'robust' peak, very similar to one recently calculated[18](#page-14-16)[,19](#page-15-0) with density functional theory for the newly discovered superconductor, H3S.[20](#page-15-1) We find a significant enhancement of T<sup>c</sup> for electron densities obtained for the chemical potential close to the energy of this peak. In summary, while the bulk of this paper is devoted to a comprehensive survey for T<sup>c</sup> (and in some cases the isotope coefficient and the superconducting order parameter), as a function of electron density and coupling strength, in both two and three dimensions, for a variety of "cubic" lattice structures, we find that the bcc structure itself results in a substantial enhancement of Tc.

It is probably best to specify the simplifying assumptions that we utilize: (i) We assume a momentum independent pairing interaction, and hence this study is confined to a superconducting order parameter with s-wave symmetry. (ii) We will adopt a non-retarded framework for the interaction, i.e. we will use the BCS formalism, rather than the Eliashberg formalism. Many authors (see, e.g. Ho et al.[6](#page-14-5) ) have pointed out that retardation effects will smear the effective electronic density of states, so that a BCS-like treatment will tend to over-estimate the effects of a singularity in the density of states. This is understood here, and it is desirable to have a follow-up study similar to this one based on the Eliashberg formalism.[21](#page-15-2) (iii) We will focus on a metal in which a single band crosses the Fermi level; furthermore, we will adopt a tight-binding model to describe the dispersion of this band, and correlation effects in the normal state are assumed to be absent. (iv) While we will adopt analytical approximations from time to time these will be for illustrative purposes only — all our main results will be numerically exact, with no weak coupling approximations, for example. The one exception is that at the band edges we do not concern ourselves with possible strong coupling effects. These effects will give rise to Bose condensation physics dominating over BCS pairing (i.e. condensation arises not from pairing per se, but from phase coherence); however, since the theoretical description of this crossover is not universally agreed upon,[22](#page-15-3)[,23](#page-15-4) for present purposes we simply use the BCS formalism in this very small regime as well.

The outline is as follows. In the next section we provide a concise formulation of the equations we solve, both at the critical temperature Tc, and at temperatures below Tc. In the following section we focus on the two dimensional square lattice, first with nearest neighbour hopping only, and then with next-nearest neighbour hopping. We examine T<sup>c</sup> = Tc(n, V, ωD), where n is the electron density, V is the coupling strength, and ω<sup>D</sup> is used as a cutoff, representing the Debye frequency of the phonons. We also examine the isotope coefficient (to be defined below) and the superconducting order parameter, ∆. For the most part ∆(n, V, ωD) tracks Tc(n, V, ωD), and the temperature dependence of ∆ is essentially indistinguishable from that achieved with a constant density of states. Results for a constant density of states have previously been presented[24](#page-15-5) within the Eliashberg[25](#page-15-6)[,26](#page-15-7) formalism. These results, recalculated with the much simpler BCS formalism, will provide a baseline for comparisons.

The fourth section will focus on the three dimensional cubic lattices, simple cubic (sc), body-centred cubic (bcc) and face-centred cubic (fcc). The first two have particlehole symmetry, while the third does not, and the singularity in the electron density of states for the fcc lattice lies at the upper end of the spectrum. We also consider the impact of next nearest neighbour hopping in all three cases. Somewhat surprisingly, in the bcc and fcc cases, while the singularity is removed, a robust peak remains, and considerable enhancement of T<sup>c</sup> occurs. Equally surprisingly, in the sc case, turning on the next nearest neighbour hopping moves the density of states towards one with a singularity.

Finally, we point out that for a particular range of NNN hopping amplitude, the density of states resembles that calculated[27](#page-15-8)[,28](#page-15-9) with density functional theory, and leads to a significant enhancement of Tc.

#### II. THE PAIRING FORMALISM

The BCS equations are written as[29](#page-15-10)[,30](#page-15-11)

<span id="page-1-0"></span>
$$
\Delta\_k = -\frac{1}{N} \sum\_{k'} V\_{kk'} \frac{\Delta\_{k'}}{2E\_{k'}} \left[ 1 - 2f(E\_{k'}) \right], \qquad (2)
$$

and

<span id="page-1-1"></span>
$$n = \frac{1}{N} \sum\_{k'} \left[ 1 - \frac{\epsilon\_{k'} - \mu}{E\_{k'}} \left( 1 - 2f(E\_{k'}) \right) \right],\tag{3}$$

with

$$E\_k \equiv \sqrt{(\epsilon\_{k'} - \mu)^2 + \Delta\_{k'}^2}.\tag{4}$$

Here, ∆<sup>k</sup> is the superconducting order parameter; this parameter goes to zero at the critical temperature Tc. N is the number of unit cells in the lattice and the summation over k-points is to cover the entire First Brillouin zone (FBZ). In principle this summation also covers all bands in the FBZ, but as specified in our assumptions we focus on one band only, in which the Fermi energy lies. The pairing interaction, to be specified further below, is given by Vkk′ . Note that the dependence on the centreof-mass momentum q is absent, so that this is the interaction for the so-called "reduced BCS" Hamiltonian. We have also adopted the convention that an attractive interaction will be negative, so that Eq. [\(2\)](#page-1-0) has a minus sign. The chemical potential is denoted by µ; it will generally be altered by the presence of the superconducting state, although in practice, in weak and intermediate coupling situations it will change only by a very small amount. By using Eq. [\(3\)](#page-1-1) we take these changes into account in order to preserve the electron density, n, as the various parameters, such as temperature, or even the "fixed" parameters like ωD, are varied. Finally all the temperature dependence is included through the Fermi-Dirac distribution function, f(x) ≡ 1/[exp(βx)+1], where β ≡ 1/[kBT ] is the inverse temperature, with k<sup>B</sup> the Boltzmann constant.

In addition we need to specify an energy dispersion, ǫk. We adopt the tight binding model, so for example, with nearest-neighbour (NN) hopping only, we obtain

<span id="page-2-2"></span>
$$\epsilon\_k = -2t \left[ \cos(k\_x a) + \cos(k\_y a) \right] \tag{2D} \quad (5)$$

$$\epsilon\_k = -2t\_s \left[ \cos(k\_x a) + \cos(k\_y a) + \cos(k\_z a) \right] \quad \text{[sc]} \quad (6)$$

$$\epsilon\_k = -8t\_b \left[ \cos(\frac{k\_x a}{2}) \cos(\frac{k\_y a}{2}) \cos(\frac{k\_z a}{2}) \right] \qquad \text{[bcc]} \tag{7}$$

$$\epsilon\_k = -4t\_f \left[ \cos(\frac{k\_x a}{2}) \cos(\frac{k\_y a}{2}) + \cos(\frac{k\_x a}{2}) \cos(\frac{k\_z a}{2}) \right]$$

$$+ \cos(\frac{k\_y a}{2}) \cos(\frac{k\_z a}{2}) \qquad\qquad\qquad\qquad\qquad\qquad\text{[fcc]}\tag{8}$$

for the four structures considered, where a is the nearest neighbour distance in the 2D and (sc) cases, and is the length of the cube in the bcc and fcc cases, containing 8 atoms at each vertex along with one in the centre (bcc) and six on the face centres (fcc). Also, t, ts, tb, and t<sup>f</sup> are the nearest neighbour hopping parameters for the 2D square, 3D simple cubic, 3D bcc, and 3D fcc lattices, respectively. Note that these have bandwidths W of 8t, 12ts, 16tb, and 16t<sup>f</sup> , respectively. In the main text and figures that follow, we will generally use 't' to designate the NN hopping, and 't2' to designate the nextnearest neighbour (NNN) hopping parameter (see Appendix). Thus, unless necessary to distinguish the various cases, we will drop the additional subscript, s, b, and f, and retain them only as needed. These dispersions are further discussed in the Appendix.

At this point the main simplifying assumption in the ensuing calculations is that the pairing interaction is essentially local, so that the pairing interaction is independent of momentum. We wish to retain the notion that pairing is via boson exchange, and with the phonon mechanism in mind following BCS,[1](#page-14-0) we want to include a feature that requires the two electrons to have single particle energies that are no further than ¯hω<sup>D</sup> part from one another. This is difficult to implement in practice, so instead we adopt the standard model that restricts each of the single particle energies to be within ¯hω<sup>D</sup> of the chemical potential, µ. That is,

$$V\_{kk'} = -V\theta \left[\hbar\omega\_D - |\epsilon\_k - \mu|\right] \theta \left[\hbar\omega\_D - |\epsilon\_{k'} - \mu|\right] \quad (9)$$

where θ[x] ≡ 0 for x < 0 and θ[x] ≡ 1 for x > 0 is the Heaviside step function, and V > 0 implies that this is an attractive interaction potential. With this model in place, the order parameter becomes non-zero only for |ǫ<sup>k</sup> − µ| < ¯hωD, and its value is independent of momentum.[31](#page-15-12)

Because of the simplicity of this model potential, one can rewrite the momentum sums in Eqs. [\(2](#page-1-0)[,3\)](#page-1-1) in terms of the electronic density of states, g(ǫ) (see the Appendix). Then these equations become

<span id="page-2-0"></span>
$$\frac{1}{V} = \int\_{\mu\_{-}}^{\mu\_{+}} d\epsilon g(\epsilon) \frac{\tanh[\beta E(\epsilon)/2]}{2E(\epsilon)} \tag{10}$$

and

<span id="page-2-1"></span>
$$n = \int\_{\epsilon\_{\rm min}}^{\epsilon\_{\rm max}} d\epsilon g(\epsilon) \left[ 1 - \frac{(\epsilon - \mu)}{E(\epsilon)} \tanh[\beta E(\epsilon)/2] \right], \quad \text{(11)}$$

with E(ǫ) = p (ǫ − µ) <sup>2</sup> + ∆<sup>2</sup>. Here, the integration limits in Eq. [\(10\)](#page-2-0) are normally µ<sup>−</sup> = µ − ¯hω<sup>D</sup> and µ<sup>+</sup> = µ + ¯hωD, while those in Eq. [\(11\)](#page-2-1) are ǫmin, the band energy at the bottom of the band, and ǫmax, the band energy at the top of the band. An exception occurs when the chemical potential is close to one of the band edges. In this case, the integration is cut off by the band edge, so more accurately, µ<sup>−</sup> ≡ max[µ − ¯hωD, ǫmin], and µ<sup>+</sup> ≡ min[µ + ¯hωD, ǫmax].

Equations [\(10,](#page-2-0)[11\)](#page-2-1) represent two non-linear equations for the unknowns ∆ and µ, given the parameters V and n. At zero temperature the hyperbolic tangent function is replaced by unity; at T<sup>c</sup> the order parameter goes to zero so the problem is slightly different. One then has to find the temperature and the chemical potential at which both these equations are satisfied. These equations are

$$\frac{1}{V} = \int\_{\mu\_{-}}^{\mu\_{+}} d\epsilon g(\epsilon) \frac{\tanh[\beta\_{c}(\epsilon - \mu)/2]}{2(\epsilon - \mu)} \qquad \qquad [T = T\_{c}] \tag{12}$$

and

$$n = 2\int\_{\epsilon\_{\min}}^{\epsilon\_{\max}} d\epsilon g(\epsilon)f(\epsilon - \mu), \qquad \qquad [T = T\_c] \tag{13}$$

where β<sup>c</sup> ≡ 1/[kBTc]. Numerical results in subsequent sections are the result of an iterative solution to these equations.

### III. TWO DIMENSIONS

As detailed in the Appendix, the electron density of states for a two-dimensional tight-binding model with nearest neighbour hopping only is

<span id="page-3-1"></span>
$$g\_{\rm 2D}(\epsilon) = \frac{1}{2\pi^2 t a^2} K \left[ 1 - \left(\frac{\epsilon}{4t}\right)^2 \right],\tag{14}$$

where K(m) ≡ R π/<sup>2</sup> 0 dθ <sup>1</sup> <sup>−</sup> <sup>m</sup> sin<sup>2</sup> θ −1/<sup>2</sup> is the complete elliptic integral of the first kind.[32](#page-15-13) This density of states is well approximated by the expression

<span id="page-3-0"></span>
$$g\_{\rm 2D}(\epsilon) \approx \frac{1}{2\pi^2 t a^2} \log(\frac{16t}{|\epsilon|});\tag{15}$$

Eq. [\(15\)](#page-3-0) is the asymptotic form of Eq. [\(14\)](#page-3-1) as ǫ → 0, and is often used in lieu of Eq. [\(14\)](#page-3-1). Both are shown in Fig. [13](#page-12-0) in the Appendix, along with a numerical evaluation of the density of states when the next-nearest-neighbour hopping is included as well. As discussed in the Appendix, a simple numerical routine can efficiently and accurately evaluate complete elliptic integrals, so we proceed with the full form, Eq. [\(14\)](#page-3-1).

#### A. Numerical results

We show in Fig. [\(1a](#page-3-2)) Tc/ω<sup>D</sup> (we set ¯h = k<sup>B</sup> = 1) vs. electron density n, for a relatively weak coupling situation, V /t = 2, for three different values of ωD/t = 1.0, 0.1, and 0.01. The latter two values are more realistic as in general, ω<sup>D</sup> << t, i.e. phonon energy scales are much smaller than electronic energy scales. The curves provide results for the complete self-consistent solution without approximation, i.e. using the density of states from Eq. [\(14\)](#page-3-1). There is a clear enhancement of Tc, especially near the van Hove singularity. and the enhancement is most pronounced for smaller values of ωD. Also shown is the result for a constant density of states; for electron densities near half-filling this shows that T<sup>c</sup> can be enhanced by more than an order of magnitude. In Fig. [\(1b](#page-3-2)) we show the same quantity as a function of ωD/t, now for a variety of values of V /t, all for halffilling (shown with curves). T<sup>c</sup> will tend to increase as a function of ωD, but we have plotted the ratio, Tc/ω<sup>D</sup> vs. ωD/t, which shows an enhancement of Tc/ω<sup>D</sup> as ω<sup>D</sup> → 0.

#### B. Analytical results

Analytical results are possible through a series of simplifications, as follows. First, we focus on half-filling, n = 1. This means that the chemical potential remains fixed at µ = 0, independent of temperature. Second, we adopt the approximation given in Eq. [\(15\)](#page-3-0), which, based on the comparisons of the density of states given in the Appendix, we anticipate will be very accurate. Indeed,

![](_page_3_Figure_11.jpeg)

<span id="page-3-2"></span>FIG. 1. (a) Plot of Tc/ω<sup>D</sup> vs. n for 0 < n < 1 (the results for 2 > n > 1 are symmetric) for V /t = 2, and ωD/t = 1.0, 0.1, 0.01. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 8t is the electron bandwidth. This result is not sensitive to ω<sup>D</sup> except at the band edges. In the inset we show an expanded view of the region near zero density (also the case near n = 2), showing how T<sup>c</sup> ∝ √ n due to the lower band edge taking the place of µ− ω<sup>D</sup> for the lower cutoff. There is a clear enhancement near the van Hove singularity, especially for small ωD/t. (b) Tc/ω<sup>D</sup> vs. ωD/t for n = 1 for several values of the coupling strength V , along with several approximations discussed in the text.

this is the case, as indicated by the results depicted with square symbols in Fig. [\(1b](#page-3-2)). In particular, these results are always very accurate as ω<sup>D</sup> → 0, as this is where the density of states at the singularity is most important; this is also where the approximation Eq. [\(15\)](#page-3-0) is most accurate.

A so-called "strong-coupling" approximation to the

BCS equation is obtained as follows. We assume ωD/T<sup>c</sup> << 1, which means that the hyperbolic tangent function can be linearized. The remaining integral is then elementary, so that

$$\frac{T\_c}{\omega\_D} \approx \frac{V}{4\pi^2 t} \left[ \log(\frac{16t}{\omega\_D}) + 1 \right] \qquad \text{[strong}-\text{coupling]}. \tag{16}$$

These results are indicated with x's, and only for V = 10t in Fig. [\(1b](#page-3-2)), where it is seen to be very accurate. Here we caution the reader that it is an accurate approximation to the fully self-consistent solution as indicated, but in fact BCS theory itself is not expected to be very accurate in this regime at finite temperature. So here it merely serves as a check that our solutions to the equations are accurate.

The opposite case, that of weak coupling, is the one most normally used; furthermore we expect BCS theory to be reasonably accurate, at least in three dimensions. In two dimensions, these results are also generally not so accurate, because Kosterlitz-Thouless physics[37](#page-15-14) is expected to come into play. Our approximation follows the standard one,[30](#page-15-11) but accommodates the density of states with the logarithm singularity [Eq. [\(15\)](#page-3-0)]; we obtain

<span id="page-4-0"></span>
$$\frac{T\_c}{\omega\_D} \approx 1.134 \exp\left\{ A - \sqrt{A^2 + \frac{4\pi^2 t}{V} - B - \log[2(1.134)]} \right\},\tag{17}$$

(weak coupling) where

$$A \equiv A(\omega\_D/t) \equiv 1 + \log\left(\frac{8t}{1.134\omega\_D}\right) \tag{18}$$

and

$$B \equiv \int\_0^\infty dx \, \text{sech}^2 x \, \log^2 x \quad \approx 1.989... \qquad (19)$$

These results are indicated with asterisks for V = 2t and V = 1t in Fig. [\(1b](#page-3-2)), where they are indeed very accurate. Although not shown, they become less accurate as V increases. Notice, however, that even for Tc/ω<sup>D</sup> ≈ 0.4 (V = 2t and ω<sup>D</sup> → 0) the weak coupling approximation works very well. Eq. [\(17\)](#page-4-0) is useful to illustrate how the logarithmic singularity in the density of states changes the usual exponential suppression for superconducting T<sup>c</sup> to one that can be significantly enhanced, as now the square root of the inverse coupling strength appears in the exponential, a fact first pointed out, to our knowledge, in Ref. [\[10\]](#page-14-8). This is most readily seen by allowing V /t → 0 in Eq. [\(17\)](#page-4-0), to get

$$\frac{T\_c}{\omega\_D} \approx 1.134 \exp\left\{-\sqrt{\frac{4\pi^2 t}{V}}\right\}.\tag{20}$$

#### C. Beyond Nearest Neighbour Hopping

Going beyond nearest neighbour hopping in two dimensions destroys the particle hole symmetry, but the

![](_page_4_Figure_14.jpeg)

<span id="page-4-1"></span>FIG. 2. Plot of Tc/ω<sup>D</sup> vs. n for V /t = 2, and ωD/t = 1.0, 0.1, 0.01, for the 2D case where next-nearest-neighbour (NNN) hopping is also present. The van Hove singularity is now located at ǫ = 4t<sup>2</sup> which corresponds to a filling n ≈ 1.2. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 8t is the electron bandwidth. The same enhancement occurs as when NNN hopping is not included, when the chemical potential approaches the energy of the van Hove singularity. As was the case in Fig. 1 the enhancement of Tc/ω<sup>D</sup> is amplified as ω<sup>D</sup> decreases.

singularity in the density of states remains, albeit at some different value for the chemical potential (i.e. filling). The Appendix displays the Density of States for various values of the next-nearest-neighbour (NNN) hopping parameter. Fig. [2](#page-4-1) shows Tc/ω<sup>D</sup> vs filling and again illustrates that a significant enhancement occurs when the chemical potential is close to the van Hove singularity.

#### D. Isotope Effect

The partial isotope coefficient is defined by[25](#page-15-6)

$$\beta\_i \equiv -\frac{d\ln T\_c}{d\ln M\_i} \tag{21}$$

where M<sup>i</sup> is the mass of the i th element; the total isotope coefficient (β) is the sum of these, and for the purpose of this work we will assume an elemental superconductor; furthermore, for the harmonic approximation ω<sup>D</sup> ∝ 1/ √ M, Eq. [\(1\)](#page-0-0) implies the expected standard result, β = 1/2. This positive value indicates that increasing the Debye frequency is expected to raise Tc.

The presence of a non-constant density of states will quantitatively change this result; in particular, if the chemical potential is at a van Hove singularity, then decreasing the mass (i.e. increasing the Debye frequency)

![](_page_5_Figure_0.jpeg)

<span id="page-5-0"></span>FIG. 3. Isotope coefficient β vs ωD/t for (a) NN hopping only and n = 1, (b) NNN hopping with t<sup>2</sup> = 0.25t and n = 1.219, and (c) NNN hopping, again with t<sup>2</sup> = 0.25t, but with n = 1.5. In (a) and (b) the filling is such that the chemical potential is at the van Hove singularity. In both these cases the behaviour is qualitatively similar and the isotope coefficient decreases as a function of ωD/t (Note that the physically relevant regime is ωD/t << 1). Moreover, the most significant decrease occurs in both instances for weaker coupling, which is anyways where BCS theory is to be most trusted. In (c) the chemical potential is away from the van Hove singularity. This results in a peak in the isotope coefficient, at a value of ω<sup>D</sup> which tracks the energy difference between the chemical potential and the van Hove singularity. This is as expected, as the highest impact on T<sup>c</sup> will occur when the the value of ω<sup>D</sup> can include the states at the van Hove singularity. At large (and unrealistic) values of ω<sup>D</sup> the isotope coefficient decreases significantly as two (a) or one (b and c) band edges replaces ω<sup>D</sup> as the cutoff.

will increase T<sup>c</sup> less than what one would expect normally. This is because more states are included in the energy-lowering due to condensation, as before, but the energy regime where this occurs (about ω<sup>D</sup> on either side of the chemical potential) has a lower electronic density of states, so the incremental benefit is decreased from what it is if the density of states is constant.

Fig. [3](#page-5-0) shows the isotope coefficient vs. ωD/t for (a) nearest-neighbour (NN) hopping only, at half-filling, (b) NNN with the chemical potential at the van Hove singularity (n = 1.219), and (c) NNN with a filling of n = 1.5. The results in Fig. [3\(](#page-5-0)a) and (b) are qualitatively similar; the coefficient rapidly decreases as ω<sup>D</sup> increases, as would be expected, since the relevant energy regime moves further away from the singular part of the density of states with increasing ωD. In both cases a precipitous drop occurs when ω<sup>D</sup> exceeds a value that corresponds to the distance from the chemical potential to the band edge. While these values are unrealistically large, it is worth understanding what is occurring. In the case of NN hopping this value is 4t, and then the isotope coefficient becomes zero, since further increasing ω<sup>D</sup> plays no role in determining Tc, as the role of the cutoff is now taken by the band edge, and not ωD. For NNN hopping only one band edge takes on the role of the cutoff; the second bandwidth will enter for larger values of ωD/t than those shown.

In Fig. [3\(](#page-5-0)c) the chemical potential is well away from the van Hove singularity; then the isotope coefficient β peaks in value when the value of ω<sup>D</sup> is 'tuned' to equal the difference in energy between the chemical potential and the energy of the van Hove singularity. Coupling to these states has the most significant effect on Tc, which results in a peak and in the achievement of anomalously high values of β.

#### E. The zero temperature energy gap

One can ask if the just-described effects of a van Hove singularity similarly apply to the finite temperature energy gap. We have thus solved the finite temperature gap equations, Eqs. [\(10,](#page-2-0)[11\)](#page-2-1) in several representative cases. For the most part we find that little differs for the pairing gap, ∆, as a function of vicinity of the Fermi energy to the van Hove singularity. For example, the temperature dependence, ∆(T ), as a function of temperature T is barely discernible from the usual temperature dependence obtained with a constant density of states. By way of example, we show in Fig. [4](#page-6-0) the pairing gap at zero temperature, ∆, and the superconducting critical temperature, Tc, as a function of electron density, n, for the 2D case with NN hopping only. Both peak near n = 1, i.e. the location of the van Hove singularity in the density of states; in this sense, ∆ tracks Tc. The ratio for example, 2∆/(kBTc), grows from 3.53 at low densities to about 3.7 at half-filling, a rather insignificant change. We turn to three dimensions now, and focus on Tc.

![](_page_6_Figure_1.jpeg)

<span id="page-6-0"></span>FIG. 4. T<sup>c</sup> and the zero temperature order parameter, ∆, vs. electron density n, for the 2D case with NN hopping only, with V = 2t and ω<sup>D</sup> = 0.1t. The behaviour of ∆ follows that of Tc, with a slight enhancement of the ratio as the singularity is approached.

#### IV. THREE DIMENSIONS

We now turn to similar calculations in 3D. As summarized in Eqs. [\(6-8\)](#page-2-2), we focus only on the cubic lattice structures, sc, fcc, and bcc. The densities of states for these were first calculated by Jelitto[17](#page-14-15) and are provided in the Appendix for cases involving NNN hopping as well. As we mentioned earlier, while these calculations were performed almost half a century ago, most researchers are not aware[33](#page-15-15) that singularities indeed exist in the tight-binding model for the bcc and fcc cases (for NN hopping only) and even in the sc case when NNN hopping is included. This is true only for 'special' values of the hopping parameters, but as detailed in the appendix, remnants of these singularities remain even when other values of the hopping parameters are used. Based on what we found in two dimensions, along with some exploratory calculations, here we will focus on T<sup>c</sup> and the isotope coefficient, β; results for ∆ follow those of Tc, as we found in two dimensions.

## A. simple cubic NN

The simple cubic density of states consists of van Hove singularities only in the derivative of the density of states with respect to energy (see the red curve in Fig. [14](#page-13-0) in the Appendix). The behaviour of T<sup>c</sup> is therefore not so unusual. Fig. [5\(](#page-6-1)a) shows Tc/ω<sup>D</sup> vs. n for a fairly weak coupling case (V = 3t) from zero density to halffilling (n = 1). Note that this lattice is bipartite and has

![](_page_6_Figure_7.jpeg)

<span id="page-6-1"></span>FIG. 5. (a) T<sup>c</sup> vs. filling, n and (b) the isotope coefficient β vs ωD/t for a variety of values of ω<sup>D</sup> in (a) and for a number of coupling strengths in (b). These results are for the simple cubic three-dimensional case, with NN hopping only, and for n = 1. Results are as expected and as explained in the text.

particle-hole symmetry. Hence, results for n > 1 are a mirror reflection of those for n < 1, and we display only the latter. We show results for three values of ωD; in fact as long as ω<sup>D</sup> << t the electron density of states at the chemical potential plays the most important role, as is evident from how T<sup>c</sup> tracks g(ǫ<sup>F</sup> ), albeit as a function of occupation rather than as a function of energy. Only for ω<sup>D</sup> = t does the T<sup>c</sup> curve begin to become "rounded" compared to the density of states. Also shown is the result obtained for a constant density of states, 1/W, where W = 12t for the three dimensional simple cubic tight-binding model. In Fig. [5\(](#page-6-1)b) we show the isotope coefficient as a function of ωD/t for four different values

![](_page_7_Figure_0.jpeg)

<span id="page-7-0"></span>FIG. 6. Plot of Tc/ω<sup>D</sup> vs. n for V /t = 4, and ωD/t = 0.1, 0.01, for three different values of t2/t = 0, 0.125, 0.250. Note that the results are relatively insensitive to ω<sup>D</sup> except for t<sup>2</sup> = 0.25t, where a singularity exists in the density of states near the top of the band (see Fig. [14](#page-13-0) in the Appendix), and Tc/ω<sup>D</sup> continues to increase near n = 2 as ω<sup>D</sup> decreases. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 12t is the electron bandwidth for the sc lattice with |t2|/t ≤ 1/4. This latter result is not sensitive to ω<sup>D</sup> except at the band edges, and is shown only for ωD/t = 0.01.

of the coupling strength, V /t. The results for V = 3t and V = 1.5t cannot be distinguished from one another, indicating that V = 3t is already in the weak coupling limit. The isotope coefficient becomes reduced from the 'canonical' value of 0.5 only when ω<sup>D</sup> increases beyond the energy of the first van Hove singularity near the origin, at ±2t. The β decreases steadily to zero, achieved for ω<sup>D</sup> ≥ W/2 = 6t. The dependency on coupling strength is very minor. Away from half-filling there are no surprises, and both T<sup>c</sup> and β track the density of states at the chemical potential. As was the case in two dimensions, the isotope coefficient displays a peak when the size of ω<sup>D</sup> allows a coupling to states with significantly higher density of states, i.e. when |µ| > 2t, where µ is the chemical potential.

## B. simple cubic NNN

Remarkably, including sufficient NNN hopping in the sc lattice results in a singularity in the density of states at the top of the band (see the blue curve in Fig. [14](#page-13-0) in the Appendix), similar to what occurs for the FCC lattice with NN hopping only (see below). In Fig. [6](#page-7-0) we show Tc/ω<sup>D</sup> vs. electron density for three different values of the NNN hopping, t2/t = 0, 0.125, 0.250 and two different values of ωD. The value of ω<sup>D</sup> is not so important except in the case t2/t = 0.25, where a singularity occurs in the density of states; this results in a singularity in Tc/ω<sup>D</sup> near n = 2 as ω<sup>D</sup> decreases. Also shown is the result for a constant density of states with value g(ǫ) = 1/W, where W = 12t is the electron bandwidth for the sc lattice with |t2|/t ≤ 1/4. Clearly the potential enhancement of Tc/ω<sup>D</sup> is very large at high fillings. A particle-hole symmetry exists with these results for negative values of t2/t (not shown). The important point is that for values of t2/t close to 0.25 a peak will remain in the density of states, giving rise to a large enhancement in Tc/ωD.

#### C. body-centred cubic NN

As noted in the Appendix the tight-binding model with a bcc lattice with NN hopping only displays a singularity at ǫ = 0. This singularity is a logarithm squared and hence stronger than the two-dimensional singularity which diverges logarithmically. Fig. [7\(](#page-8-0)a) shows Tc/ω<sup>D</sup> vs. n for a fairly weak coupling case (V = 4t) for zero density to half-filling (n = 1). Like the sc 3D case, the bcc lattice is bipartite, and with NN hopping only, this lattice has particle-hole symmetry. Hence, as in that case, results for n > 1 are a mirror reflection of those for n < 1, and we display only the latter. We show results for three values of ωD; again, as long as ω<sup>D</sup> << t, the electron density of states at the chemical potential plays the most important role. In particular, for the smallest value of ω<sup>D</sup> shown, T<sup>c</sup> again tracks g(ǫ<sup>F</sup> ) as a function of occupation (rather than as a function of energy). The enhancement above the result for a constant density of states with the same bandwidth (horizontal line just above zero) is enormous. Here, W = 16t for the bcc NN tight-binding model. In Fig. [7\(](#page-8-0)b) we show the same quantity, Tc/ω<sup>D</sup> vs. ω<sup>D</sup> for a variety of coupling strengths at half-filling. As expected, this BCS calculation shows T<sup>c</sup> increasing with V ; we remind the reader again that this calculation is expected to be valid only for some weak coupling range. The important point is that Tc/ω<sup>D</sup> eventually diverges as ω<sup>D</sup> decreases, because the density of states at µ = 0 is diverging, and the density of states right at the Fermi level becomes the only key quantity as the Debye frequency decreases.

Away from half-filling results are again as expected; the chemical potential is at an energy where the density of states is relatively low. As ω<sup>D</sup> increases to a point where the singularity in the density of states becomes relevant then Tc/ω<sup>D</sup> will peak. Unlike Fig. [7\(](#page-8-0)b), where Tc/ω<sup>D</sup> monotonically decreases as ω<sup>D</sup> increases, Tc/ω<sup>D</sup> is non-monotonic, i.e. the result is sensitive to the singularity not at the Fermi level.

The isotope coefficient is similar to what we have seen before; in Fig. [8\(](#page-8-1)a) we show β as a function of ωD/t for n = 1 and for a variety of coupling strengths. Clearly
![](0__page_8_Figure_1.jpeg)

![](0__page_8_Figure_2.jpeg)

<span id="page-8-1"></span>FIG. 7. (a) Plot of Tc/ω<sup>D</sup> vs. n for 0 < n < 1 (the results for 2 > n > 1 are symmetric) for V /t = 4, and ωD/t = 1.0, 0.1, 0.01. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 16t is the electron bandwidth. This latter result is not sensitive to ω<sup>D</sup> except at the band edges. There is a significant enhancement near the van Hove singularity, which continues to grow without bound for decreasing ωD/t. (b) Tc/ω<sup>D</sup> vs. ωD/t for n = 1 for various values of V . This view highlights the sharp increase in Tc/ω<sup>D</sup> as ω<sup>D</sup> → 0.

the isotope coefficient is greatly reduced even for fairly low values of ωD/t. In Fig. [8\(](#page-8-0)b) we show the same result for n = 0.5; now the isotope coefficient peaks to very high values, as values of ω<sup>D</sup> are reached that bring the singularity in the electron density of states in "resonance" with the chemical potential through ωD. That is, a prominent peak in β occurs, particularly in weak cou-

<span id="page-8-0"></span>FIG. 8. (a) Isotope coefficient β vs. ωD/t for the 3D bcc NN case (t<sup>2</sup> = 0), at half-filling, for a variety of coupling strengths. The isotope coefficient is significantly reduced from 0.5, due to the singularity in the density of states. (b) Isotope coefficient β vs. ωD/t for the 3D bcc NN case (t<sup>2</sup> = 0), at quarter filling n = 0.5, for the same coupling strengths as in (a). The isotope coefficient now has a significant peak, particularly for weak coupling, when ω<sup>D</sup> is such that states in the peak of the density of states are primarily included in determining T<sup>c</sup> (see text).

pling, when µ + ω<sup>D</sup> = ǫsing, where ǫsing = 0 is the energy at which the singularity occurs. We thus have the intriguing possibility of an anomalously high isotope coefficient associated with a non-optimal critical temperature.

The bcc tight-binding model shows significant enhancement for Tc, for a wide range of electron density (Fig. [7\(](#page-8-1)a)). However, as was the case in 2D, it is important to examine the impact of a NNN hopping parameter.

![](0__page_9_Figure_1.jpeg)

![](0__page_9_Figure_2.jpeg)

<span id="page-9-0"></span>FIG. 9. Plot of Tc/ω<sup>D</sup> vs. electron density n for V /t = 4, for the BCC lattice structure, now with NNN hopping, t<sup>2</sup> = 0.3t, and ωD/t = 1.0, 0.1, 0.01. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 16t is the electron bandwidth. As before, this latter result is not sensitive to ω<sup>D</sup> except at the band edges. There is a clear enhancement of T<sup>c</sup> over a wide range of densities near the maximum in the Density of States (see Fig. [15\)](#page-13-0). Note, however, that because the density of states is no longer singular, Tc/ω<sup>D</sup> now saturates as ω<sup>D</sup> decreases (red to blue curve).

We do this in the next subsection.

# D. body-centred cubic NNN

The introduction of NNN hopping for the BCC lattice structure changes the nature of the density of states in a profound way. As illustrated in the Appendix, the singular behaviour is entirely removed. Nonetheless, a highly peaked structure remains, which we expect will continue to cause a considerable enhancement of Tc. This enhancement is more significant than the one found at half-filling since the Fermi surface is no longer nested, and therefore competing instabilities to superconductivity will be suppressed. Since the density of states is not singular, however, we expect that T<sup>c</sup> will not continue to increase as ω<sup>D</sup> is decreased (as was the case in Fig. [7\(](#page-8-1)a) at n = 1). Nonetheless, as noted in the Appendix, the maximum is, in many respects, more 'robust' than in the NN case, in that a larger area is contained in the maximum region than in the NN case. This was also the case in 2D, and is the case for the FCC lattice structure (see Fig. [16](#page-14-0) below).

Figure [9](#page-9-0) shows Tc/ω<sup>D</sup> vs. n, for a variety of values of

<span id="page-9-1"></span>FIG. 10. Plot of Tc/ω<sup>D</sup> vs. electron density n for V /t = 4, for the FCC lattice structure, with NN hopping only, and ωD/t = 1.0, 0.1, 0.01. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 16t is the electron bandwidth. As before, this result is not sensitive to ω<sup>D</sup> except at the band edges. There is a clear enhancement of T<sup>c</sup> over a wide range of densities near the maximum in the Density of States (see Fig. [16\)](#page-14-0). Note that near n = 2, this maximum will increase without bound as ω<sup>D</sup> decreases.

ωD, for the particular case of t<sup>2</sup> = 0.3t. As anticipated, the enhancement with decreasing ω<sup>D</sup> now saturates; near the peak values, Tc/ω<sup>D</sup> hardly increases as ω<sup>D</sup> = 0.1t (red curve) decreases to ω<sup>D</sup> = 0.01t (blue curve). This is in contrast to the scenario shown in Fig. [7,](#page-8-1) where, near the peak electron density, Tc/ω<sup>D</sup> continues to increase indefinitely as ω<sup>D</sup> decreases. Here, however, the Fermi surface is no longer nested, and competing instabilities, not considered here, will be significantly suppressed (see, for example, Fig. 1 in Ref. [\[36\]](#page-15-0) for a demonstration of the suppression of a charge-density-wave instability due to the removal of nesting). Thus, the large enhancement displayed in Fig. [9](#page-9-0) will likely remain even when competing instabilities are considered.

#### E. face-centred cubic NN and NNN

Similar to the BCC NN case, the electron density of states for an FCC lattice with NN hopping only, also displays a singularity, right at the band edge, as shown by the red curve, i.e. the rightmost curve, in Fig. [16](#page-14-0) in the Appendix. In this case the FCC lattice is not bipartite, so no nesting occurs. In Fig. [10](#page-9-1) we show results

![](0__page_10_Figure_1.jpeg)

<span id="page-10-0"></span>FIG. 11. Plot of Tc/ω<sup>D</sup> vs. electron density n for V /t = 4, for the FCC lattice, now with NNN hopping, t<sup>2</sup> = 0.3t, and ωD/t = 1.0, 0.1, 0.01. Also shown is the result for a constant density of states, g(ǫ) = 1/W, where W = 17.2t is the electron bandwidth for t<sup>2</sup> = 0.3t. Even for the non-constant density of states results, note the lack of sensitivity of Tc/ω<sup>D</sup> to ω<sup>D</sup> over essentially all electron densities, for sufficiently small values of ωD/t. There remains a clear enhancement of Tc/ω<sup>D</sup> over a wide range of densities near the maximum in the Density of States (see Fig. [16\)](#page-14-0).

for relatively weak coupling, V = 4t, as a function of electron density, for several values of ωD. The expected enhancement in Tc/ω<sup>D</sup> occurs near the singularity, now at the top of the band, and this increases indefinitely as ω<sup>D</sup> decreases. Note that for decreasing ωD, enhancement of Tc/ω<sup>D</sup> continues only for a limited electron density region near the singularity.

This is expected to be a robust result, in that there will not be a large enhancement of competing instabilities due to the lack of nesting. On the other hand, one always expects a small amount of NNN hopping, and as this removes the singularity in the electron density of states, one might well ask whether the ensuing enhancement of Tc/ω<sup>D</sup> will also disappear. Fig. [16,](#page-14-0) which also displays the electronic density of states for non-zero values of t2/t, illustrates that a "robust" peak remains. We focus now on results for t2/t = 0.3. Fig. [11](#page-10-0) shows Tc/ω<sup>D</sup> vs electron density for relatively weak coupling, and shows that a strong enhancement of Tc/ω<sup>D</sup> continues to occur near the peak structure in the density of states. In fact the values of Tc/ω<sup>D</sup> are comparable to (or even greater than!) those achieved (for electron densities 1.0 < n < 1.9) even with ω<sup>D</sup> = 0.01t in the case with t<sup>2</sup> = 0 (see Fig. [10\)](#page-9-1),

![](0__page_10_Figure_5.jpeg)

<span id="page-10-1"></span>FIG. 12. Plot of Tc/ω<sup>D</sup> vs. electron density n for various values of coupling strength, V /t = 2, 3, and 4. We used the BCC lattice with NNN hopping, t<sup>2</sup> = −0.3t, which provides a reasonable facsimile to the peak in the density of states (see inset) calculated with DFT methods. For illustration purposes, if V = 2t (V /(16t) = 0.125), with nearest-neighbour hopping t = 1 eV, and ω<sup>D</sup> = 100 meV, then T<sup>c</sup> ≈ 160 K (at n = 1). In this range of ω<sup>D</sup> the results for Tc/ω<sup>D</sup> are insensitive to ωD, and therefore T<sup>c</sup> scales with ωD.

where a singularity exists in the density of states.

## F. H3S as a case of bcc with NNN hopping

First-principles calculations show that the Fermi energy occurs near a well-defined peak in the electronic density of states (see Fig. 8 of Ref. [\[27](#page-15-1)] and Fig. 2b or 9 of Ref. [\[28](#page-15-2)]. A very reasonable facsimile of this peak is given by our tight-binding model for the BCC lattice structure with a negative NNN hopping parameter, as shown in the inset of Fig. [12.](#page-10-1) Not surprisingly, T<sup>c</sup> as a function of electron density will display the same peak structure, as illustrated in the main part of Fig. [12](#page-10-1) (compare with Fig. [9\)](#page-9-0). While it is not so useful to attempt an actual fit to T<sup>c</sup> with such limited data and such a limited theoretical framework (see Ref. [\[28\]](#page-15-2) for positive steps in removing several of the limitations), for the sake of completeness, we show T<sup>c</sup> vs electron density for a number of weak coupling strengths. By way of illustration, for V = 2t (V /(16t) = 0.125), with nearest-neighbour hopping t = 1 eV, and ω<sup>D</sup> = 100 meV (all conservative values), then T<sup>c</sup> ≈ 160 K. We expect actual reductions due to retardation and other effects,[28](#page-15-2) but these estimates suggest that the possibility of an enhanced T<sup>c</sup> due to a peaked electronic density of states is quite realistic.

### V. SUMMARY

In the context of BCS theory, we have examined the role of van Hove singularities in the electronic density of states on the superconducting critical temperature and the isotope coefficient (and briefly the pairing gap) for various band structures given by tight-binding models in two and three dimensions. We have adopted the simplest kind of pairing potential, an attractive interaction with energy scale ωD, which gives rise to an order parameter with s-wave symmetry. While the model follows the original BC[S](#page-14-1)<sup>1</sup> paper and is therefore suggestive of a phonon-mediated interaction, it in fact has greater generality.

Many such models have been proposed for high temperature superconductors, and many calculations have been performed, as documented in the references. However, here we have gone beyond the existing literature in two respects. First, we have systematically treated the tight-binding models over all electron densities, with careful account of the BCS number equation (our Eq. [\(3\)](#page-1-0) below T<sup>c</sup> or Eq. [\(11\)](#page-2-0) at Tc), and we have utilized the tight-binding density of states as given, whether for NN hopping only or with NNN hopping included as well. Secondly, we have performed calculations for systems with van Hove singularities in three dimensions. While the simple cubic lattice structure is well known to produce a density of states without singularities (the van Hove singularities manifest themselves in cusps and the derivatives of the density of states), it is less appreciated that both the face-centred cubic and the body-centred cubic exhibit singularities in their densities of states.[17](#page-14-2) It is also surprising that the sc lattice structure results in a singular density of states when the NNN hopping t<sup>2</sup> in increased to t/4. Well before this value is reached the density of states exhibits a strong peak.

We have illustrated that these singularities can give rise to very large enhancements in Tc. Even in the case of the sc lattice, when NNN hopping is included, significant enhancements of T<sup>c</sup> can occur. The bcc lattice is bipartite and therefore nested. As is well known from other nested Fermi surface problems (though certainly less studied for the bcc lattice in particular) other competing instabilities are expected to play an important role, and the BCS calculations provided here become doubtful, as superconducting T<sup>c</sup> will often be suppressed. For example, a CDW instability would certainly compete in the case of a bcc lattice, with ~q = (4π/a, 0, 0) along with equivalent wave vectors. Of course with NNN hopping the CDW instability would likely become incommensurate, and would become suppressed as well. Moreover, this is not an issue with the fcc lattice, as it is not bipartite, the singularity in the density of states in this instance occurs near the top of the band, and other finite q instabilities will 12

not play such a significant role. The possibility still remains that a competing q = 0 instability will suppress superconductivity, but consideration of these is beyond the scope of this paper.

Nonetheless, for both the bcc and fcc lattice structures, some NNN hopping is expected, and we have shown here that this immediately leads to the disappearance of the singularity in the density of states. In fact a 'robust' peak remains in the density of states, and as we have shown, a very large enhancement of T<sup>c</sup> continues to be present, now in a regime where the BCS calculation is more trustworthy, at least for the bcc lattice structure. This is true for several reasons — for example, as discussed competing instabilities will be suppressed, but in addition, narrow structures in the density of states will be smeared both by impurities, and by retardation effects not accounted for in our BCS calculations. Given the number of superconductors with the fcc and bcc lattice structures, it would be interesting to perform a survey to see if there is any correlation between their critical temperatures and the 'remnants' of these van Hove singularities. In fact, the recent discovery of superconducting hydrogen sulfide under high pressure by Drozdov et al.[20](#page-15-3) has motivated theoretical work[27](#page-15-1) that has identified a van Hove singularity in the electronic density of states. The underlying lattice structure is bcc and thus far the origin of this singularity is not clear. This work, particularly the previous subsection, strongly suggests that ultimately the origin of the singularity (and ultimately an important factor in the high superconducting critical temperature) may in fact be BCC structure of the material, along with circumstances that place the Fermi energy in the vicinity of the robust peak that remains even when NNN hopping is included.

We have also computed the isotope coefficient in a variety of cases. It is clear that anomalies in this coefficient will exist due to peaks in the density of states. In some respects the few observations where anomalies are found in known superconductors can be regarded as signatures of peaks in the density of states (although of course other explanations also exist). We have also briefly examined the pairing gap, but there is no significant deviation from what standard BCS theory predicts, i.e. the gap tracks the critical temperature.

#### ACKNOWLEDGMENTS

This work was supported in part by the Natural Sciences and Engineering Research Council of Canada (NSERC). TXRS is a recipient of an "Emerging Leaders in the Americas Program" (ELAP) scholarship from the Canadian government, and we are grateful for this support.

# Appendix A: Density of States within Tight-Binding

The general equation for the Density of States (DOS) is

$$g(\epsilon) = \frac{1}{N} \sum\_{k \in \text{FBZ}} \delta(\epsilon - \epsilon\_k),\tag{A1}$$

where ǫ<sup>k</sup> is the dispersion relation, the summation is over all points in the First Brillouin Zone (FBZ), and N is the number of k-points in the FBZ. Dispersion relations are determined by overlap integrals and geometry; for a Bravais lattice the dispersion relation can be written as

$$\epsilon\_k = -\sum\_{\delta} t\_{\delta} \cos \vec{k} \cdot \vec{\delta} \tag{A2}$$

where the sum is over all neighbours of a particular lattice site, with decreasing amplitude tδ, to reflect the decreasing overlap between atoms that are further apart from one another. This decrease with distance is usually exponential, so very often only nearest neighbour overlaps are considered to be non-zero. It turns out that this simplified model often possesses special symmetries, not necessarily inherent in the more general model, and therefore, if for no other reason, further than nearest neighbour overlaps are often considered as well. Within the tight binding approach only a few nearest neighbours are retained, so for example, we obtain

<span id="page-12-2"></span>
$$\begin{aligned} \epsilon\_k &= -2t \left[ \cos(k\_x a) + \cos(k\_y a) \right] \\ &- 4t\_2 \cos(k\_x a) \cos(k\_y a) \qquad \text{[2D NNN]} \qquad \text{(A3)} \\ \epsilon\_k &= -2t\_s \left[ \cos(k\_x a) + \cos(k\_y a) + \cos(k\_z a) \right] \\ &- 4t\_{s2} \left[ \cos(k\_x a) \cos(k\_y a) + \cos(k\_x a) \cos(k\_z a) \right] \\ &+ \cos(k\_y a) \cos(k\_z a) \qquad \text{[sc NNN]} \qquad \text{(A4)} \end{aligned}$$

$$\epsilon\_{k} = -8t\_{b} \left[ \cos(\frac{k\_{x}a}{2}) \cos(\frac{k\_{y}a}{2}) \cos(\frac{k\_{z}a}{2}) \right] \quad \text{[bcc NNN]}$$

$$\text{v} \quad [\text{v} \quad [\text{v} \quad \text{ } (k\_{z}) \quad [\text{v} \quad \text{ } (k\_{z}) \quad [\text{v} \quad \text{ } (k\_{z})]]] \quad (\text{A.5})$$

$$-2t\_{b2}\left[\cos(k\_x a) + \cos(k\_y a) + \cos(k\_z a)\right] \qquad \text{(A5)}$$

$$\int\_{\text{all}} \left[\begin{array}{c} k\_x a \ \left< \right|\_{\text{green}}, k\_y a \end{array} \right>\_{\text{yellow}} k\_x a \Big|\_{\text{yellow}}, k\_z a \Big>\_{\text{yellow}} k\_z a \Big>$$

$$\begin{aligned} \epsilon\_k &= -4t\_f \left[ \cos(\frac{k\_x a}{2}) \cos(\frac{k\_y a}{2}) + \cos(\frac{k\_x a}{2}) \cos(\frac{k\_z a}{2}) \right. \\ &\quad + \cos(\frac{k\_y a}{2}) \cos(\frac{k\_z a}{2}) \text{[fcc NNN]]} \end{aligned}$$

$$-2t\_{f2} \left[ \cos(k\_x a) + \cos(k\_y a) + \cos(k\_z a) \right]. \tag{A6}$$

We repeat here important definitions already mentioned in the text. The distance a is the nearest neighbour distance in the 2D and simple cubic (sc) cases, and is the length of the cube in the body-centred cubic (bcc) and face-centred cubic (fcc) cases; these latter two each contain 8 atoms, one at each vertex, along with one in the centre (bcc) and six on the face centres (fcc). Also, t, ts, tb, and t<sup>f</sup> are the nearest neighbour hopping parameters and t2, ts2, tb2, and tf<sup>2</sup> are the next-nearest neighbour hopping parameters for the 2D square, 3D sc, 3D bcc, and 3D fcc lattices, respectively.

![](0__page_12_Figure_12.jpeg)

<span id="page-12-1"></span>FIG. 13. Plot of the tight-binding 2D density of states for various values of the next-nearest neighbour (NNN) hopping parameter, ρ ≡ t2/t, as given analytically in Eq. [\(A7\)](#page-12-0). A logarithmic singularity remains even in the presence of NNN. Note that the results for negative values of ρ are symmetric (about ǫ = 0) to those shown with positive values of ρ. As mentioned in the text, numerical results, using Eq. [\(A8\)](#page-13-1), are also shown, and are indistinguishable from the analytical results. The black dashed curve is the approximation given by Eq. [\(15\)](#page-3-0) in the text, valid for ρ = 0.

Note that without NNN hopping these have bandwidths W of 8t, 12ts, 16tb, and 16t<sup>f</sup> , respectively. We have additionally included next-nearest neighbour hopping in all these cases; for the 3D cases all but the sc case exhibit singularities when only nearest-neighbour hops are considered; in two dimensions the existence of a singularity is retained as next-nearest neighbour hops are introduced, while in the three dimensions, in either the face-centred cubic (fcc) or body-centred cubic (bcc) cases, the singularity disappears.

For our purposes the important property emerging from these different band dispersions is the shape of the density of states, defined above. In two dimensions, the DOS can be determined analytically in terms of complete elliptic integrals. The result is

<span id="page-12-0"></span>
$$g\_{\rm 2D}(\rho;\epsilon) = \frac{1}{2\pi^2 t a^2} \frac{1}{\sqrt{1 - 4\rho \bar{\epsilon}}} K \left[ 1 - \frac{\left(\rho - \bar{\epsilon}\right)^2}{1 - 4\rho \bar{\epsilon}} \right], \quad \text{(A7)}$$

where ¯ǫ ≡ ǫ/(4t) and ρ ≡ t2/t, with the restriction that −1/2 < ρ < 1/2. A different expression applies for |ρ| > 1/2, but we omit this regime as being unphysical. Fig. [\(13\)](#page-12-1) shows the DOS for a few values of ρ > 0; note that g2D(ρ; ǫ) = g2D(−ρ; −ǫ), so the DOS with negative values of ρ are mirror images of those shown. It is evident from Eq. [\(A7\)](#page-12-0) that the logarithmic singularity occurs at ǫ = 4t2.

![](0__page_13_Figure_1.jpeg)

<span id="page-13-2"></span>FIG. 14. Plot of the tight-binding 3D SC density of states for various values of the next-nearest neighbour (NNN) hopping parameter, ρ ≡ t2/t. Note that a singularity develops at the top of the band, for ǫ = 3t, as the two van Hove singularities (originally at ǫ = 2t and at ǫ = 6t for t<sup>2</sup> = 0) merge into one. Results are shown for positive t<sup>2</sup> since the results for negative values of ρ are symmetric (about ǫ = 0) to those shown with positive values of ρ.

Also shown, but indistinguishable from the analytical curves drawn using Eq. [\(A7\)](#page-12-0), are results obtained numerically, using the Gaussian representation for a δ-function. With x ≡ kxa/π and y ≡ kya/π, the 2D version for the first dispersion given in Eq. [\(A6\)](#page-12-2) is

<span id="page-13-1"></span>
$$g\_{\delta}(\epsilon) = \frac{1}{2ta^2} \frac{1}{\sqrt{\pi \delta^2}} \int\_0^1 dx \int\_0^1 dy \, \exp\left\{-\left[\frac{\epsilon - \epsilon\_k}{2t\delta}\right]^2\right\},\tag{A8}$$

where the approximation improves for smaller value of the smearing parameter, δ. In Fig. [\(13\)](#page-12-1) we use δ = 0.0005t. For 3D dispersions an additional integral over z ≡ kza/π from zero to unity is required and a <sup>2</sup> <sup>→</sup> <sup>a</sup> 3 .

In three dimensions, the integrals must be done numerically. It is straightforward to simplify some of the results when there is no next-nearest-neighbour hopping,[17](#page-14-2) and we cite some of these results for convenience. For the rest we develop formulas in some instances or simply use Eq. [\(A8\)](#page-13-1), as this is straightforward and continues to work extremely well, even in 3 dimensions.

The result for SC with next-nearest-neighbour hopping is

$$g\_{\rm SC}(\epsilon) = \int \frac{dx}{2\pi^2 t a^3} \frac{K(y)}{\sqrt{(1 + 2\rho \cos \pi x)^2 - \rho(4\bar{\epsilon} + 2 \cos \pi x)}},\tag{A9}$$

![](0__page_13_Figure_9.jpeg)

<span id="page-13-0"></span>FIG. 15. Plot of the tight-binding 3D BCC density of states for various values of the next-nearest neighbour (NNN) hopping parameter, ρ ≡ t2/t. Note that the singularity for ǫ = 0 disappears as t<sup>2</sup> becomes non-zero. Results are shown for positive t<sup>2</sup> since the results for negative values of ρ are symmetric (about ǫ = 0) to those shown with positive values of ρ. Even with non-zero t<sup>2</sup> a significant peak in the density of states remains. The inset shows numerical results as a function of ǫ very close to the cusp located at ǫcusp = 6tρ−4ρ 3 t for ρ = t2/t = 0.3, for various values of the smearing parameter, δ/t = 0.01, 0.005, 0.002, 0.001, 0.0001.

where

$$y \equiv 1 - \frac{(\rho - \bar{\epsilon} - \frac{1}{2} \cos \pi x)^2}{(1 + 2\rho \cos \pi x)^2 - \rho (4\bar{\epsilon} + 2 \cos \pi x)} \qquad \text{(A10)}$$

and ¯ǫ ≡ ǫ/(4t), K(y) is the complete elliptic integral of the first kind, and ρ ≡ t2/t is the ratio of the NNN to the NN hopping amplitude, and a 3 is the unit cell volume. The limits on the integration are such that y remains positive and less than unity at all times. As remarked in the text, this density of states has no singularities for t<sup>2</sup> = 0 (there remain van Hove singularities in the form of cusps and singularities in the derivative of g(ǫ)), but develops a singularity as t<sup>2</sup> increases. This is evident in Fig. [14,](#page-13-2) where we use the 3D version of Eq. [\(A8\)](#page-13-1) to plot the density of states.

The result for the BCC lattice with nearest-neighbour hopping only is[17](#page-14-2)

$$g\_{\rm BCC}(\epsilon) = \frac{2}{a^3} \frac{1}{2\pi^3 t} \int\_{|\bar{\epsilon}|}^1 dx \, \frac{1}{\sqrt{x^2 - \bar{\epsilon}^2}} K \left[ 1 - x^2 \right]. \tag{A11}$$

Note that the unit cell volume for the BCC lattice is a <sup>3</sup>/2; that is why we isolate this factor at the front of

![](0__page_14_Figure_1.jpeg)

<span id="page-14-0"></span>FIG. 16. Plot of the tight-binding 3D FCC density of states for various values of the next-nearest neighbour (NNN) hopping parameter, ρ ≡ t2/t. For t<sup>2</sup> = 0 there is a singularity at the top of the band (red curve as indicated). As t<sup>2</sup> becomes nonzero the singularity disappears and the maximum shifts to the left. In fact, as t<sup>2</sup> grows the maximum in the density of states becomes 'robust' in the sense that a significant area exists in the maximum region (also the case with BCC and with the 2D result — see the ρ = 0.45 result in Fig. [13\)](#page-12-1).

the previous formula. Using the fact that K(1 − x 2 ) → ln(4/x) as x → 0, one can straightforwardly derive that

$$\lim\_{\bar{\epsilon}\to 0} g\_{\rm BCC}(\epsilon) = \frac{2}{a^3} \frac{1}{2\pi^3 t} \left[ \frac{3}{2} \ln^2(\frac{1}{|\bar{\epsilon}|}) + 3 \ln 2 \ln(\frac{1}{|\bar{\epsilon}|}) + 2(\ln 2)^2 \right],\tag{A12}$$

so that the divergence is a <sup>≈</sup> ln<sup>2</sup> 1 |ǫ¯| , singularity, stronger than occurs in two dimensions. For the case with NNN hopping we use Eq. [\(A8\)](#page-13-1) to determine the result numerically; these are shown in Fig. [\(15\)](#page-13-0). The van Hove points are at energies −8t − 6tρ (bottom of the band), 2tρ, 6tρ <sup>−</sup> <sup>4</sup>tρ<sup>3</sup> , 6tρ, and 8t − 6tρ (top of the band). Note that the bandwidth remains 16t even when NNN hopping is non-zero.

Finally, for FCC, we show numerical results for NNN hopping as well. Our use of the exponential representation of the δ-function still allows sufficient resolution to show the various van Hove singularities, apparent in Fig. [\(16\)](#page-14-0). However, with NNN hopping the singularity at the top of the band for t<sup>2</sup> = 0 disappears, but various cusps remain, signifying discontinuities in the first derivative. Note that the FCC lattice is not bipartite, and nesting is not present, even in the case of NN hopping only.

- <span id="page-14-1"></span>1 J. Bardeen, L.N. Cooper and J.R. Schrieffer, Theory of Superconductivity, Phys. Rev. 106, 162 (1957); Phys. Rev. 108, 1175 (1957).
- 2 J. Labb´e. S. Bariˇsi´c and J. Friedel, Strong-coupling Superconductivity in V3X Type of Compounds, Phys. Rev. Lett. 19, 1039 (1967).
- 3 S.J. Nettel and H. Thomas, Electron-density of States and Superconducting T<sup>c</sup> in A15-Compounds, Solid State Commun. 21 683 (1977).
- <sup>4</sup> P. Horsch and H. Rietschel, New Aspect of Superconductivity in A-15 Compounds, Z. Phys. B 27 153 (1977).
- 5 S.G. Lie and J.P. Carbotte, Dependence of T<sup>c</sup> on Electronic Density of States. Solid State Commun. 26 511 (1978).
- <sup>6</sup> K.M. Ho, M.L. Cohen, and W.E. Pickett, Maximum Superconducting Transition-Temperatures in A15 Compounds, Phys. Rev. Lett. 41 815 (1978).
- <sup>7</sup> W.E. Pickett, Effect of a Varying Density of States on Superconductivity, Phys. Rev. B 21 3897 (1980).
- <sup>8</sup> B. Mitrovi´c and J.P. Carbotte, Effects of Energy-Dependence in the Electronic Density of States on some Normal State Properties, Can. J. Phys. 61 758 (1983); Effects of Energy-Dependence in the Electronic Density of States on some Superconducting Properties, Can. J. Phys. 61 784 (1983); Free-Energy Formula for a Strong Coupling Superconductor with Energy-dependent Electronic Density of States, Can. J. Phys. 61 872 (1983).
- 9 See, for example, the very recent review by G.R. Stewart, Physica C 514, 28-35 (2015), titled "Superconductivity in the A15 Structure". This is a review in the Special Issue on Superconducting Materials, edited by J.E. Hirsch, M.B.

Maple and F. Marsiglio.

- <sup>10</sup> J.E. Hirsch and D.J. Scalapino, Enhanced Superconductivity in Quasi Two-dimensional Systems, Phys. Rev. Lett. 56, 2732 (1986).
- <sup>11</sup> J. Labb´e and J. Bok, Superconductivity in Alcaline-Earth-Substituted La2CuO4: a Theoretical Model, Europhys. Lett. 3, 1225-1230 (1987).
- <sup>12</sup> C.C. Tsuei, D.M. Newns, C.C. Chi and P.C. Pattnaik, Anomalous Isotope Effect and van Hove Singularity in Superconducting Cu Oxides, Phys. Rev. Lett. 65, 2724-2727 (1990).
- <sup>13</sup> R.S. Markiewicz, A Survey of the van Hove Scenario for High-T<sup>c</sup> Superconductivity with Special Emphasis on Pseudogaps and Striped Phases, J. Phys. Chem. Solids 58, 1179- 1310 (1997).
- <sup>14</sup> J. Bok and J. Bouvier, Superconductivity and the van Hove Scenario, J. Supercond. Nov. Magn. 25, 657-667 (2012).
- <sup>15</sup> O.K. Andersen, A.I. Liechtenstein, O. Rodriguez, I.I. Mazin, O. Jepsen, V.P. Antropov, O. Gunnarsson, and S. Gopalan, Electrons, phonons, and their interaction in YBa2Cu3O7, Physica C 185-189, 147-155 (1991).
- <sup>16</sup> A.A. Abrikosov, J.C. Campuzano, and K. Gofron, Experimentally observed extended saddle point singularity in the energy spectrum of YBa2Cu3O6.9 and YBa2Cu4O8 and some of the consequences, Physica C 214, 73-79 (1993).
- <span id="page-14-2"></span><sup>17</sup> R.J. Jelitto, The Density of States of some Simple Excitations in Solids, J. Phys. Chem. Solids 30, 609-626 (1969).
- <span id="page-14-3"></span><sup>18</sup> Defang Duan, Yunxian Liu, Fubo Tian, Da Li, Xiaoli Huang, Zhonglong Zhao, Hongyu Yu, Bingbing Liu, Wenjing Tian, and Tian Cui, Pressure-induced metallization of

dense (H2S)2H<sup>2</sup> with high-T<sup>c</sup> superconductivity, Scientific Reports 4, 6968-1-6 (2014).

- <span id="page-15-4"></span><sup>19</sup> N. Bernstein, C. Stephen Hellberg, M.D. Johannes, I.I. Mazin, and M.J. Mehl, What superconducts in sulfur hydrides under pressure and why, Phys. Rev. B91, 060511(R)-1-5 (2015).
- <span id="page-15-3"></span><sup>20</sup> A.P. Drozdov, M.I. Eremets, I.A. Troyan, V. Ksenofontov, and S.I. Shylin, Conventional superconductivity at 203 kelvin at high pressures in the sulfur hydride system, Nature 525, 73-76 (2015).
- <sup>21</sup> We became aware of Ref. [\[28\]](#page-15-2) only after this paper was initially submitted; they have investigated retardation effects and indeed find that a reduction in T<sup>c</sup> will occur as a result of their inclusion.
- <sup>22</sup> P. Nozi`eres and S. Schmitt–Rink, Bose Condensation in an Attractive Fermion Gas - From Weak to Strong Coupling Superconductivity, J. Low Temp. Phys. 59, 195 (1985).
- <sup>23</sup> A recent description of the two-dimensional electron gas with attractive interactions within the T-matrix formalism is given in F. Marsiglio, P. Pieri, A. Perali, F. Palestini, and G.C. Strinati, Pairing effects in the normal phase of a twodimensional Fermi gas, Phys. Rev. B91, 054509 (2015).
- <sup>24</sup> F. Marsiglio, Eliashberg Theory of the Critical Temperature and Isotope Effect. Dependence on Bandwidth, Band-Filling, and Direct Coulomb Repulsion, J. Low Temp. Phys. 87 659-682 (1992).
- <sup>25</sup> F. Marsiglio and J.P. Carbotte, 'Electron-Phonon Superconductivity', Review Chapter in Superconductivity, Conventional and Unconventional Superconductors, edited by K.H. Bennemann and J.B. Ketterson (Springer-Verlag, Berlin, 2008), pp. 73-162.
- <sup>26</sup> G.M. Eliashberg, Interactions between Elecrons and Lattice Vibrations in a Superconductor, Zh. Eksperim. i Teor. Fiz. 38 966 (1960); Soviet Phys. JETP 11 696-702 (1960).
- <span id="page-15-1"></span><sup>27</sup> Y. Quan and W.E. Pickett, Van Hove singularities and spectral smearing in high-temperature superconducting H3S, Phys. Rev. B93, 104526 (2016).
- <span id="page-15-2"></span><sup>28</sup> Wataru Sano, Takashi Koretsune, Terumasa Tadano, Ryosuke Akashi, and Ryotaro Arita, Effect of Van Hove singularities on high-T<sup>c</sup> superconductivity in H3S, Phys. Rev. B93, 094525-1-16 (2016).
- <sup>29</sup> J.R. Schrieffer, In: Theory of Superconductivity (Benjamin/Cummings, Don Mills, 1964).
- <sup>30</sup> M. Tinkham, In: Introduction to Superconductivity, (Second Edition, McGraw–Hill, New York, 1996).
- <sup>31</sup> In this manner retardation effects are taken into account in BCS theory in a very phenomenological way through the imposed cutoff. This cutoff is imposed in momentum space (not in frequency space); nonetheless, use of a simple identity for the factor 1−2f(Ek) allows the right-hand-side of the so-called gap equation [Eq. [\(2\)](#page-1-1)] to be rewritten in terms of Matsubara frequencies with a cutoff in this space, so it better resembles Eliashberg theory.
- <sup>32</sup> Frank W.J. Olver, Daniel W. Lozier, Ronald F. Boisvert, and Charles W. Clark, NIST Handbook of Mathematical Functions, (Cambridge University Press, Cambridge, 2010).
- <sup>33</sup> One of us (FM) has repeatedly asked colleagues in the superconducting community around the world about this for the last 20 years, and with one exception they were not aware of the divergences in three dimensions. The Jelitto paper[17](#page-14-2) has been cited more than 140 times, so clearly some researchers are aware of this fact. However, neither of Refs. [\[18](#page-14-3) and [19\]](#page-15-4) cite the Jelitto paper. While later references[27](#page-15-1)[,34](#page-15-5) cite a van Hove singularity, neither of these seem to be aware that the relevant van Hove singularity is likely a remnant of the BCC singularity (when only NN hopping is included). Ref. [\[35\]](#page-15-6) actually uses a rather sophisticated tight-binding model to achieve a good fit with first-principle calculations, but they also appear to be unaware of the BCC singularity lurking nearby in parameter space.
- <span id="page-15-5"></span><sup>34</sup> D.A. Papaconstantopoulos, B.M. Klein, M.J. Mehl, and W.E. Pickett, Cubic H3S around 200 GPa: An atomic hydrogen superconductor stabilized by sulfur, Phys. Rev. B91, 184511-1-5 (2015).
- <span id="page-15-6"></span><sup>35</sup> Luciano Ortenzi, Emmanuele Cappelluti, and Luciano Pietronero, Band structure and electron-phonon coupling in H3S: a tight-binding model, [arXiv:1511.04304.](http://arxiv.org/abs/1511.04304)
- <span id="page-15-0"></span><sup>36</sup> F. Marsiglio, Pairing and charge-density-wave correlations in the Holstein model at half-filling, Phys. Rev. B42, 2416- 2424 (1990).
- <sup>37</sup> J.M. Kosterlitz and D.J. Thouless, Ordering, Metastability and Phase-transitions in 2-Dimensional Systems, J. Phys. C6, 1181 (1973).