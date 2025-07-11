# arXiv:2410.20078

**Paper ID:** b25604adef69f66a0fd668d72fd72bdb

**PDF Path:** apl/Superconductors/arXiv:2410.20078.pdf

**Processing Status:** complete

**Captions Added:** 9

**Generated:** 2025-06-24T13:44:28.033087

---

# A Time-Dependent Ginzburg-Landau Framework for Sample-Specific Simulation of Superconductors for SRF Applications

Aiden V. Harbick[∗](#page-0-0) and Mark K. Transtrum[†](#page-0-1)

Department of Physics and Astronomy, Brigham Young University, Provo, Utah 84602, USA

(Dated: October 25, 2024)

Modern superconducting radio frequency (SRF) applications require precise control of a wide range of material properties, from microscopic material parameters to mesoscopic/macroscopic surface structures. Mesoscopic simulation of superconductors has proven itself to be a powerful tool in SRF development, connecting microscopic properties to the mesoscopic structures of the material. We outline a Time-Dependent Ginzburg-Landau (TDGL) framework which combines both experimental measurements and theoretical estimations of the properties of realistic sample materials into spatially varying TDGL parameters. We also derive a way to estimate SRF quality factors from TDGL solutions. We demonstrate an application of this framework by modeling Sn-deficient islands in Nb3Sn samples. We calculate the dissipation due to interactions of the superconducting order parameter and electromagnetic fields with these defects and estimate the impact to SRF cavity quality factor.

#### I. INTRODUCTION

Superconducting Radio-Frequency (SRF) cavities are a crucial component of particle accelerators, as they utilize AC electromagnetic fields to accelerate beams of charged particles. Nb has been the industry standard for SRF applications for decades due to its high critical temperature (∼9K) relative to the other elemental superconductors. Within the past decade, the need for SRF cavities with capabilities beyond the limits of Nb cavity performance has led to the study of a variety of alternative SRF materials. Among these materials, Nb3Sn has emerged as a promising candidate; Nb3Sn boasts both a higher critical temperature (∼18K) and higher critical fields [\[1,](#page-12-0) [2\]](#page-12-1). One particular advantage of SRF cavities (as compared to traditional normal conducting RF cavities) is their high quality factors (Q) [\[3\]](#page-12-2). A major benefit of Nb3Sn SRF cavities compared to their Nb counterparts is that they can maintain similar Qs (on the order of 1010) at higher temperatures (4.2K vs. 2K) [\[2\]](#page-12-1), significantly reducing cryogenic costs. NbZr is another promising alternative SRF material which has seen recent attention [\[4,](#page-12-3) [5\]](#page-12-4), with most existing NbZr samples exhibiting critical temperatures between 10 and 13K (the theoretical maximum is 17.7K), but this material has not yet been tested at cavity scale. The simulations in this paper focus on Nb3Sn, but the methods we will present can be generalized to any material of interest.

The oscillating electric fields used for acceleration in SRF cavities induce magnetic fields parallel to the cavity surface, so for large accelerating gradients, the critical magnetic fields of the cavity material are the fundamental limits on cavity performance. Type-II superconductors such as Nb and Nb3Sn have two critical fields, Hc<sup>1</sup> and Hc2. For fields below Hc1, the Meissner state is stable and magnetic flux is expelled from the cavity. Between Hc<sup>1</sup> and Hc2, there is a mixed state in which superconducting vortices trap lines of magnetic flux, forming normal cores inside the otherwise superconducting state. Under an AC field, as is the case for SRF cavities, the vortices quickly move in and out of the cavity over the course of an AC cycle. This vortex motion leads to large amounts of dissipation [\[6\]](#page-12-5). For fields above Hc2, the mixed state becomes unstable and in this state SRF cavities will quench (i.e. go normal conducting). It is important to note that this is not the only mechanism for SRF cavity quench, the dissipation caused by moving vortices in the mixed state can cause heating in the cavity, which can also lead to quenching through a change in cavity temperature. As such, for SRF applications, it is important that the cavity remain within the Meissner state during operation.

While the Meissner state is no longer thermodynamically stable above Hc1, it can remain metastable up until the so-called superheating field, Hsh [\[7\]](#page-12-6). It is well known that most high-power SRF cavities operate in this metastable Meissner state [\[8\]](#page-12-7). As such, Hsh is the theoretical limiting field for operation of SRF cavities, since the dissipative vortices which are detrimental to SRF performance become unavoidable for fields above Hsh. Hsh has been studied for decades by condensed matter theorists. These studies have most commonly been within a Ginzburg-Landau (GL) framework [\[9](#page-12-8)[–12\]](#page-12-9), but the superheating field has also been studied extensively utilizing the Eilenberger equations [\[13](#page-12-10)[–15\]](#page-12-11).

Hsh provides the maximum possible field (and therefore the maximum accelerating gradient) for SRF cavity operation, but local features of a material such as impurities or surface geometries can act as nucleation sites for vortices. This means that in practice, realistic material samples will be limited by what we will call the vortex penetration field, Hvort, which is the lowest field at which the material nucleates vortices. This quantity can vary greatly between different samples and depends on a large variety of different effects, so estimation of Hvort for re-

<span id="page-0-0"></span><sup>∗</sup> [aharbick@student.byu.edu](mailto:aharbick@student.byu.edu)

<span id="page-0-1"></span><sup>†</sup> [mktranstrum@byu.edu](mailto:mktranstrum@byu.edu)

alistic sample materials remains a rich area of research. For the case of Nb3Sn in particular, theoretical Hsh calculations suggest that Nb3Sn cavities could reach accelerating gradients as high as around 100 MV/m, yet the highest accelerating field achieved so far by a Nb3Sn SRF cavity is around 24 MV/m [\[16\]](#page-12-12), with most other cavities reaching their quench field well below this. Additionally, many existing cavities exhibit a phenomenon in which Q significantly degrades as the cavity approaches its quench field, a phenomenon dubbed "Q-slope" [\[17\]](#page-12-13), also sometimes called high field Q-slope (HFQS) when it occurs primarily at higher fields near the quench field. These performance degradations are the result of material defects introduced during the Nb3Sn growth process, so understanding how different defects seen within samples affect things like Hvort or dissipation more generally is critical to developing better growth techniques.

The need to accurately model specific features such as material impurities within superconducting materials motivates us to develop a framework which will allow us to directly model the spatial variations of superconducting properties due to the material compositions observed in realistic sample materials. To do this, we use timedependent Ginzburg-Landau (TDGL) theory. TDGL has already proven itself to be a powerful tool for mesoscopicscale simulations relevant to SRF applications [\[18–](#page-12-14)[21\]](#page-12-15). Besides SRF simulations, TDGL has broad application such as in single photon detectors [\[22,](#page-13-0) [23\]](#page-13-1), superconducting quantum interference devices (SQUIDs) [\[24,](#page-13-2) [25\]](#page-13-3), weak links [\[26](#page-13-4)[–30\]](#page-13-5), or superconducting nanowires [\[31–](#page-13-6)[33\]](#page-13-7).

TDGL is a mesoscopic-scale model which abstracts the microscopic details of superconductivity into quantities which can be used to describe things like vortex dynamics. To model sample specific materials in order to understand the mechanisms behind Q-slope or other cavity quenching phenomena in Nb3Sn SRF cavities, we will make use of both experimental and theoretical work which has been performed to analyze the microscopic properties of Nb3Sn.

There has been a large body of work experimentally characterizing SRF grade vapor-diffused Nb3Sn samples. The primary suspect for SRF performance degradation comes from defects or other imperfections in the Nb3Sn surface significantly lowering the barrier to flux penetration [\[34\]](#page-13-8). In particular, defects which have been studied are abnormally thin or patchy grains [\[35–](#page-13-9)[37\]](#page-13-10), Snsegregated grain boundaries [\[19,](#page-12-16) [38,](#page-13-11) [39\]](#page-13-12), and Sn-deficient regions [\[35,](#page-13-9) [36,](#page-13-13) [40,](#page-13-14) [41\]](#page-13-15). In addition to experimental characterizations, there have also been a variety of ab initio calculations for Nb3Sn using density functional theory. In addition to calculations of general properties of Nb3Sn [\[42\]](#page-13-16), such as the electron and phonon density of states and Eliashberg spectral function, these quantities have also been estimated with respect to varying intrinsic strain [\[43\]](#page-13-17) as well as normal resistivity [\[44\]](#page-13-18). Variations in the superconducting T<sup>c</sup> as well as electron density of states have also been calculated with respect to varying tin concentration [\[45,](#page-13-19) [46\]](#page-13-20), which applies to both Snsegregation at grain boundaries and Sn-deficient regions. These Sn-deficient regions, or Sn-deficient islands as we will call them, are the primary material defect we will study in order to validate our methods.

Experimental characterizations can give data about the material compositions and physical structure of superconducting materials, and ab initio calculations provide detailed descriptions of the electronic/phononic structure and the resulting superconducting properties, both of these a microscopic scale. TDGL plays the role of modeling mesoscopic scale phenomena (such as vortex dynamics) which are difficult or even impossible to measure directly via experiment, and are too large to easily model with microscopic scale theories such as DFT. There have been a number of studies which have used TDGL to model material inhomogenieties [\[18,](#page-12-14) [19,](#page-12-16) [47–](#page-13-21) [51\]](#page-14-0), but these studies did not incorporate precise values of the TDGL parameters from microscopic material models. The limitation to the approach used in these references is that it requires either looking through a large portion of the TDGL parameter space in order to find values which lead to expected predictions, or more commonly, picking somewhat arbitrary values, which limits confidence in the results.

In this paper, we outline a new framework in TDGL theory which allows us to directly calculate the values of the TDGL parameters based on local properties of the superconductor. This framework allows us to model realistic features of superconductor samples, and estimate critical fields as well as energy dissipation under changing fields. Under our framework, TDGL serves as a link between experimental material characterizations and ab initio calculations of material-specific parameters, allowing us to further connect these microscopic characterizations with macroscopic SRF performance metrics in a sample-specific way.

The accelerator physics community is large and active, and many studies are highly multidisciplinary. The method presented in this work boasts the advantage of leveraging the expertise of accelerator physicists and engineers, materials scientists, and condensed matter theorists to make sample-specific predictions about realistic SRF materials.

This paper is organized as follows: In section [II,](#page-2-0) we discuss the TDGL equations and show how to calculate TDGL parameters from material properties. We then outline how the spatial variation of the TDGL parameters properties is estimated, combining the results of DFT calculations with experimental material characterizations. We briefly outline the equation formulation we use to solve TDGL with a finite element method. We end the section by showing how dissipation can be calculated from TDGL solutions and deriving how SRF cavity quality factors can be calculated based on this dissipation. In section [III](#page-6-0) we demonstrate our framework by first applying the quality factor estimates to a defectfree Nb simulation. We then estimate critical fields and dissipation for Sn-deficient islands in Nb3Sn, comparing the estimated quality factor of Nb3Sn coated SRF cavities with different surface densities of Sn deficient islands. Finally, in section [IV,](#page-10-0) we conclude the paper, justifying the use of TDGL given its known limitations, and discuss the implications of our method and results for future SRF cavity research.

#### <span id="page-2-0"></span>II. METHODS

## A. The Time-Dependent Ginzburg-Landau Equations

Ginzburg-Landau (GL) theory is one of the oldest theories of superconductivity, and it remains relevant today owing to its relative simplicity and direct physical insights into the electrodynamic response of superconductors under static applied fields and currents [\[52\]](#page-14-1). The time-dependent Ginzburg-Landau (TDGL) equations were originally proposed by Schmid [\[53\]](#page-14-2) in 1966 and Gor'kov and Eliashberg [\[54\]](#page-14-3) derived them rigorously from BCS theory later in 1968. The TDGL equations (in Gaussian units) are given by:

$$
\Gamma \left( \frac{\partial \psi}{\partial t} + \frac{ie\_s \phi}{\hbar} \psi \right) + \frac{1}{2m\_s} \left( -i\hbar \nabla - \frac{e\_s}{c} \mathbf{A} \right)^2 \psi + \alpha \psi + \beta |\psi|^2 \psi = 0 \tag{1}
$$

$$\frac{4\pi\sigma\_n}{c}\left(\frac{1}{c}\frac{\partial\mathbf{A}}{\partial t} + \nabla\phi\right) + \nabla \times \nabla \times \mathbf{A} - \frac{2\pi i e\_s \hbar}{m\_s c} (\psi^\* \nabla \psi - \psi \nabla \psi^\*) + \frac{4\pi e\_s^2}{m\_s c^2} |\psi|^2 \mathbf{A} = 0. \tag{2}$$

These equations are solved for the complex superconducting order parameter, ψ, and the magnetic vector potential, A. The magnitude squared of ψ is proportional to the density of superconducting electrons. The parameters α and β are phenomenological, and were originally introduced as coefficients of the series expansion of the Ginzburg-Landau free energy. Additionally, ϕ is the scalar potential; σ<sup>n</sup> is the normal electron conductivity; Γ is the phenomenological relaxation rate of ψ. Furthermore, e<sup>s</sup> = 2e and m<sup>s</sup> = 2m<sup>e</sup> represent the total charge and total effective mass of a Cooper pair, respectively. The TDGL equations are subject to boundary conditions

$$\left(i\hbar\nabla\psi + \frac{e\_s}{c}\mathbf{A}\psi\right)\cdot\mathbf{n} = 0\tag{3}$$

$$(\nabla \times \mathbf{A}) \times \mathbf{n} = \mathbf{H}\_a \times \mathbf{n} \tag{4}$$

$$\left(\nabla\phi + \frac{1}{c}\frac{\partial\mathbf{A}}{\partial t}\right)\cdot\mathbf{n} = 0,\tag{5}$$

where n is the outward normal vector to the boundary surface and H<sup>a</sup> is the applied magnetic field. Eq. [3](#page-2-1) ensures no current flows out of the superconducting domain, and noting that E = −∇ϕ − 1 c ∂A ∂t , Eqs. [4](#page-2-2) and [5](#page-2-3) are electromagnetic interface conditions with an applied magnetic field.

The parameters α, β, and Γ were originally introduced into the theory as phenomenological, temperaturedependent constants [\[55\]](#page-14-4). It is worth noting that α < 0 corresponds to the superconducting state whereas α ≥ 0 corresponds to the normal state; β is strictly positive regardless of the system's state. The TDGL equations can also be derived from microscopic theory using the timedependent Gor'Kov equations [\[54\]](#page-14-3). A useful consequence

of this derivation is that it allows the TDGL parameters to be directly related to experimentally observable properties of the material in question. The material dependencies are given by Ref. [56:](#page-14-5)

<span id="page-2-7"></span><span id="page-2-6"></span><span id="page-2-4"></span>
$$\begin{split} \alpha(\nu(0), T\_c, T) &= -\nu(0) \left( \frac{1 - \frac{T^2}{T\_c^2}}{1 + \frac{T^2}{T\_c^2}} \right) \\ &\approx -\nu(0) \left( 1 - \frac{T}{T} \right) \end{split} \tag{6}$$

<span id="page-2-1"></span>
$$\chi(\nu(0), T\_c, T) = \frac{7\zeta(3)\nu(0)}{8\pi^2 T\_c^2} \left(\frac{1}{1 + \frac{T^2}{T\_c^2}}\right)^2 \qquad (7)$$

<span id="page-2-5"></span>
$$\approx \frac{7\zeta(3)\nu(0)}{8\pi^2 T\_c^2}$$

$$\Gamma(\nu(0), T\_c) = \frac{\nu(0)\pi\hbar}{8T\_c},\tag{8}$$

<span id="page-2-3"></span><span id="page-2-2"></span>where ν(0) is the density of states at the Fermi-level, T<sup>c</sup> is the critical temperature, T is the temperature, and ζ(x) is the Riemann zeta function.

A major contribution of this paper is to introduce a framework for modeling sample-specific features of superconducting materials by connecting ab initio calculations of the material's properties to experimental characterizations of the material. Eqs. [6-](#page-2-4)[8](#page-2-5) determine how the parameters of TDGL depend on the underlying material properties. We allow these parameters to vary spatially to capture the sample-specific features observed from experimental characterizations.

Additionally, when solving the TDGL equations numerically, it is standard to normalize all the parameters of the model in order to obtain dimensionless quantities. In order to satisfy the above requirements, the steps are as follows: Pick a reference value for each of α, β, Γ, and σn, these will most often just be the corresponding values for the bulk material. Label these reference values α0, β0, Γ0, and σn0. Define dimensionless spatially varying functions, a(r), b(r), γ(r), and s(r), relative to their reference values. Apply the following transformations to Eqs. [1](#page-2-6) and [2:](#page-2-7)

α −→ α0a(r) (9)

$$
\beta \longrightarrow \beta\_0 b(\mathbf{r}) \tag{10}
$$

$$
\Gamma \longrightarrow \Gamma\_0 \gamma(\mathbf{r}) \tag{11}
$$

$$
\sigma\_n \longrightarrow \sigma\_{n0} s(\mathbf{r}).\tag{12}
$$

Then proceed with standard nondimensionalization procedures for TDGL (see the Appendix for more details). The advantage of this approach is that the nondimensionalization procedures, when used on these transformed equations, leave behind only the dimensionless functions which capture the spatial variation of the sample material in natural units. The resulting equations are

$$\gamma \left( \frac{\partial \psi}{\partial t} + i \kappa\_0 \phi \psi \right) + \left( \frac{-i}{\kappa\_0} \nabla - \mathbf{A} \right)^2 \psi - a\psi + b|\psi|^2 \psi = 0 \tag{13}$$

$$\frac{1}{u\_0} \left( \frac{\partial \mathbf{A}}{\partial t} + \nabla \phi \right) + \nabla \times \nabla \times \mathbf{A} + \frac{i}{2\kappa\_0} \left( \psi^\* \nabla \psi - \psi \nabla \psi^\* \right) + |\psi|^2 \mathbf{A} = 0,\tag{14}$$

where κ<sup>0</sup> = λ<sup>0</sup> ξ0 is the Ginzburg-Landau parameter of the reference material. The quantity λ<sup>0</sup> = q msc <sup>2</sup>β<sup>0</sup> 4πe<sup>2</sup> s |α0| is the penetration depth of the reference material, and ξ<sup>0</sup> = q <sup>ℏ</sup><sup>2</sup> 2ms|α0| is its coherence length. The parameter u<sup>0</sup> = τψ<sup>0</sup> τj<sup>0</sup> is a ratio of characteristic time scales in the reference material, where τψ<sup>0</sup> = Γ<sup>0</sup> |α0| is the characteristic relaxation time of ψ in the reference material and τj<sup>0</sup> = σn0msβ<sup>0</sup> e 2 s |α0| is the characteristic relaxation time of the current. We have also inserted a minus in front of a, which is just a convention to make positive values of a correspond to the superconducting state (Note: this is the opposite of how α is usually interpreted within Ginzburg-Landau theory, however it is standard to make this change when performing nondimensionalization). The boundary conditions become:

$$\left(\frac{i}{\kappa\_0}\nabla\psi + \mathbf{A}\psi\right)\cdot n = 0\tag{15}$$

$$(\nabla \times \mathbf{A}) \times n = \mathbf{H}\_a \times n \tag{16}$$

$$\left(\nabla\phi + \frac{\partial\mathbf{A}}{\partial t}\right) \cdot n = 0. \tag{17}$$

It should be noted that γ(r) and s(r) allow the local characteristic time relaxations to vary, which only affects the dynamics of the solutions. Where the time dynamics are relevant these parameters cannot be ignored; however, in many cases, the primary interest of TDGL calculations is in determining the energetic stability and critical fields, such as the superheating field. In these cases, the time dynamics are not relevant and γ and s can be safely set <span id="page-3-1"></span>to arbitrary constant values, such as unity.

## <span id="page-3-0"></span>B. Determining Spatial Variation of TDGL Parameters

In the previous section, we have shown how to introduce spatial variation to the TDGL parameters. The process of calculating the values of a(r), b(r), and γ(r) is where we bring in experiment and ab initio theory. From Eqs. [6-](#page-2-4)[8,](#page-2-5) we know these parameters depend on well-defined microscopic quantities, namely ν(0) and Tc. These quantities can be calculated using densityfunctional theory (DFT). Local densities of states are straightforward to compute in DFT, providing local values for ν(0) and superconducting quantities such as T<sup>c</sup> are calculated by applying Eliashberg theory within a DFT framework and directly calculating electron-phonon coupling from first principles. Experiment can then give information about the compositions of sample materials, and DFT calculations can determine the ν(0) and T<sup>c</sup> associated with these compositions. Using these values, a(r), b(r), and γ(r) are calculated from Eqs. [6-](#page-2-4)[8,](#page-2-5) and the material geometries from the experimental results determine the spatial variation.

In this paper, we will demonstrate our framework on Sn-deficient islands. These defects have been studied extensively [\[35,](#page-13-9) [36,](#page-13-13) [40,](#page-13-14) [41\]](#page-13-15) so it is straightforward to find estimates of the general size of these islands and their material compositions in the literature. DFT has also been used to calculate ν(0) and T<sup>c</sup> for Nb-Sn systems of different concentrations. In this paper we use a commonly observed Sn concentration of 18% and take the DFT values from Ref. [46](#page-13-20) to construct a(r), b(r), and γ(r). This paper will primarily focus on the results of our TDGL calculations, for further information regarding the details and results from the DFT calculations used in this paper, we refer the interested reader to Refs. [45](#page-13-19) and [46.](#page-13-20)

### C. Finite Element Formulation

In this paper we solve the TDGL equations in 3D via a finite element method proposed by Gao [\[57\]](#page-14-6). We choose the popular temporal gauge which sets the scalar potential ϕ = 0 (see Du [\[58\]](#page-14-7) for a more detailed overview of gauge choices for TDGL). Taking the curl of Eq. [14](#page-3-0) allows us to directly solve for the magnetic field and we get a system of 3 equations for ψ, A, and H:

$$\frac{\partial \psi}{\partial t} + \left(\frac{i}{\kappa\_0} \nabla + \mathbf{A}\right)^2 \psi - a\psi + b|\psi|^2 \psi = 0\tag{18}$$

$$\frac{1}{u\_0}\frac{\partial \mathbf{A}}{\partial t} + \nabla \times \mathbf{H} + \frac{i}{2\kappa\_0} \left( \psi^\* \nabla \psi - \psi \nabla \psi^\* \right) + |\psi|^2 \mathbf{A} = 0\tag{19}$$

$$\frac{1}{u\_0}\frac{\partial \mathbf{H}}{\partial t} + \nabla \times \nabla \times \mathbf{H} + \frac{i}{\kappa\_0} \nabla \psi^\* \times \nabla \psi + |\psi|^2 \mathbf{H} + \nabla |\psi|^2 \times \mathbf{A} = 0. \tag{20}$$

These equations are subject to the boundary conditions:

$$\left(\frac{i}{\kappa\_0}\nabla\psi + \mathbf{A}\psi\right)\cdot\mathbf{n} = 0\tag{21}$$

$$\mathbf{H} \times \mathbf{n} = \mathbf{H}\_a \times \mathbf{n},\tag{22}$$

where n is still the outward normal unit vector and H<sup>a</sup> is the applied field. We are then able to solve the system by discretizing the test and trial functions with Lagrange finite elements, and using a backward difference scheme for the time derivatives. All computations in this paper were done using the open source software FEniCS [\[59\]](#page-14-8).

#### D. Dissipation in TDGL

When simulating SRF materials, dissipation is often a physical quantity of interest. Under TDGL, a formula for dissipation can be derived by considering the time derivative of the free energy and the free energy current flux density. A more detailed derivation is found in Ref. [56,](#page-14-5) but we quote the final result here:

<span id="page-4-0"></span>
$$D = 2\Gamma \left| \left( \frac{\partial \psi}{\partial t} + \frac{ie\_s \phi \psi}{\hbar} \right) \right|^2 + \sigma\_n \mathbf{E}^2. \tag{23}$$

This quantity is a power density, with the first term corresponding to the superconducting dissipation arising from the relaxation of the order parameter. The second term is the dissipation of normal currents which are largest near the surface where magnetic field can still appreciably penetrate.

It is worth considering how this expression for the dissipation density is related to existing theories of RF power loss and surface resistance. The first term in Eq. [23](#page-4-0) is associated with the dissipation due to the rate of change of ψ. This term is typically small, except in the vortex state where it becomes the dominant source of dissipation. A dissipation of this form is similar to that proposed by Tinkham [\[60\]](#page-14-9), who suggested the vortex dissipation should be proportional to ∂ψ ∂t <sup>2</sup> . The additional term within the parenthesis in Eq. [23](#page-4-0) is a result of the gauge invariance of TDGL.

The second term in Eq. [23](#page-4-0) can be directly related to the phenomenological "two-fluid model," which was first proposed by Gorter and Casimir [\[61\]](#page-14-10) in 1934 and was applied by London [\[62\]](#page-14-11) later that year to calculate the power loss of a superconductor. The two-fluid model approximates the electrons of the system as consisting of two non-interacting 'fluids': the superconducting electrons, in the form of cooper pairs which carry lossless supercurrent, and the normal electrons, which exist as thermally excited quasiparticles that produce typical dissipative currents. Under the two-fluid model, the normal fluid losses produce dissipation of the form [\[63,](#page-14-12) [64\]](#page-14-13)

$$P \propto \sigma\_n E^2,$$

which is identical to the second term of Eq. [23.](#page-4-0) For AC applied currents, the electric field is proportional to the frequency and magnetic field, E ∝ ωH, meaning that overall the power loss will be of the form

<span id="page-4-1"></span>
$$P \sim \omega^2 H^2. \tag{24}$$

It is also well known that within RF cavities, the power loss is given by

<span id="page-4-2"></span>
$$P = \frac{1}{2} \oint\_{A} R\_s |H(r)|^2 dA,\tag{25}$$

where R<sup>s</sup> is the surface resistance of the cavity walls and A is the surface area. Comparing Eqs. [24](#page-4-1) and [25,](#page-4-2) we have that the second term of Eq. [23](#page-4-0) directly leads to a surface resistance proportional to the square of the frequency, R<sup>s</sup> ∝ ω 2 . In the late 1950s, Mattis and Bardeen [\[65\]](#page-14-14) as well as Abrikosov, Gor'Kov, and Khalatnikov [\[66\]](#page-14-15) independently derived the now well-known "BCS resistance." Under the low frequency and low field limit, the BCS resistance reduces to the form [\[67,](#page-14-16) [68\]](#page-14-17)

<span id="page-5-4"></span>
$$R\_{bcs} \simeq \frac{\omega^2}{T} e^{\frac{-\Delta}{k\_B T}}\tag{26}$$

where ∆ = 1.76kBT<sup>c</sup> is the superconducting energy gap [\[56\]](#page-14-5). We thus see that our calculated expression for the surface resistance is in good agreement with the ω <sup>2</sup> dependence of the BCS prediction in the low frequency and field limit.

#### E. Estimating Cavity Quality Factor

One of the most important quantities that can be directly estimated from the dissipation is the cavity quality factor, Q. Quality factor is given by

<span id="page-5-2"></span>
$$Q = \frac{2\pi E}{\Delta E},\tag{27}$$

where E is the energy stored in the cavity and ∆E is the energy dissipated in the cavity walls each RF period. These quantities (working in SI units for this section) can be expressed as integrals:

$$E = \frac{1}{2}\mu\_0 \int\_V dV \mathbf{H}^2\tag{28}$$

$$
\Delta E = \int\_0^T dt \int\_{V\_{surf}} dV\_{surf} D,\tag{29}
$$

where V is the cavity volume, T is the RF period, and D is given by Equation [23.](#page-4-0) Vsurf is the volume in the first few penetration depths of the cavity surface where essentially all of the dissipation occurs. TDGL simulation outputs are unit-free, so it is helpful to pull constants with units out of these integrals, leaving behind dimensionless functions which can be calculated from TDGL solutions. We start by expressing Equation [28](#page-5-0) in cylindrical coordinates:

$$E = \frac{1}{2}\mu\_0 \int rdr \int d\phi \int dz \mathbf{H}^2$$

We then define some dimensionless quantities:

$$
\tilde{r} = \frac{r}{R} \tag{30}
$$

$$
\tilde{z} = \frac{z^{\tilde{l}}}{L} \tag{31}
$$

$$
\hat{\mathbf{H}} = \frac{\mathbf{H}}{H\_a} \tag{32}
$$

Where R is the maximum radius of the cavity, L is the length of the cavity in the axial direction, and H<sup>a</sup> is the maximum value of the applied field at the surface of the cavity during an RF period. These quantities allow the definition of a unit-less integral that only depends on the cavity geometry:

$$I\_H \equiv \int \tilde{r} d\tilde{r} \int d\tilde{z} \tilde{\mathbf{H}}^2 \tag{33}$$

Using these definitions with Equation [28](#page-5-0) and assuming that H has azimuthal symmetry results in

<span id="page-5-3"></span>
$$E = \pi \mu\_0 H\_a^2 L R^2 I\_H. \tag{34}$$

Turning to the dissipated energy integral, suppose all of the dissipation occurs within a distance d below the cavity surface, where d << R. This allows the cylindrical integral to be converted into cartesian coordinates, with the azimuthal direction becoming the new x direction, the axial direction becoming the new y direction, and the radial direction becoming the new z direction. Doing this, we have

$$
\Delta E = \int\_0^T dt \int\_0^{2\pi R} dx \int\_0^L dy \int\_0^d dz D. \tag{35}
$$

When calculating this from simulation outputs, the integral is necessarily calculated over a small region of the overall cavity surface. Let L<sup>x</sup> and L<sup>y</sup> be the simulation domain size in the x and y directions respectively, and let N be the total number of simulation areas needed to fully partition the cavity surface. Then the dissipation integral becomes

$$
\Delta E = N \int\_0^T dt \int\_0^{L\_x} dx \int\_0^{L\_y} dy \int\_0^d dz D,\tag{36}
$$

<span id="page-5-0"></span>and N can be approximated as

$$N = \frac{2\pi RL}{L\_x L\_y}.\tag{37}$$

Contuining as before we again define dimensionless coordinates:

$$
\tilde{x} = \frac{x}{\lambda} \tag{38}
$$

$$
\tilde{y} = \frac{y}{\lambda} \tag{39}
$$

$$
\tilde{z} = \frac{z}{\lambda} \tag{40}
$$

$$
\tilde{t} = \frac{T\_{sim}}{T}t \tag{41}
$$

where λ is the penetration depth and Tsim is the period in units of simulation time. These convert the integral to

$$
\Delta E = \lambda^3 \frac{T}{T\_{sim}} N \int\_0^{T\_{sim}} d\bar{t} \int\_0^{\frac{L\_x}{\lambda}} d\bar{x} \int\_0^{\frac{L\_y}{\lambda}} d\bar{y} \int\_0^{\frac{\bar{d}}{\lambda}} d\bar{z} D. \tag{42}
$$

Additionally, under the temporal gauge (ϕ = 0), Equation [23](#page-4-0) can be expressed as

<span id="page-5-1"></span>
$$D = 2\mu\_0 H\_c^2 \frac{T\_{sim}}{T} \left( \left| \frac{\partial \bar{\psi}}{\partial \bar{t}} \right|^2 + \sigma\_n \mu\_0 \lambda^2 \frac{T\_{sim}}{T} \left( \frac{\partial \bar{\mathbf{A}}}{\partial \bar{t}} \right)^2 \right), \tag{43}$$

where ψ˜ and A˜ are the unit-free versions of the vector potential and order parameter that are solved for with Eqs. [13](#page-3-1) and [14](#page-3-0) (a derivation of Equation [43](#page-5-1) can be found in the Appendix). Finally, we define some more dimensionless integrals over the TDGL solutions:

$$I\_{\psi} \equiv \int\_{0}^{T\_{sim}} d\tilde{t} \int\_{0}^{\frac{L\_{x}}{\lambda}} d\tilde{x} \int\_{0}^{\frac{L\_{y}}{\lambda}} d\tilde{y} \int\_{0}^{\frac{d}{\lambda}} d\tilde{z} \left| \frac{\partial \tilde{\psi}}{\partial \tau} \right|^{2} \tag{44}$$

$$I\_A \equiv \int\_0^{T\_{sim}} d\tilde{t} \int\_0^{\frac{L\_x}{\lambda}} d\tilde{x} \int\_0^{\frac{L\_y}{\lambda}} d\tilde{y} \int\_0^{\frac{d}{\lambda}} d\tilde{z} \left(\frac{\partial \tilde{\mathbf{A}}}{\partial \tau}\right)^2 \quad (45)$$

Combining everything and noting that ω = 2π T , we get

<span id="page-6-1"></span>
$$\Delta E = 2\mu\_0 H\_c^2 \lambda^3 \frac{2\pi RL}{L\_x L\_y} \left( I\_\psi + \omega \frac{\sigma\_n \mu\_0 \lambda^2 T\_{sim}}{2\pi} I\_A \right) . \tag{46}$$

Now using Equations [27,](#page-5-2) [34,](#page-5-3) and [46](#page-6-1) we get an expression for the quality factor,

$$Q = \frac{\tilde{H}\_a^2 R L\_x L\_y I\_H}{2\lambda^3 \left(I\_\psi + \omega \frac{\sigma\_n \mu\_0 \lambda^2 T\_{x\text{im}}}{2\pi} I\_A\right)},\tag{47}$$

where H˜ <sup>a</sup> ≡ <sup>√</sup> H<sup>a</sup> 2H<sup>c</sup> is the applied field in simulation units. It is common to express the quality factor as

<span id="page-6-5"></span>
$$Q = \frac{G}{R\_s},\tag{48}$$

where R<sup>s</sup> is the cavity surface resistance and G is a geometric factor that depends only on quantities which are determined by the cavity geometry. We can define these quantities under the framework we have presented as

$$G = \frac{1}{2}\mu\_0\omega R I\_H \tag{49}$$

$$R\_s = \frac{\mu\_0 \omega \lambda^3}{\tilde{H}\_a^2 L\_x L\_y} \left( I\_\psi + \omega \frac{\sigma\_n \mu\_0 \lambda^2 T\_{sim}}{2\pi} I\_A \right). \tag{50}$$

For a typical 1.3 GHz 9-cell Nb TESLA cavity, G = 270 Ω [\[69\]](#page-14-18), so in practice we can just use this value or other known values of G, and only calculate R<sup>s</sup> from Equation [50.](#page-6-2)

The σ<sup>n</sup> in the above equations is the conductivity of only the normal electrons. A simple way to estimate this value is to consider the Drude model conductivity [\[70\]](#page-14-19),

<span id="page-6-3"></span>
$$
\sigma = \frac{ne^2\tau}{m},
\tag{51}
$$

where e and m are the electron charge and mass, n is the number density of electrons, and τ is their mean free collision time. In the superconducting state, we can make the usual assumption from the two-fluid model that n = n<sup>s</sup> +n<sup>n</sup> where n<sup>s</sup> and n<sup>n</sup> are the number densities of the superconducting and normal electrons respectively. Under TDGL, n<sup>s</sup> = α β |ψ˜| <sup>2</sup> and so the normal conductivity is

$$n\_n = n - \frac{\alpha}{\beta} |\tilde{\psi}|^2. \tag{52}$$

Using this along with Equation [51,](#page-6-3) the normal electron conductivity in the Meissner state can be estimated with

$$
\sigma\_n = \frac{ne^2\tau}{m} \left( 1 - \frac{1}{n} \frac{\alpha}{\beta} |\tilde{\psi}|^2 \right). \tag{53}
$$

In the above calculations, it was assumed that the cavity surface is partitioned into small fractional areas, and the dissipated energy is calculated over one of these areas, and then multiplied by the total number of them. If some defect was present in the simulation domain, it would mean that a Q calculation based purely on that value would be assuming that such a defect is uniformly distributed over the surface of the domain. In Equation [50,](#page-6-2) the only thing that changes when simulating a different material or different defect is the value of the quantity in parenthesis, every other part of the process for calculating Q remains the same. This means that we can calculate a more realistic cavity surface resistance by calculating the R<sup>s</sup> with Equation [50](#page-6-2) for a few different simulations, and then taking a weighted average of these values. Let R<sup>i</sup> be the surface resistance of the ith simulation, and let p<sup>i</sup> be the percentage of the fractional areas partitioning the cavity surface which are represented by this simulation. Then

<span id="page-6-4"></span>
$$R\_{tot} = \sum\_{i} p\_i R\_i,\tag{54}$$

where P i p<sup>i</sup> = 1. The simplest application for Equation [54](#page-6-4) is to perform two simulations, one baseline simulation with no defects or material inhomogeneity, and then another simulation containing some defect of interest. Choosing a value p to represent the percentage of fractional areas containing the defect and then following through with the rest of the quality factor calculation provides a simple way to estimate Q for different average defect densities by simply changing the value of p.

<span id="page-6-2"></span>The value of Q calculated from TDGL outputs will typically be underestimated at low field. This is because of the assumption of gapless superconductivity, which results in higher surface resistances than is predicted with the BCS surface resistance (Equation [26\)](#page-5-4). Despite this, our approach still often predicts quality factors within an order of magnitude of the experimental values. Additionally, the relative behavior of Q at different applied fields, especially when averaging the impact of multiple kinds of defects as described above, qualitatively captures effects such as high field Q-slope.

#### <span id="page-6-0"></span>III. VALIDATION STUDY

We now the demonstrate the use of the methods from Section [II.](#page-2-0) The geometry used within the simulations in this section is depicted in Fig. [1.](#page-7-0) The domains used are rectangular cuboids, with periodic boundary conditions in the x and y directions, denoted by the yellow and light blue highlights, respectively. The external field is applied

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

**Caption:** Schematic of the Simulation Geometry: The simulation geometry for the study is depicted as a cubic domain with side length X. Like-colored surfaces are assigned periodic boundary conditions while different fields are applied at red and green surfaces in the Z-direction. The illustration of the Sn-deficient island as a blue ellipsoid, with projections marked by dotted lines, indicates how the island's placement parallel to the surfaces impacts field interactions. The overarching simulation goal is to model the superconducting response and vortex behavior through changes in field application, defined by various domain boundary conditions .


FIG. 1: A Schematic of our Simulation Geomtery. Like-colored surfaces (The surfaces outlined in yellow and light blue) have periodic boundary conditions. The red and green outlined surfaces are where field is applied, which can be different on each surface. Often, the red surface is where we will apply the field, with no field applied on the green surface. The Sn-deficient island is depicted as a dark blue ellipsoid, with dotted lines showing the projection of the ellipsoid into the XZ, XY, and YZ planes, the color of the ellipse corresponds to the face it is projected into. The island's distance from the surface is measured from the surface to the outer edge of the ellipsoid, denoted by d.

on the upper and lower surfaces in the z direction, highlighted in red and green in the figure. For the simulations in this paper, we apply the field just on the red surface, with no field applied to the green surface. For simulations including Sn-deficient islands, such an island is also depicted in Fig. [1](#page-7-0) (the dotted ellipses show the projection of the ellipsoid into each surface). For the Sn-deficient island simulations in this paper, we choose the ellipsoids to have equal x and y radii and a z radius of half the x radius. The island's distance from the surface, denoted on the figure by d, is measured from the surface to the outer edge of the ellipsoid. All of our simulations were solved using cubic meshes with tetrahedral elements, all generated using the open source mesh generation program, Gmsh [\[71\]](#page-14-20). For the Sn-deficient island simulations, the OpenCASCADE geometry kernel available within Gmsh was used to make the mesh conform to the shape of the islands.

<span id="page-7-1"></span>![](_page_7_Figure_4.jpeg)

**Caption:** Quality Factor of Nb SRF Cavities Versus Applied Field: The quality factor of Nb superconducting RF (SRF) cavities is plotted against the applied field (normalized to √2Hc), calculated using geometric factor G = 270 Ω. The sharp quality drop near the superheating threshold results from vortex nucleation, highlighted by the black dotted line indicating the superheating field, providing insights into the limitations imposed by superconducting properties on cavity performance .


FIG. 2: Nb SRF Cavity Quality Factor vs. Applied Field. A plot of quality factor for a uniform Nb simulation versus the applied field (in units of √ 2Hc). The quality factor is calculated using Eqs. [48](#page-6-5) and [50](#page-6-2) with G = 270 Ω. The dotted black line indicates the superheating field of Nb. Because it is above Hsh, vortex nucleation occurred for H˜ <sup>a</sup> = 1.0 which is the

# reason for the sharp drop in Q.

### A. Nb SRF Cavity Quality Factors

We start by demonstrating the calculation of quality factor using Eqs. [48](#page-6-5) and [50.](#page-6-2) We simulate a 1.3 GHz Nb cavity (G = 270 Ω), which typically have quality factors in the range of 10<sup>10</sup> − 1011. Our simulation domain for these simulations is a cube with side length 20λ and we use a sinusoidal applied field with a period of 2000τψ. Since τ<sup>ψ</sup> = Γ α , we can use Eqs. [6](#page-2-4) and [8,](#page-2-5) and ν(0) = 90 states/(eV nm<sup>3</sup> ) (value obtain from private correspondence) to get that for Nb at 2K, τ<sup>ψ</sup> = 3.7 × 10<sup>−</sup><sup>13</sup> s. As a result, a period of 2000τ<sup>ψ</sup> corresponds to a frequency of ∼1.3 GHz. Our results for this simulation are shown in Figure [2.](#page-7-1) We see that the low field quality factor is ∼2×10<sup>9</sup> , this is only a little below the expected range of 10<sup>10</sup>−10<sup>11</sup>, which can be easily explained by the assumption of gapless superconductivity under TDGL, which causes Q to be underestimated. There is a sharp drop in Q once the applied field crosses the superheating field of niobium (the black dotted line) and vortices start to nucleate.

We can also plot the unitless dissipation values that are used to calculate Q. Figure [3](#page-8-0) depicts the dissipation values for the two terms in parentheses in Equation [46.](#page-6-1) We find that, as expected, the term due to the dissipation of the electric field ( <sup>σ</sup>nµ0<sup>λ</sup> 2 τ<sup>ψ</sup> IA) is larger than the term due to order parameter relaxation (Iψ) by about two orders of magnitude. The I<sup>ψ</sup> term grows faster than the I<sup>A</sup> term and the two are equal at an applied field of <sup>√</sup> H<sup>a</sup> 2H<sup>c</sup> =
![](0__page_8_Figure_0.jpeg)

**Caption:** Dissipation Values Analysis: Plots unitless dissipation values relevant to calculating superconducting cavity quality factors, distinguishing key contributions from electronic versus order parameter dynamics. The rapid increase in dissipation above superheating fields aligns with the stress on superconducting stability under operational conditions .


![](0__page_8_Figure_1.jpeg)

**Caption:** Comparison of Different Dissipation Terms: This plot compares unitless dissipation between terms representing electric field dissipation and order parameter relaxation under various applied fields. It shows the electric field dissipation dominates initially, but with increasing applied field, vortex-induced changes reverse this dominance as visualized in simulations .


0.6, and at higher fields, the I<sup>ψ</sup> term remains dominant, particularly above the superheating field where it grows much faster due to vortex nucleation and motion. These results demonstrate how the methods in this paper can be used to calculate the dissipation for ideal cases like uniform niobium. We will use them again for the case of Sn-deficient islands in Section [III B.](#page-8-0)

## <span id="page-8-0"></span>B. Sn-deficient Islands in Nb3Sn

There has been a considerable amount of work both experimentally and theoretically understanding the various defects present in Nb3Sn [\[19,](#page-12-0) [34–](#page-13-0)[41\]](#page-13-1). For the purposes of this paper, we will focus on Sn-deficient islands, small regions within Nb3Sn grains that have lower than expected Sn concentrations. To demonstrate the use of our sample specific TDGL framework, we will estimate the potential impact to SRF performance by determining the vortex penetration field, Hvort. This field is the lowest field at which a vortex nucleates, it is a generalization of the superheating field for surfaces containing defects, and it is equal to the superheating field in the limit of a completely uniform flat surface.

In order to simulate Sn-deficient islands, we need estimates of the properties of the material within these regions. We choose a Sn concentration of 18 at.% for our islands. Using this value as well as DFT calculations

<span id="page-8-1"></span>

| TDGL Parameter and Material Values |        |        |
|------------------------------------|--------|--------|
| Quantity                           | Bulk   | Island |
|                                    | Value  | Value  |
| Tc<br>(K)                          | 18     | 6      |
| ν(0)(states/(eV*nm3<br>))          | 101.33 | 40.53  |
| a (unitless)                       | 1      | 0.15   |
| b (unitless)                       | 1      | 1.8    |
| γ (unitless)                       | 1      | 1.2    |
| s (unitless)                       | 1      | 1      |

TABLE I: A summary of the TDGL parameter and material values used in the results of this paper. All TDGL parameters were calculated from

Equations [6](#page-2-0)[-8](#page-2-1) using the higher order temperature dependence and normalized with respect to the bulk value. The T<sup>c</sup> and ν(0) values were taken from Ref. [46](#page-13-2) assuming a Sn concentration of 18% for the island. The normal electron conductivity, s, was assumed to remain constant throughout the simulation domain.

for Nb3Sn from Sitaraman et al. [\[46\]](#page-13-2), we estimate the critical temperature and density of states. Using these values, we construct a(r), b(r), and γ(r); the values of these quantities in the bulk as well as in the island are reported in Table [I.](#page-8-1) The simulation geometry is the same as depicted in Fig. [1,](#page-7-0) with the domain a cube with side length 10λ. The applied field for these simulations was a constant value.

Figure [4](#page-9-0) depicts vortex nucleation occuring at Hvort for a particular island. We find that varying the distance of the island from the surface between 0.1 and 2 penetration depths (i.e. between ∼10-200 nm for Nb3Sn), we observe a reduction in Hvort by as much as ∼70% down from the bulk value of the superheating field for islands very near the surface, as shown in Fig. [5.](#page-9-1) We did this for several different island sizes, and found that the severity of this drop in Hvort increases with increasing island size. As such, we can conclude that large islands within 200 nm of the cavity surface are a potential cause for concern when it comes to SRF performance, particularly ones very close to the surface.

Using these solutions, we can also calculate the dissipation due to the Sn-deficient islands, and estimate their quality factors. To do so, we change our applied field to be sinusoidal with a period of 1000τ<sup>ψ</sup><sup>0</sup> , which corresponds to a frequency on the order of ∼5 GHz. Most Nb3Sn cavities use a frequency lower than this, but these simulations already had immense computational cost, so using a 5× longer period would make them prohibitively expensive. When running TDGL simulations, to get good solution convergence, the mesh elements must be smaller than a coherence length (and ideally much smaller). This means simulating Nb3Sn is extremely challenging, as it has κ = ∼26, meaning the domain both has to be large enough to capture at least a few penetration depths, while also being refined enough to have elements that are a few times smaller than a coherence length. When determining Hvort, the mesh size (and therefore computational

<span id="page-9-0"></span>![](0__page_9_Figure_1.jpeg)

**Caption:** 3D Visualization of |ψ|² During Vortex Nucleation: This volume plot showcases |ψ|², representing superconducting order during a vortex nucleation event within a cubic domain, sized 10λ on each side. The occurrence highlights foundational steps in understanding vortex dynamics, specifically with an applied field equivalent to Hvort for the given Sn-deficient island and placement .


(a) A plot of |ψ| <sup>2</sup> during a vortex nucleation event.

![](0__page_9_Figure_3.jpeg)

**Caption:** 2D Slice through Vortex Nucleation: This figure gives a cross-sectional view of |ψ|² during vortex nucleation within a defined plane of the superconducting domain. It aids in understanding depth-dependent vortex dynamics associated with Sn-deficient islands .


(b) A 2D slice of |ψ| 2 in the XZ plane at <sup>y</sup> <sup>λ</sup> = 5

![](0__page_9_Figure_5.jpeg)

**Caption:** Isosurface Visualization of |ψ|² = 0.1: Demonstrates the spatial configuration of a vortex isosurface within the superconducting domain, combining visualization capabilities with specific contouring to show induced vortex nucleation around the Sn-deficient island with defined field conditions .


(c) The |ψ| <sup>2</sup> = 0.1 isosurface.

FIG. 4: Example of a Sn-deficient Island Simulation. The domain is a cube with side length 10λ, this particular simulation is for an island 0.5λ from the surface, with a X and Y radius of 2λ and a Z radius of 1λ. The applied field is <sup>√</sup> H<sup>a</sup> 2H<sup>c</sup> = 0.38, which is the value of Hvort for this island size and distance from the surface. (a) depicts a volume plot of |ψ| <sup>2</sup> over the whole domain during a vortex nucleation event. (b) is a 2D slice in the XZ plane at <sup>y</sup> <sup>λ</sup> = 5. (c) is the |ψ| <sup>2</sup> = 0.1 isosurface, which shows the shape of the vortex as well as the island that induced the nucleation.

<span id="page-9-1"></span>![](0__page_9_Figure_8.jpeg)

**Caption:** Vortex Penetration Versus Island Distance: This graph displays the vortex penetration field (Hvort) as a function of island position relative to the superconducting surface, measured in units of penetration depth. The figure compares results from three island sizes, with the black dotted line indicating the bulk Nb3Sn superheating field. It reveals that islands closer to the surface substantially reduce the Hvort, emphasizing the effect of island proximity on superconducting performance .


FIG. 5: Vortex Penetration Field Versus Distance from Surface. We plot our calculated vortex penetration field estimates, varying the distance of the Sn deficient island from the superconductor surface. The field values are reported in units of <sup>√</sup> 2H<sup>c</sup> and the distances are in units of penetration depths. We did this for 3 different islands, with x radii of 0.5, 1, and 2 penetration depths. The black dotted line denotes the superheating field of bulk Nb3Sn. Smaller volume islands have a diminishing impact on Hvort.

cost) can be reduced by only significantly refining the mesh right near the surface, where vortices will nucleate. Once a vortex begins to nucleate, we know that the applied field is above Hvort and thus the rest of that simulation is no longer relevant. However, calculating dissipation requires enough resolution throughout the mesh for a vortex to be able to nucleate, and then move around to dissipate energy, and with a sinusoidal applied field, the vortices can partially nucleate, and then get pushed back out of the superconductor as the field changes. The result of this is that simulations to estimate the dissipation and quality factor of Sn-deficient islands are much more computationally expensive than estimating Hvort, and so we will only be simulating the smallest island size at a distance from the surface of 0.5λ and we decreased our domain size to a 5λ × 2.5λ × 2.5λ rectangular cuboid with the long direction oriented in the same direction as the applied field (and thus the initial orientation of any vortices that nucleate).

Figure [6](#page-10-0) shows the results for our quality factors. We ran two simulations, a baseline simulation with no island and a Sn-deficient island with the size described above. Then using Equation [54,](#page-6-1) we estimated the effective total cavity surface resistance for a number of different island densities. The percentage of island coverage in this figure is calculated by determining the fraction of the total simulation surface area occupied by the island projected into the XY plane and multiplying it by the weighting on the surface resistance from the island simulations used in Equation [54.](#page-6-1) the 0% island coverage curve (in blue) is the quality factor calculated only from the simulation without an island, and the 6.3% curve (in pink) is the quality factor calculated only from the simulation with an island. We see that above Hvort for this island size and distance from the surface vortices begin nucleating, with basically all of the deviations between different island coverages happening only above this value. For cavities with ≥ 1.5% of their surface containing an island, the drop in quality factor starts to become very significant, with even small increases in island density leading to significant drops in quality factor. The noise in the calculations here is largely due to numerical variations in how many vortices enter, and how much they move around. The temperature of the simulation is held constant at the Nb3Sn operating temperature of 4.2K, so it is likely that these vortices would realistically lead to heating and subsequent cavity quench once the field is much higher than Hvort. The low field magnitude of the quality factor in these simulations is ∼10<sup>8</sup> . This is lower than the expected value of ∼1010, partly because of being at a higher frequency than 1.3 GHz, and also because the embedded assumption of gapless superconductivity in TDGL means that it will typically underestimate Q (as discussed in Section [II E\)](#page-5-0). Given infinite computational time (or a lower κ material), this kind of calculation could be repeated for a variety of island sizes and distances, and applying the defect averaging approach described in Section [II E,](#page-5-0) the resulting Q plots would look increasingly like the high field Q slope plots taken experimentally from real cavities.

## IV. CONCLUSION

In this paper we have presented a method which allows time-dependent Ginzburg-Landau to serve as a mesoscopic link between microscopic experimental measurements and ab initio calculations with macroscopic SRF performance metrics, such as quality factor and quench fields. By allowing the parameters of TDGL to vary in space, and writing them in terms of well defined local quantities like the Fermi-level density of states and the critical temperature, material profiles can be constructed which allow TDGL to accurately simulate the superconducting and electromagnetic response of realistic material samples. We used this method to model Sn-deficient islands and calculated that there could be up to a 70% reduction in Hsh for islands which are close to the cavity surface. The theoretical maximum accelerating gradient for Nb3Sn cavities is around 100 MV/m, but the highest accelerating field achieved so far by a Nb3Sn SRF cavity is around 24 MV/m [\[16\]](#page-12-1), a ∼75% decrease. Since SRF cavity quench fields are directly related to Hsh, our predicted 70% reduction in Hvort is strong evidence that near-surface (within ∼10-200 nm) Sn-deficient islands may be one of the contributing factors to poor cavity performance. We further support this evidence

<span id="page-10-0"></span>![](0__page_10_Figure_4.jpeg)

**Caption:** Quality Factor for Sn-deficient Island Coverage: Illustrates the dramatic quality factor drop as Sn-deficient islands cover larger areas of cavity surfaces, under applied fields exceeding Hvort. The figure aligns simulation outcomes with experimental observations, providing a comprehensive view of material deficiencies affecting SRF cavity performance.


FIG. 6: Quality Factor for a Sn-deficient Island. Quality factor plots using Equation [54](#page-6-1) to estimate different densities of Sn-deficient islands by averaging results from simulations both with and without an island. The ellipsoid island has X and Y radii of 0.5λ and a Z radius of 0.25λ. The top edge of the island is 0.5λ below the upper surface. Vortices begin nucleating for the island simulation at an applied field of 0.525.

The percentage of island coverage was calculated by taking the fraction of the total simulation surface area occupied by the island (which for these simulations was ∼6.3%) and multiplying it by the weighting on the surface resistance from the island simulations used in Equation [54.](#page-6-1) So the 0% island coverage curve (in blue) is the quality factor calculated only from the simulation without an island, and the 6.3% curve (in pink) is the quality factor calculated only from the simulation with an island.

with calculations of the quality factor due to Sn-deficient islands, finding significant drops in quality factor above Hvort, even for relatively small fractions of the cavity surface area containing islands, and this effect would likely be even more significant in potential future simulatation performating the same analysis for larger islands closer to the surface (or even fully exposed) and coupling the dissipation to a temperature field to allow for thermal runaway effects which could simulate cavity quenching.

A recent study performing centrifugal barrel polishing (CBP) on Nb3Sn SRF cavities to smooth the RF surface recently found that doing so led to a decrease in the overall quality factor, as well as lower field for the onset of Q-slope and a lower quench field. Performing another Sn vapor deposition recovered the previous quality factor performance, but with a higher quench field [\[72\]](#page-14-0). Our results provide strong evidence for the origin of these effects, described as follows: CBP smooths the RF surface, but results in the removal of surface material, exposing previously deep Sn-deficient islands. These near-surface islands lead to increased dissipation, lowering the overall quality factor. Vortex nucleation induced by the islands at higher fields leads to onset of Q-slope and eventually an early cavity quench. Performing an additional Sn deposition removes the Sn deficiencies to recover the original quality factor, but now reaching higher quench fields due to the smoother surface.

Whenever using the TDGL equations to make predictions about experimental systems, it is important to consider their limitations. They are formally valid for gapless superconductivity, as gapped superconductors have a singularity in their density of states which prevents expansions in powers of the energy gap [\[73\]](#page-14-1). Additionally, the equations are only quantitatively valid near the critical temperature. The first of these conditions can be avoided with the use of a generalized version of TDGL first proposed by Kramer and Watts-Tobin [\[74\]](#page-14-2). Future work could include the use of this generalized TDGL; however, it should be noted that Proslier et al. [\[75\]](#page-14-3) observed a broadening in the density of states due to Nb oxides leading to a gapless superconducting surface layer in certain Nb SRF cavities. Gurevich and Kubo [\[68,](#page-14-4) [76\]](#page-14-5) later showed that under typical SRF operating conditions and material compositions, there is a generic broadening of the density of states and lowering of the gap, which further justifies the use of TDGL for SRF applications.

The methods for simulating realistic sample-specific material defects and calculating the resulting quality factors presented in this paper represents a promising tool for understanding the impact of many different underlying material features on SRF performance. Many of these features are impossible to probe experimentally, so this method will be useful in assisting the design and manufacturing of the next generation of SRF cavities.

## V. ACKNOWLEDGEMENT

This work was supported by the US National Science Foundation under Award OIA-1549132, the Center for Bright Beams. We would like to thank Dr. Nathan Sitaraman for helpful conversations and estimates of the Fermi level density of states for Nb, and Dr. Michelle Kelley for helpful comments about early drafts of the methods section.

Appendix: TDGL Non-dimensionalization

To nondimensionalize the TDGL equations, we start with Eqs. [1](#page-2-2) and [2,](#page-2-3) and make the following coordinate transformations:

$$\begin{aligned} \nabla &\longrightarrow \frac{1}{\lambda\_0} \ddot{\nabla} \\ \frac{\partial}{\partial t} &\longrightarrow \frac{1}{\tau\_{\psi\_0}} \frac{\partial}{\partial \ddot{t}} \\ \mathbf{A} &\longrightarrow \sqrt{2} H\_{c0} \lambda\_0 \ddot{\mathbf{A}} \\ \psi &\longrightarrow \sqrt{\frac{|\alpha\_0|}{\beta\_0}} \ddot{\psi} \\ \phi &\longrightarrow \phi\_0 \ddot{\phi}. \end{aligned}$$

If we substitute in Eqs. [9,](#page-3-0) [10,](#page-3-1) [11,](#page-3-2) and [12](#page-3-3) for α, β, Γ, and σ<sup>n</sup> respectively, we can then define the quantities:

$$\begin{aligned} \lambda\_0 &= \sqrt{\frac{m\_s c^2 \beta\_0}{4\pi e\_s^2 |\alpha\_0|}}\\ \xi\_0 &= \sqrt{\frac{\hbar^2}{2m\_s |\alpha\_0|}}\\ \kappa\_0 &= \frac{\lambda\_0}{\xi\_0} \\ H\_{c0} &= \sqrt{\frac{4\pi \alpha\_0^2}{\beta\_0}}\\ \tau\_{\psi\_0} &= \frac{\Gamma\_0}{|\alpha\_0|}\\ \tau\_{j\_0} &= \frac{\sigma\_{n0} m\_s \beta\_0}{e\_s^2 |\alpha\_0|}\\ u\_0 &= \frac{\tau\_{\psi\_0}}{\tau\_{j\_0}}\\ \dot{\phi}\_0 &= \frac{\hbar \kappa\_0}{e\_s \tau\_{\psi\_0}}. \end{aligned}$$

Using these relations, the resulting equations under the above coordinate transformations simplify into Eqs. [13](#page-3-4) and [14](#page-3-5) (where we then drop the tildes).

## Appendix: Nondimensionalizing the TDGL Dissipation

We start with Equation [23,](#page-4-0)

$$D = 2\Gamma \left| \left( \frac{\partial \psi}{\partial t} + \frac{ie\_s \phi \psi}{\hbar} \right) \right|^2 + \sigma\_n \mathbf{E}^2$$

choosing the temporal gauge (ϕ = 0) we have

$$D = 2\Gamma \left| \frac{\partial \psi}{\partial t} \right|^2 + \sigma\_n \left( \frac{\partial \mathbf{A}}{\partial t} \right)^2.$$

Next, we make the same coordinate transformations as from the previous section (and the same time transformation as from the methods section) and use the expressions for τ<sup>ψ</sup> and H<sup>c</sup> on the first term:

$$\begin{split} D &= \frac{2\Gamma\alpha}{\beta} \frac{T\_{sim}^{2}}{T^{2}} \left| \frac{\partial\tilde{\psi}}{\partial\tilde{t}} \right|^{2} + 2\sigma\_{n} H\_{c}^{2} \lambda^{2} \frac{T\_{sim}^{2}}{T^{2}} \left( \frac{\partial\tilde{\mathbf{A}}}{\partial\tilde{t}} \right)^{2} \\ &= \frac{2\tau\_{\psi}\alpha^{2}}{\beta} \frac{T\_{sim}^{2}}{T^{2}} \left| \frac{\partial\tilde{\psi}}{\partial\tilde{t}} \right|^{2} + 2\sigma\_{n} H\_{c}^{2} \lambda^{2} \frac{T\_{sim}^{2}}{T^{2}} \left( \frac{\partial\tilde{\mathbf{A}}}{\partial\tilde{t}} \right)^{2} \\ &= \frac{2H\_{c}^{2}}{4\pi} \frac{T\_{sim}}{T} \left| \frac{\partial\tilde{\psi}}{\partial\tilde{t}} \right|^{2} + 2\sigma\_{n} H\_{c}^{2} \lambda^{2} \frac{T\_{sim}^{2}}{T^{2}} \left( \frac{\partial\tilde{\mathbf{A}}}{\partial\tilde{t}} \right)^{2}, \end{split}$$

where in the last line we used the fact that T = τψTsim. Finally, this expression is in Gaussian units so we con-

- [1] S. Posen and M. Liepe, Advances in development of nb3Sn superconducting radio-frequency cavities, [Phys.](https://doi.org/10.1103/PhysRevSTAB.17.112001) [Rev. ST Accel. Beams](https://doi.org/10.1103/PhysRevSTAB.17.112001) 17, 112001 (2014).
- [2] S. Posen and D. L. Hall, nb3Sn superconducting radiofrequency cavities: fabrication, results, properties, and prospects, [Superconductor Science and Technology](https://doi.org/10.1088/1361-6668/30/3/033004) 30[, 033004 \(2017\).](https://doi.org/10.1088/1361-6668/30/3/033004)
- [3] A. Gurevich, Theory of rf superconductivity for resonant cavities, Superconductor Science and Technology 30, 034004 (2017).
- [4] Z. Sun, T. Oseroff, Z. Baraissov, D. K. Dare, K. Howard, B. Francis, A. C. Hire, N. Sitaraman, T. A. Arias, M. K. Transtrum, R. Hennig, M. O. Thompson, D. A. Muller, and M. U. Liepe, Zrnb(co) rf superconducting thin film with high critical temperature in the theoretical limit, [Advanced Electronic Materials](https://doi.org/https://doi.org/10.1002/aelm.202300151) 9, 2300151 (2023).
- [5] N. S. Sitaraman, Z. Sun, B. L. Francis, A. C. Hire, T. Oseroff, Z. Baraissov, T. A. Arias, R. G. Hennig, M. U. Liepe, D. A. Muller, and M. K. Transtrum (Center for Bright Beams), Enhanced surface superconductivity of niobium by zirconium doping, [Phys. Rev. Appl.](https://doi.org/10.1103/PhysRevApplied.20.014064) 20, [014064 \(2023\).](https://doi.org/10.1103/PhysRevApplied.20.014064)
- [6] J. Bardeen and M. J. Stephen, Theory of the motion of vortices in superconductors, [Phys. Rev.](https://doi.org/10.1103/PhysRev.140.A1197) 140, A1197 [\(1965\).](https://doi.org/10.1103/PhysRev.140.A1197)
- [7] D. B. Liarte, S. Posen, M. K. Transtrum, G. Catelani, M. Liepe, and J. P. Sethna, Theoretical estimates of maximum fields in superconducting resonant radio frequency cavities: stability theory, disorder, and laminates, [Super](https://doi.org/10.1088/1361-6668/30/3/033002)[conductor Science and Technology](https://doi.org/10.1088/1361-6668/30/3/033002) 30, 033002 (2017).
- [8] S. Posen, N. Valles, and M. Liepe, Radio frequency magnetic field limits of nb and nb3Sn, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.115.047001) 115, [047001 \(2015\).](https://doi.org/10.1103/PhysRevLett.115.047001)
- [9] L. Kramer, Stability limits of the meissner state and the mechanism of spontaneous vortex nucleation in superconductors, Phys. Rev. 170[, 475 \(1968\).](https://doi.org/10.1103/PhysRev.170.475)
- [10] A. J. Dolgert, S. J. Di Bartolo, and A. T. Dorsey, Superheating fields of superconductors: Asymptotic analysis and numerical results, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.53.5650) 53, 5650 (1996).
- [11] M. K. Transtrum, G. Catelani, and J. P. Sethna, Superheating field of superconductors within ginzburg-landau theory, Phys. Rev. B 83[, 094505 \(2011\).](https://doi.org/10.1103/PhysRevB.83.094505)
- [12] D. B. Liarte, M. K. Transtrum, and J. P. Sethna, Ginzburg-landau theory of the superheating field anisotropy of layered superconductors, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.94.144504) 94, [144504 \(2016\).](https://doi.org/10.1103/PhysRevB.94.144504)

vert to SI units so that it is compatible with the other expressions in Section [II E:](#page-5-0)

$$\begin{split} D &= 2\tau\_{\psi}\mu\_{0}H\_{c}^{2}\frac{T\_{sim}^{2}}{T^{2}}\left|\frac{\partial\tilde{\psi}}{\partial\tilde{t}}\right|^{2} + 2\sigma\_{n}\mu\_{0}^{2}H\_{c}^{2}\lambda^{2}\frac{T\_{sim}^{2}}{T^{2}}\left(\frac{\partial\tilde{\mathbf{A}}}{\partial\tilde{t}}\right)^{2} \\ &= 2\mu\_{0}H\_{c}^{2}\frac{T\_{sim}}{T}\left(\left|\frac{\partial\tilde{\psi}}{\partial\tilde{t}}\right|^{2} + \sigma\_{n}\mu\_{0}\lambda^{2}\frac{T\_{sim}}{T}\left(\frac{\partial\tilde{\mathbf{A}}}{\partial\tilde{t}}\right)^{2}\right) \end{split}$$

- [13] G. Catelani and J. P. Sethna, Temperature dependence of the superheating field for superconductors in the high-κ london limit, Phys. Rev. B 78, [10.1103/Phys-](https://doi.org/10.1103/PhysRevB.78.224509)[RevB.78.224509](https://doi.org/10.1103/PhysRevB.78.224509) (2008), cited by: 53; All Open Access, Green Open Access.
- [14] F. P.-J. Lin and A. Gurevich, Effect of impurities on the superheating field of type-ii superconductors, Phys. Rev. B 85, [10.1103/PhysRevB.85.054513](https://doi.org/10.1103/PhysRevB.85.054513) (2012), cited by: 40; All Open Access, Green Open Access.
- [15] T. Kubo, Superheating fields of semi-infinite superconductors and layered superconductors in the diffusive limit: structural optimization based on the microscopic theory, [Superconductor Science and Technology](https://doi.org/10.1088/1361-6668/abdedd) 34[, 045006 \(2021\).](https://doi.org/10.1088/1361-6668/abdedd)
- <span id="page-12-1"></span>[16] S. Posen, J. Lee, D. Seidman, A. Romanenko, B. Tennis, O. Melnychuk, and D. Sergatskov, Advances in nb3sn superconducting radiofrequency cavities towards first practical accelerator applications, Superconductor Science and Technology 34, [10.1088/1361-6668/abc7f7](https://doi.org/10.1088/1361-6668/abc7f7) (2021), cited by: 19; All Open Access, Green Open Access, Hybrid Gold Open Access.
- [17] G. M¨uller, H. Piel, J. Pouryamout, P. Boccard, and P. Kneisel, Proceedings of the workshop on thin film coating methods for superconducting accelerating cavities (2000).
- [18] A. R. Pack, J. Carlson, S. Wadsworth, and M. K. Transtrum, Vortex nucleation in superconductors within time-dependent ginzburg-landau theory in two and three dimensions: Role of surface defects and material inhomogeneities, Phys. Rev. B 101[, 144504 \(2020\).](https://doi.org/10.1103/PhysRevB.101.144504)
- <span id="page-12-0"></span>[19] J. Carlson, A. Pack, M. K. Transtrum, J. Lee, D. N. Seidman, D. B. Liarte, N. S. Sitaraman, A. Senanian, M. M. Kelley, J. P. Sethna, T. Arias, and S. Posen, Analysis of magnetic vortex dissipation in sn-segregated boundaries in nb3Sn superconducting rf cavities, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.103.024516) 103, [024516 \(2021\).](https://doi.org/10.1103/PhysRevB.103.024516)
- [20] B. Oripov and S. M. Anlage, Time-dependent ginzburglandau treatment of rf magnetic vortices in superconductors: Vortex semiloops in a spatially nonuniform magnetic field, Physical Review E 101, [10.1103/Phys-](https://doi.org/10.1103/PhysRevE.101.033306)[RevE.101.033306](https://doi.org/10.1103/PhysRevE.101.033306) (2020), cited by: 13; All Open Access, Bronze Open Access, Green Open Access.
- [21] B. Oripov, T. Bieler, G. Ciovati, S. Calatroni, P. Dhakal, T. Junginger, O. B. Malyshev, G. Terenziani, A.-M. Valente-Feliciano, R. Valizadeh, S. Wilde, and S. M. Anlage, High-frequency nonlinear response of supercon-

ducting cavity-grade Nb surfaces, [Phys. Rev. Appl.](https://doi.org/10.1103/PhysRevApplied.11.064030) 11, [064030 \(2019\).](https://doi.org/10.1103/PhysRevApplied.11.064030)

- [22] J. Allmaras, A. Kozorezov, B. Korzh, K. Berggren, and M. Shaw, Intrinsic timing jitter and latency in superconducting nanowire single-photon detectors, [Phys. Rev.](https://doi.org/10.1103/PhysRevApplied.11.034062) Appl. 11[, 034062 \(2019\).](https://doi.org/10.1103/PhysRevApplied.11.034062)
- [23] M. J¨onsson, R. Vedin, S. Gyger, J. A. Sutton, S. Steinhauer, V. Zwiller, M. Wallin, and J. Lidmar, Current crowding in nanoscale superconductors within the ginzburg-landau model, [Phys. Rev. Appl.](https://doi.org/10.1103/PhysRevApplied.17.064046) 17, 064046 [\(2022\).](https://doi.org/10.1103/PhysRevApplied.17.064046)
- [24] L. Bishop-Van Horn, E. Mueller, and K. A. Moler, Vortex dynamics induced by scanning squid susceptometry, Phys. Rev. B 107[, 224509 \(2023\).](https://doi.org/10.1103/PhysRevB.107.224509)
- [25] L. Bishop-Van Horn, pytdgl: Time-dependent ginzburglandau in python, [Computer Physics Communications](https://doi.org/https://doi.org/10.1016/j.cpc.2023.108799) 291[, 108799 \(2023\).](https://doi.org/https://doi.org/10.1016/j.cpc.2023.108799)
- [26] T. J. Rieger, D. J. Scalapino, and J. E. Mercereau, Charge conservation and chemical potentials in timedependent ginzburg-landau theory, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.27.1787) 27, [1787 \(1971\).](https://doi.org/10.1103/PhysRevLett.27.1787)
- [27] T. J. Rieger, D. J. Scalapino, and J. E. Mercereau, Timedependent superconductivity and quantum dissipation, Phys. Rev. B 6[, 1734 \(1972\).](https://doi.org/10.1103/PhysRevB.6.1734)
- [28] S. Frota-Pessˆoa, J. A. Blackburn, and B. B. Schwartz, Short weak link with distinct chemical potentials at the boundary, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.20.993) 20, 993 (1979).
- [29] G. R. Berdiyorov, X. H. Chao, F. M. Peeters, H. B. Wang, V. V. Moshchalkov, and B. Y. Zhu, Magnetoresistance oscillations in superconducting strips: A ginzburg-landau study, Phys. Rev. B 86[, 224504 \(2012\).](https://doi.org/10.1103/PhysRevB.86.224504)
- [30] G. Kimmel, A. Glatz, and I. S. Aranson, Phase slips in superconducting weak links, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.95.014518) 95, 014518 [\(2017\).](https://doi.org/10.1103/PhysRevB.95.014518)
- [31] L. Peng, Y. Hu, Z. Li, K. Deng, Y. Zhu, L. Xu, and Y. Zhou, Finite element treatment of vortex states in 3d mesoscopic cylindrical superconductors in a tilted magnetic field, [Acta Physica Polonica A](https://doi.org/10.12693/APhysPolA.133.152) 133, 152 – 156 [\(2018\),](https://doi.org/10.12693/APhysPolA.133.152) cited by: 2; All Open Access, Bronze Open Access.
- [32] K. Arutyunov, D. Golubev, and A. Zaikin, Superconductivity in one dimension, [Physics Reports](https://doi.org/https://doi.org/10.1016/j.physrep.2008.04.009) 464, 1 (2008).
- [33] S. Michotte, S. M´at´efi-Tempfli, L. Piraux, D. Y. Vodolazov, and F. M. Peeters, Condition for the occurrence of phase slip centers in superconducting nanowires under applied current or voltage, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.69.094512) 69, 094512 [\(2004\).](https://doi.org/10.1103/PhysRevB.69.094512)
- <span id="page-13-0"></span>[34] S. Posen, N. Valles, and M. Liepe, Radio frequency magnetic field limits of nb and nb3Sn, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.115.047001) 115, [047001 \(2015\).](https://doi.org/10.1103/PhysRevLett.115.047001)
- [35] Y. Trenikhina, S. Posen, A. Romanenko, M. Sardela, J.- M. Zuo, D. L. Hall, and M. Liepe, Performance-defining properties of nb3sn coating in srf cavities, [Superconduc](https://doi.org/10.1088/1361-6668/aa9694)[tor Science and Technology](https://doi.org/10.1088/1361-6668/aa9694) 31, 015004 (2017).
- [36] J. Lee, S. Posen, Z. Mao, Y. Trenikhina, K. He, D. L. Hall, M. Liepe, and D. N. Seidman, Atomic-scale analyses of nb3sn on nb prepared by vapor diffusion for superconducting radiofrequency cavity applications: a correlative study, [Superconductor Science and Technology](https://doi.org/10.1088/1361-6668/aaf268) 32, [024001 \(2018\).](https://doi.org/10.1088/1361-6668/aaf268)
- [37] U. Pudasaini, G. V. Eremeev, J. W. Angle, J. Tuggle, C. E. Reece, and M. J. Kelley, Growth of Nb3Sn coating in tin vapor-diffusion process, [Journal of Vacuum Science & Technology A](https://doi.org/10.1116/1.5113597) 37,

[051509 \(2019\),](https://doi.org/10.1116/1.5113597) [https://pubs.aip.org/avs/jva/article](https://arxiv.org/abs/https://pubs.aip.org/avs/jva/article-pdf/doi/10.1116/1.5113597/13962261/051509_1_online.pdf)[pdf/doi/10.1116/1.5113597/13962261/051509](https://arxiv.org/abs/https://pubs.aip.org/avs/jva/article-pdf/doi/10.1116/1.5113597/13962261/051509_1_online.pdf) 1 online.pdf.

- [38] J. Lee, Z. Mao, K. He, Z. H. Sung, T. Spina, S.-I. Baik, D. L. Hall, M. Liepe, D. N. Seidman, and S. Posen, Grainboundary structure and segregation in nb3sn coatings on nb for high-performance superconducting radiofrequency cavity applications, [Acta Materialia](https://doi.org/https://doi.org/10.1016/j.actamat.2020.01.055) 188, 155 (2020).
- [39] S.-H. Oh, D. Seol, Y.-J. Jeong, S.-H. Na, J. Kim, W.-S. Ko, J. B. Jeon, and B.-J. Lee, Diffusion in a15 nb3sn: An atomistic study, [Acta Materialia](https://doi.org/https://doi.org/10.1016/j.actamat.2022.118050) 234, 118050 (2022).
- [40] C. Becker, S. Posen, N. Groll, R. Cook, C. M. Schlep¨utz, D. L. Hall, M. Liepe, M. Pellin, J. Zasadzinski, and T. Proslier, Analysis of Nb3Sn surface layers for superconducting radio frequency cavity applications, [Applied Physics Letters](https://doi.org/10.1063/1.4913617) 106, [082602 \(2015\),](https://doi.org/10.1063/1.4913617) [https://pubs.aip.org/aip/apl/article](https://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/doi/10.1063/1.4913617/13593886/082602_1_online.pdf)[pdf/doi/10.1063/1.4913617/13593886/082602](https://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/doi/10.1063/1.4913617/13593886/082602_1_online.pdf) 1 online.pdf.
- <span id="page-13-1"></span>[41] E. Viklund, J. Lee, D. Seidman, and S. Posen, Three-dimensional reconstruction of nb3sn films by focused ion beam cross sectional microscopy, IEEE Transactions on Applied Superconductivity 33, [10.1109/TASC.2023.3257819](https://doi.org/10.1109/TASC.2023.3257819) (2023), cited by: 0.
- [42] J. K. Freericks, A. Y. Liu, A. Quandt, and J. Geerk, Nonconstant electronic density of states tunneling inversion for a15 superconductors: nb3Sn, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.65.224510) 65, [224510 \(2002\).](https://doi.org/10.1103/PhysRevB.65.224510)
- [43] W. Markiewicz, Elastic stiffness model for the critical temperature tc of nb3sn including strain dependence, Cryogenics 44[, 767 \(2004\).](https://doi.org/https://doi.org/10.1016/j.cryogenics.2004.03.019)
- [44] M. G. T. Mentink, M. M. J. Dhalle, D. R. Dietderich, A. Godeke, F. Hellman, and H. H. J. ten Kate, The effects of disorder on the normal state and superconducting properties of nb3sn, [Superconductor Science and Tech](https://doi.org/10.1088/1361-6668/30/2/025006)nology 30[, 025006 \(2016\).](https://doi.org/10.1088/1361-6668/30/2/025006)
- [45] M. M. Kelley, N. S. Sitaraman, and T. A. Arias, Ab initio theory of the impact of grain boundaries and substitutional defects on superconducting nb3Sn, [Superconductor](https://doi.org/10.1088/1361-6668/abc8ce) [Science and Technology](https://doi.org/10.1088/1361-6668/abc8ce) 34, 015015 (2020).
- <span id="page-13-2"></span>[46] N. S. Sitaraman, M. M. Kelley, R. D. Porter, M. U. Liepe, T. A. Arias, J. Carlson, A. R. Pack, M. K. Transtrum, and R. Sundararaman, Effect of the density of states at the fermi level on defect free energies and superconductivity: A case study of nb3Sn, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.103.115106) 103, 115106 [\(2021\).](https://doi.org/10.1103/PhysRevB.103.115106)
- [47] A. E. Koshelev, I. A. Sadovskyy, C. L. Phillips, and A. Glatz, Optimization of vortex pinning by nanoparticles using simulations of the time-dependent ginzburglandau model, Phys. Rev. B 93[, 060508 \(2016\).](https://doi.org/10.1103/PhysRevB.93.060508)
- [48] I. Sadovskyy, A. Koshelev, C. Phillips, D. Karpeyev, and A. Glatz, Stable large-scale solver for ginzburg–landau equations for superconductors, [Journal of Computational](https://doi.org/https://doi.org/10.1016/j.jcp.2015.04.002) Physics 294[, 639 \(2015\).](https://doi.org/https://doi.org/10.1016/j.jcp.2015.04.002)
- [49] M. P. Sørensen, N. F. Pedersen, and M. Ogren, The ¨ dynamics of magnetic vortices in type ii superconductors with pinning sites studied by the time dependent ginzburg–landau model, [Physica C: Superconductivity](https://doi.org/https://doi.org/10.1016/j.physc.2016.08.001) [and its Applications](https://doi.org/https://doi.org/10.1016/j.physc.2016.08.001) 533, 40 (2017).
- [50] S. J. Chapman, C. M. Elliott, A. K. Head, S. D. Howison, F. M. Leslie, J. R. Ockendon, J. Deang, Q. Du, M. Gunzburger, and J. Peterson, Vortices in superconductors: modelling and computer simulations, [Philosoph](https://doi.org/10.1098/rsta.1997.0098)[ical Transactions of the Royal Society of London. Se](https://doi.org/10.1098/rsta.1997.0098)[ries A: Mathematical, Physical and Engineering Sciences](https://doi.org/10.1098/rsta.1997.0098)

355[, 1957 \(1997\).](https://doi.org/10.1098/rsta.1997.0098)

- [51] A. Al Luhaibi, A. Glatz, and J. B. Ketterson, Driven responses of periodically patterned superconducting films, Phys. Rev. B 106[, 224516 \(2022\).](https://doi.org/10.1103/PhysRevB.106.224516)
- [52] V. L. Ginzburg and L. D. Landau, On the theory of superconductivity, in [On Superconductivity and Superfluidity:](https://doi.org/10.1007/978-3-540-68008-6_4) [A Scientific Autobiography](https://doi.org/10.1007/978-3-540-68008-6_4) (Springer Berlin Heidelberg, Berlin, Heidelberg, 2009) pp. 113–137.
- [53] A. Schmid, A time dependent ginzburg-landau equation and its application to the problem of resistivity in the mixed state, [Physik der kondensierten Materie](https://doi.org/10.1007/BF02422669) 5, 302 [\(1966\).](https://doi.org/10.1007/BF02422669)
- [54] L. Gor'Kov and G. Eliashberg, Generalization of the ginzburg-landau equations for non-stationary problems in the case of alloys with paramagnetic impurities, Sov. Phys. JETP 27, 328 (1968).
- [55] Q. Du, M. D. Gunzburger, and J. S. Peterson, Analysis and approximation of the ginzburg–landau model of superconductivity, SIAM Review 34[, 54–81 \(1992\).](https://doi.org/10.1137/1034003)
- [56] N. Kopnin, Theory of nonequilibrium superconductivity, Vol. 110 (Oxford University Press, 2001).
- [57] H. Gao and W. Sun, A new mixed formulation and efficient numerical solution of ginzburg– landau equations under the temporal gauge, [SIAM](https://doi.org/10.1137/15M1022744) [Journal on Scientific Computing](https://doi.org/10.1137/15M1022744) 38, A1339 (2016), [https://doi.org/10.1137/15M1022744.](https://arxiv.org/abs/https://doi.org/10.1137/15M1022744)
- [58] Q. Du, Global existence and uniqueness of solutions of the time-dependent ginzburg-landau model for superconductivity, Applicable Analysis 53, 1 (1994).
- [59] M. S. Alnæs, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg, C. Richardson, J. Ring, M. E. Rognes, and G. N. Wells, The fenics project version 1.5, Archive of Numerical Software 3, 9 (2015).
- [60] M. Tinkham, Viscous flow of flux in type-ii superconductors, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.13.804) 13, 804 (1964).
- [61] C. J. Gorter and H. Casimir, On supraconductivity i, Physica 1, 306 (1934).
- [62] H. London, Production of heat in supraconductors by alternating currents, Nature 133, 497 (1934).
- [63] J. Halbritter, On surface resistance of superconductors, Zeitschrift f¨ur Physik 266, 209 (1974).
- [64] J. Turneaure, J. Halbritter, and H. Schwettman, The surface impedance of superconductors and normal conductors: The mattis-bardeen theory, Journal of Superconductivity 4, 341 (1991).
- [65] D. C. Mattis and J. Bardeen, Theory of the anomalous skin effect in normal and superconducting metals, Phys-

ical Review 111, 412 (1958).

- [66] A. Abrikosov, L. Gor'Kov, and I. Khalatnikov, A superconductor in a high frequency field, Sov. Phys. JETP 35, 182 (1959).
- [67] A. Gurevich, Theory of rf superconductivity for resonant cavities, Superconductor Science and Technology 30, 034004 (2017).
- <span id="page-14-4"></span>[68] A. Gurevich and T. Kubo, Surface impedance and optimum surface resistance of a superconductor with an imperfect surface, Phys. Rev. B 96[, 184515 \(2017\).](https://doi.org/10.1103/PhysRevB.96.184515)
- [69] B. Aune, R. Bandelmann, D. Bloess, B. Bonin, A. Bosotti, M. Champion, C. Crawford, G. Deppe, B. Dwersteg, D. A. Edwards, H. T. Edwards, M. Ferrario, M. Fouaidy, P.-D. Gall, A. Gamp, A. G¨ossel, J. Graber, D. Hubert, M. H¨uning, M. Juillard, T. Junquera, H. Kaiser, G. Kreps, M. Kuchnir, R. Lange, M. Leenen, M. Liepe, L. Lilje, A. Matheisen, W.-D. M¨oller, A. Mosnier, H. Padamsee, C. Pagani, M. Pekeler, H.- B. Peters, O. Peters, D. Proch, K. Rehlich, D. Reschke, H. Safa, T. Schilcher, P. Schm¨user, J. Sekutowicz, S. Simrock, W. Singer, M. Tigner, D. Trines, K. Twarowski, G. Weichert, J. Weisend, J. Wojtkiewicz, S. Wolff, and K. Zapfe, Superconducting tesla cavities, [Phys. Rev. ST](https://doi.org/10.1103/PhysRevSTAB.3.092001) Accel. Beams 3[, 092001 \(2000\).](https://doi.org/10.1103/PhysRevSTAB.3.092001)
- [70] N. W. Ashcroft and N. D. Mermin, Solid State Physics (Holt, Rinehart and Winston, New York, 1976).
- [71] C. Geuzaine and J.-F. Remacle, Gmsh: A 3-d finite element mesh generator with built-in pre-and postprocessing facilities, International journal for numerical methods in engineering 79, 1309 (2009).
- <span id="page-14-0"></span>[72] E. Viklund, D. N. Seidman, D. Burk, and S. Posen, Improving nb3sn cavity performance using centrifugal barrel polishing (2023), [arXiv:2305.10226 \[physics.acc-ph\].](https://arxiv.org/abs/2305.10226)
- <span id="page-14-1"></span>[73] M. Tinkham, Introduction to Superconductivity (Courier Corporation, 2004).
- <span id="page-14-2"></span>[74] L. Kramer and R. J. Watts-Tobin, Theory of dissipative current-carrying states in superconducting filaments, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.40.1041) 40, 1041 (1978).
- <span id="page-14-3"></span>[75] T. Proslier, J. F. Zasadzinski, L. Cooley, C. Antoine, J. Moore, J. Norem, M. Pellin, and K. E. Gray, Tunneling study of cavity grade nb: Possible magnetic scattering at the surface, [Applied Physics Letters](https://doi.org/10.1063/1.2913764) 92, 212505 (2008), [https://doi.org/10.1063/1.2913764.](https://arxiv.org/abs/https://doi.org/10.1063/1.2913764)
- <span id="page-14-5"></span>[76] T. Kubo and A. Gurevich, Field-dependent nonlinear surface resistance and its optimization by surface nanostructuring in superconductors, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.100.064522) 100, 064522 [\(2019\).](https://doi.org/10.1103/PhysRevB.100.064522)