# arXiv:2412.04159

**Paper ID:** 7146b4f0f0f8d182197ef76d521a6fbb

**PDF Path:** apl/Superconductors/arXiv:2412.04159.pdf

**Processing Status:** complete

**Captions Added:** 9

**Generated:** 2025-06-24T13:44:28.037895

---

# arXiv:2412.04159v1 [cond-mat.supr-con] 5 Dec 2024

# In-situ Investigation of the Phase Formation and Superconductivity in V3Si Thin Films at High Temperatures

<sup>1</sup>M. Bose, <sup>2</sup>D.L. Cortie, <sup>3</sup>S. Rubanov, <sup>2</sup>A.P. Le Brun,1T.R. Finlayson and <sup>1</sup>J.C. McCallum

<sup>1</sup>School of Physics, The University of Melbourne, Parkville, Victoria 3010

<sup>2</sup>Australian Nuclear Science and Technology Organisation, NSW, Australia

3 Ian Holmes Imaging Centre, Bio21 Institute, University of Melbourne, Parkville, Victoria 3010

Vanadium silicide (V3Si) is a promising superconductor for integration with silicon-based electronics, however the interfacial growth kinetics have a strong influence on the resulting superconducting properties and are not yet fully understood. In this study, we have used neutron reflectometry to reveal the phase transformation during thin film growth driven by different annealing strategies. We examined the silicide formation when a thin layer of vanadium undergoes reactive diffusion with a silicon dioxide film on silicon at temperatures from 650–800 ◦C. To further investigate the time evolution of different phases under various annealing temperatures, a chemical model was developed and subsequent simulations were performed. The results of this model were validated using X-ray diffraction and cross-sectional TEM analysis. Correlations were observed between the structure and superconducting properties. Over-annealing films leads to complete depletion of the SiO<sup>2</sup> barrier layer, forming diffuse interfaces and driving the formation of undesirable silicon-rich silicides. Avoiding this by controlling time and temperature, allows higher quality superconducting films to be achieved. The T<sup>c</sup> of the films was found to be 13 K, and the annealing conditions influenced the critical fields and the paramagnetic Meissner effect near Tc. For optimally–annealed films, superconducting order parameters were calculated. Ginzberg-Landau theory was applied to explain flux penetration.

# I. INTRODUCTION

Superconducting quantum circuits have gained prominence in various quantum technologies. They form part of mainstream quantum computing applications due to their ease of integration with traditional Complementary Metal-Oxide-Semiconductor (CMOS) technology and their very short gating times in the order of nanoseconds [\[1\]](#page-11-0). Incorporation of both superconducting and semiconducting components within a single semiconductor device has great potential in the development of novel quantum device structures [\[2\]](#page-11-1). However, increasing scale and fidelity are still bottlenecks in the development of superconducting quantum computers, especially with large scale industry demands. Also, the current superconducting qubits need to be cooled down to cryogenic temperatures, usually at millikelvin, using a dilution refrigerator, for coherent manipulation of quantum states [\[3\]](#page-11-2). If higher qubit operating temperatures can be realized, this can increase the available cooling power, leading to improved scalability and integration of qubits [\[4,](#page-11-3) [5\]](#page-11-4). In this context, choosing a superconducting material with a high critical temperature that can be easily integrated with classical electronics is pertinent.

The A15 family of superconductors, which exhibit the β-W crystal structure with the cubic space group Pm3, have been well-known for the past 70 years. Among these, superconductivity in the binary compound vanadium silicide (V3Si) having a superconducting critical temperature, Tc, of 17.1 K, was discovered by Hardy G.F. and Hulm J.K. [\[6,](#page-11-5) [7\]](#page-11-6). The realization of an upper critical field in excess of 23 T [\[8\]](#page-11-7) resulted in intensive research on this material for potential applications. The successful growth of single crystals of V3Si [\[9\]](#page-11-8) resulted in the subsequent discovery of a structural transformation at a temperature, Tm, just a few degrees above Tc, by both neutron [\[10\]](#page-11-9) and x-ray diffraction [\[11\]](#page-11-10), as well as by other techniques such as the temperature dependence of the elastic constant, (C<sup>11</sup> - C12)/2 [\[12\]](#page-11-11), and specific heat [\[13\]](#page-11-12). In the case of the elastic constant measurements, while all single crystals studied exhibited intensive elastic softening with temperature from room temperature to just above Tc, not all samples exhibited the structural transformation [\[12\]](#page-11-11). X-ray diffraction studies of polycrystalline V3Si have also revealed the structural transformation a few degrees above T<sup>c</sup> in some samples [\[14\]](#page-11-13), Consequently, there has been a wealth of research both theoretical [\[15,](#page-11-14) [16\]](#page-11-15) and experimental [\[17–](#page-11-16)[21\]](#page-11-17), attempting to explain the relationship between the structural behaviour and superconductivity. More recently, the superconductivity of V3Si thin films has received attention in the literature [\[22,](#page-11-18) [23\]](#page-11-19), on account of the potential of a thin film of V3Si grown on a suitable substrate, for device development.

V3Si thin films can be developed through various methods, including sputtering [\[24\]](#page-11-20), dual-beam evaporation of vanadium and silicon [\[25\]](#page-11-21), and e-beam evaporation [\[26,](#page-11-22) [27\]](#page-11-23) of vanadium on different substrates. In most cases, a subsequent high-temperature annealing step is necessary for silicide formation with the desired stoichiometry. A15 superconducting thin films are receiving renewed attention in recent times due to their potential use in spintronics, primarily due to their manifestation of the spin Hall effect (SHE) [\[28\]](#page-11-24) and in quantum computing applications [\[29\]](#page-11-25), driven by their high critical temperature and critical field. Therefore, understanding the phase transformation processes involved in the formation of the V3Si films and their superconducting properties are of importance. However, there are difficulties associated with developing thin films, primarily in controlling the thickness of the V3Si layer. The growth kinetics such as annealing temperature and time that directly affect the film's thickness, interface roughnesses and superconducting quality are not yet completely understood. Hence, it is of importance to determine the optimal growth parameters that help to achieve precise control over the superconducting layer. The chemical mechanisms that govern the growth kinetics and their effect on the layer properties are also worthy of attention to identify how varying growth conditions modify the superconducting properties of the films. Addressing these challenges is critical in advancing the use of V3Si in quantum devices, where stable and reproducible device outcomes and performance will be required across different advanced fabrication technologies.

In this work, we performed in-situ annealing of the prepared V thin film samples and investigated the phase changes at different temperatures using neutron reflectometry for the first time. Neutron reflectometry is a technique that can be used to understand the surface and interface structure down to the nanoscale (0.5-300 nm thickness range). Neutrons are highly penetrating, making them the best possible source for in-situ reflectometry studies [\[30\]](#page-11-26). V3Si single crystals have previously been subjected to neutron scattering experiments for studies of acoustic-phonon softening [\[31\]](#page-11-27), phonon density of states [\[32\]](#page-11-28) and the flux-line lattice (FLL) transition [\[33\]](#page-11-29).

Regardless of the deposition method for the V3Si films, the major challenge is achieving the correct 3:1 phase ratio. This is because there are several factors affecting the phase transformation, such as ratio between V and Si, annealing conditions such as temperature, time and the presence of oxygen. The neutron reflectometry method proposed in this study is capable of addressing these challenges by revealing evolution of the phase changes during the in-situ annealing. For this study we use the e-beam method, where metallic V thin films are evaporated on to an oxidized silicon substrate. The e-beam evaporation is a cost-effective way of developing thin films and hence it is of interest to identify the phase transformations during formation of V3Si by this pathway.

After the phase transformation studies, the superconducting properties of the samples were analyzed. The second half of this paper is completely dedicated to the discussion of magnetometry measurement results, that include details of the superconducting order parameters. We also examine the flux penetration in to these films, based on vortex core size using Ginzburg-Landau theory.

# II. EXPERIMENTAL TECHNIQUE

# A. Sample Preparation

In this study, we utilized p-type silicon substrates measuring 2 x 2 cm<sup>2</sup> with a room temperature resistivity of 10 ohm-cm. The sample preparation involved the thermal growth of a 200 nm silicon dioxide (SiO2) layer on the silicon substrate through dry oxidation at 1000◦C. Subsequently, a 300 nm thick V layer was deposited by electron-beam evaporation at a rate of 7 ˚A/s, with a base pressure of 2 × 10−<sup>7</sup> mbar.

# B. Neutron Reflectometry Method

Neutron reflection involves the specular reflection of neutron planewaves from a 1D surface potential, which can be used to profile the isotopic profile of thin films and surfaces with a depth resolution between 1-300 nm. The process can be described by a 1D time-independent Schrodinger equation:

$$\left[-\frac{\hbar^2}{2m\_n}\frac{\delta}{\delta z^2} + V(z)\right]\Psi(z) = E\Psi(z). \tag{1}$$

where for unpolarized neutrons the potential is:

$$V(z) = \frac{2\pi\hbar^2}{m\_n}\rho\_N(z),\tag{2}$$

where m<sup>n</sup> is the neutron mass, ρ<sup>N</sup> is the nuclear scattering length density (SLD):

$$
\rho\_N = \sum\_i N\_i b\_i,\tag{3}
$$

where the sum runs over the different isotopes in each layer, each with number density N<sup>i</sup> and neutron coherent scattering length b<sup>i</sup> . The numerical methods to model the process are well known, and ultimately boundary conditions at different interfaces determine the neutron wave function, and the amplitude of the neutron wave function |Ψ| 2 that is measured at the detector. In practice, the mathematics of neutron reflection is often understood using analogous optical wave equations, such as those that apply to X-ray reflectometry.

Neutron reflectometry measurements were conducted using the Spatz neutron beam instrument at the 20 MW OPAL research reactor (Australian Nuclear Science and Technology Organisation (ANSTO), Lucas Heights, Australia) [\[34\]](#page-12-0). The Spatz instrument views the cold neutron source and uses the time-of-flight (ToF) principle. The disc chopper settings used were a pairing of discs 1 and 2 set 480 mm apart for a fractional wave length resolution δλ λ of 5%. The chopper discs were rotating at a speed of 25 Hz to generate a usable neutron bandwidth of wavelengths 2.8 to 18 ˚A. The samples were illuminated with a 20 mm footprint along the beam and used a first angle of incidence of 0.6◦ with collimation slit settings of 1.90 mm and 0.14 mm and a second angle of incidence of 2.7◦ with slit settings of 8.56 mm and 0.62 mm. Counting time for each angle was 2400 s and 5400 s, respectively.

Collected data were reduced within the refnx package [\[35\]](#page-12-1) where the reflectometry data are normalized to a direct beam measurement that is taken over the same collimation conditions. ToF is converted to wavelength by using the de Broglie equation, corrected for detector efficiency, joining the two data sets for each angle of

incidence at an appropriate overlap region, and scaling the data so that the critical edge is equal to a reflectivity of one. The final data are recorded as momentum transfer, Q (in units of ˚A −1 ), reflectivity, R, its standard deviation, dR, and the instrument resolution, dQ. The momentum transfer, Q, which is the scattering vector, is calculated as,

$$Q = \frac{(4\pi\sin\theta)}{\lambda},\tag{4}$$

where θ is the angle of incidence and λ is the neutron wavelength. All data were collected in 'event mode' where each neutron has its position on the detector and its time recorded, allowing data to be reduced to desired time intervals post-data collection. Data for the 0.6◦ angle of incidence were additionally reduced as described above, but also at 600 s time intervals, over the 2400 s to determine any time dependent changes over a given temperature.

Data were also fitted within the refnx software package [\[35\]](#page-12-1). In the models used the samples are divided into a series of layers, defined in terms of thickness ( ˚A), roughness (in units of ˚A) and scattering length density (SLD, ρ) (in units of ×10<sup>−</sup><sup>6</sup> ˚A −2 ). As no enrichment was performed, the natural isotopic abundance was assumed in the current work, in which case the SLD can be determined from the standard mass density and molar mass of each component:

$$
\rho\_N = N\_A \sum\_i \frac{p\_i}{A\_i} < b\_i >,\tag{5}
$$

where N<sup>A</sup> is Avogadro's constant, p<sup>i</sup> is physical density, Ai is atomic mass, and < b<sup>i</sup> > is the nuclear scattering length for component i averaged over its natural isotope composition. Each parameter in each layer is varied within physically relevant bounds using a differential evolution algorithm, until the fitted data match the experimental data closely, by minimizing χ 2 through a least-squares-curve-fitting method or using Bayesian fitting methods. Prior probability is the probability distribution function for a parameter, θ, given the pre-existing knowledge of a system. The posterior probability is given as p(θ|D, I), for a system, I, and observed data, D, and it is consistent with the prior probability and the observed data. It is similar to a least-squares analysis confidence interval. Once a good fit is achieved through the use of the differential evaluation algorithm, a Markov-chain Monte Carlo (MCMC) simulation is performed to examine the posterior probability distribution of parameters, and to assess a family of fits [\[35\]](#page-12-1). The output of the MCMC simulation produces the distribution, spread and covariance of the reflectometry parameters through resampling and refitting from randomized starting points repeatedly as explained in [\[36\]](#page-12-2). For the MCMC analysis 3000 steps (with 500 nburn and 500 equilibration), 200 walkers, and 500 thinning yielded a total of 800 fits. The final parameters presented are the median values from the distribution of all the fits with an uncertainty, that is half of the 15.87 to 84.13 percentile range.

# C. High temperature annealing

The sample was placed into an enclosed cylindrical high temperature (maximum temperature 1600◦C) vacuum furnace originally designed at Institut Laue-Langevin (ILL), France. Heat is generated in the furnace by passing a 300 A current through the core of the furnace that has a thin niobium element, the centre of which contains the sample. The furnace walls contain eight thin layers of niobium radiation shields for maintaining a uniform temperature at the sample [\[37\]](#page-12-3). The outer aluminum walls of the furnace are highly transparent to neutrons. The neutron transmission through the furnace exceeded 85%, within the usable wavelength. Base pressures of 10−<sup>7</sup> to 10−<sup>6</sup> mbar were achieved using a turbomolecular pump that pumped the sample volume. A vertical probe stick was used to mount the sample, on a stainless steel (grade 304) holder, which was able to withstand very high temperatures. Variable temperature insitu neutron reflectometry measurements were conducted with this controlled annealing process, with the temperatures ranging from room temperature to 800◦C.

# D. Thin film characterization XRD

The crystal structures of both un-annealed and annealed samples were analyzed at room temperature using a Rigaku Smartlab II diffractometer with CuKα radiation, λ = 1.540598 ˚A. The system employs a 2θ/ω geometry. Data were collected from 2θ = 3◦ to 80◦ with a step of 0.01◦ .

# E. Magnetization measurements

DC magnetization measurements were performed using a vibrating sample magnetometer (VSM) inserted in a physical property measurement system (PPMS) from Quantum Design. The temperature dependence of the sample magnetic moment m(T) was measured using the zero-field cooling (ZFC) method at different magnetic fields up to 80 kOe. Also, the isothermal magnetization curves, (m − H), were measured for temperatures over a range from 3 K to 15 K, which are well below to slightly above the superconducting transition temperature, Tc. All the magnetization measurements in this study were performed with the magnetic field aligned perpendicular to the sample's surface.

After the neutron reflectometry measurements, in-situ annealed samples were diced into small pieces (∼3 mm<sup>2</sup> ) and were used for the magnetization measurements. The sample was mounted on to the VSM sample holder, which has a small brass trough with quartz braces for holding the sample. The sample was sealed in place with Teflon tape after placing it between the quartz braces. The VSM holder is further mounted on to the probe, which was inserted into the VSM for offset measurements to specify the sample position in the VSM, and for subsequent measurements. Here, the ZFC is achieved by rapidly cooling the sample at the rate of 20 K/min in zero magnetic field from a temperature of 300 K down to 3 K. Constant magnetic fields were applied (100, 500, 1000, and 10000 Oe) perpendicular to the sample's surface and the temperature was increased at the rate of ∼0.9 K/min from 3 K to 30 K, for the m(T) measurements. When the temperature reached 30 K for each of the measurements, the field strength was set to zero, then the sample was cooled down to 3 K and the measurement was repeated for the next field strength. Similarly, for each isothermal magnetization measurement, the sample was cooled down to 3K at zero field strength, and the field was swept from 0 to 80 kOe at the rate of ∼1000 Oe/min.

Henceforth in this paper, 'Sample A' will refer to the sample in-situ annealed during the neutron beam reflectometry measurements at 750 ◦C, while 'Sample B' will denote the sample annealed at 800 ◦C.

# F. Thin film characterization EDX/TEM

To characterize the samples after the in-situ annealing, transmission electron microscopy (TEM) was employed. To do this, the sample surfaces were first coated with a 30 nm carbon layer. Following this, the area of interest on the surface (20 µm<sup>2</sup> ) was protected by first depositing a thin layer of platinum (Pt) film using an electronbeam, followed by a thicker layer using a focused ion beam (FIB) system. Cross-sectional lamellae (∼100 nm) were then prepared utilizing a 30 keV focused ion beam through a lift-off technique, followed by a final etching with a 5 keV gallium (Ga) FIB to minimize any surface damage. Both cross-sectional TEM and high-angle annular dark-field scanning transmission electron microscopy (HAADF-STEM) images were acquired in scanning electron microscopy (SEM) mode using an FEI Tecnai TF20 TEM operating at 200 kV. The stoichiometry of the thin film layers in the samples was further analyzed using energy dispersive X-ray spectroscopy (EDX) with an ∼1 nm diameter probe beam (EDAX Apollo XLT 2 detector).

# III. RESULTS AND DISCUSSION

# A. Reflectometry results

Time-resolved reflectivity patterns were obtained at multiple temperatures for the two samples each heated to a maximum temperature of 750 ◦C and 800 ◦C respectively.

Chemical reactions at the SiO2/V interface consume Si and O, leading to a modified SiO<sup>2</sup> layer thickness, and enabling the growth of secondary VO<sup>x</sup> and V3Si phases. The time-dependency of the neutron reflectivity patterns at different temperatures can be used to identify the start, end and transition rate of the chemical reaction. The main observables in the patterns are the high frequency Kiessig fringes associated with the thick oxide

![](_page_3_Figure_9.jpeg)

**Caption:** Figure 3 details a neutron reflectivity analysis obtained at 750°C, with a depiction of the SLD profile. This figure captures changes as the chemical profile adapts over several hours of reaction time, highlighting significant adjustments in the SiO2 and V2O3 layer thicknesses. The data elucidate early-stage growth phenomena relevant to the diffusion processes studied in these experiments.


<span id="page-3-0"></span>FIG. 1. Time-dependency of the neutron reflectivity experimental data for Sample B at a) 600 ◦C, b) 700 ◦C and c) 800 ◦C. The points are the experimental data, and the solid lines are the fits. t<sup>0</sup> is defined as the time since the target temperature has been reached, alignment has been achieved and the isothermal run has started. The inset shows the silica thickness with ∆t being the time since the furnace exceeded 600 ◦C.

layers (initially SiO2). As the reflectometry patterns are, to a first approximation, the 1D Fourier transforms of the real-space atomic (nuclear) structure, the Kiessig fringe

.

periodicity can be used to estimate the layer thickness (d) via the relation

<span id="page-4-0"></span>
$$d = \frac{2\pi}{\Delta Q} \tag{6}$$

where ∆Q is the spacing between two fringe maxima. Figure [1](#page-3-0) a) shows multiple reflectometry patterns, each measured over ten minutes, at 600 ◦C where ∆Q has been labeled. The fringe spacing at 600 ◦C is ∆Q ≈ 0.03 nm−<sup>1</sup> which is consistent with a SiO<sup>2</sup> layer of 200 nm. It is well known that the latter approximation is not exact, due to dynamic scattering effects which are stronger at low Q. Using the full-model (described later, which includes dynamic scattering) the determined starting thickness is 189 ± 5 nm. In the figure, the points are the experimental data and the solid lines are fits to the data using the SLD model described in the following section. The variation of the fringe spacing (∆Q) provides a good first indication of the variation of the oxide layers during the growth. For example, Figure [1](#page-3-0) b and Figure [1](#page-3-0) c show the time dependency for 700 ◦C and 800 ◦C over a series of ten-minute intervals. The data for 700 ◦C show a clear time dependency, indicating interfacial chemical reactions which lead to a modified SiO<sup>2</sup> thickness that shifts the high frequency fringes.

In contrast the patterns at 600 ◦C and 800 ◦C are independent of time. The inset shows the fitted SiO<sup>2</sup> thickness. At 600 ◦C, the temperature was too low for the reaction to proceed on observable timescales so the SiO<sup>2</sup> thickness is effectively constant (189 ± 5 nm) over the first hour. Between 700 and 800 ◦C, the SiO<sup>2</sup> thickness dropped dramatically. By 800 ◦C, the SiO<sup>2</sup> layer was fully consumed at the start of the measurement time, and the main reaction stopped. Whilst some high frequency fringes are present in the 800 ◦C data, these are heavily damped in comparison with those at 600 ◦C, and also appear at different Q-points. Detailed modelling in the next section shows these are not from the SiO<sup>2</sup> but instead mainly arise from the rough, vanadium oxide surface layer that is formed by the reaction.

By modelling the full intensity of the reflection patterns as a function of Q including dynamic scattering effects, the scattering model also enables measurement of structural parameters, including interface roughness (diffusion profiles). The model is also able to discriminate between the thickness of the SiO<sup>2</sup> and VO<sup>x</sup> layers which is not possible by simply applying Equation [6.](#page-4-0) The fitting procedure provides quantitative values for the layer thicknesses during the growth, and therefore the reaction kinetics, at different times and temperatures. In order to fit the evolution of the thin film reflectivity data, a self-consistent chemical model was developed that imposed constraints on the scattering length density (SLD) profile. Constrained fitting was necessary because direct Fourier inversion was not possible, as is generally the case in neutron scattering, since the observed intensity signal is the modulus of the neutron wave function, |ψ| 2 , and phase information is lost. A model is therefore needed to fit the data in this system.

Modeling is complicated, however, by the very low coherent neutron scattering cross-section of vanadium metal film there is a low contrast compared to those of the oxide layers. Additionally, while it was possible to measure to high Q for the time-independent datasets below 600 ◦C and above 800 ◦C, only low Q data could be reliably fitted at 700–750 ◦C due to the sample kinetics which necessitated short integration times, such that only the high intensity regions of the reflectivity curve could be observed with acceptable statistical uncertainty. To yield reliable fits using a self-consistent chemical model, the initial layer thicknesses were determined prior to the reaction, and a mass-balanced model was implemented to fit the oxide and V3Si layer thicknesses during the reaction. The model assumes that the key chemical reactions are of the general form:

$$xSiO\_2 + yV \rightarrow xV\_3Si + (y-3x)VO\_{2x} \tag{7}$$

where y > x. Based on trial fitting, and consistent with the XRD results, we found that the specific reaction (x=3, y= 13) describes the data very well at most temperatures, via the reaction:

$$3SiO\_2 + 13V \rightarrow 3V\_3Si + 2V\_2O\_3 \tag{8}$$

The method assumes that the layer volume, determined by film area and thickness (dSiO2) of the SiO<sup>2</sup> is related to the amount of SiO<sup>2</sup> used in the reaction. The number of SiO<sup>2</sup> units used in the reaction per unit area is:

$$
\Delta x = (d\_{SiO\_2} - d\_{SiO\_2}') \times N\_{Si}^{SiO\_2} \tag{9}
$$

where dSiO<sup>2</sup> = dSiO2(t, T) is the thickness of the layer at a temperature (T) and time (t); d ′ is the thickness at room temperature at t=0, before the reaction. NSiO<sup>2</sup> Si is the number density of Si atoms per volume which is determined from the mass density ρSiO<sup>2</sup> and molar mass M such that NSiO<sup>2</sup> Si = ρSiO2/(NAMSiO2). Once ∆x is determined, this then fixes the amount of V3Si and V2O<sup>3</sup> produced via the chemical reaction, and this therefore determines the thicknesses of the V3Si and V2O<sup>3</sup> layers via the relations:

$$d\_{V\_3Si} = \Delta x / N\_{Si}^{V3Si} \tag{10}$$

$$d\_{V\_2O\_3} = 2 \times \Delta x / N\_O^{V\_2O\_3} \tag{11}$$

where N<sup>V</sup> <sup>3</sup>Si Si is the number density of Si atoms in the V3Si phase, and N<sup>V</sup> <sup>2</sup>O<sup>3</sup> <sup>O</sup> is the number density of oxygen atoms in the V2O<sup>3</sup> phase. Finally, to balance the chemical reaction, the thickness of any left-over (unreacted) vanadium is:

$$d\_V = d\_V' - (3\Delta x + \frac{4}{3}\Delta x) \times N\_V^V = d\_V' - N\_V^V \frac{13}{3} \Delta x \tag{12}$$

where d ′ V is the starting V thickness prior to the reaction; N<sup>V</sup> V is the number density of vanadium atoms in the metallic vanadium phase and the (3) appears because the

|       | Compound Mass density | X-ray SLD               | Neutron SLD             |
|-------|-----------------------|-------------------------|-------------------------|
|       | (g/cm3<br>)           | (10−6<br>/ ˚A<br>2<br>) | (10−6<br>/ ˚A<br>2<br>) |
| Si    | 2.33                  | 20.07                   | 2.07                    |
| SiO2  | 2.2                   | 18.86                   | 3.47                    |
| V     | 6.11                  | 46.97                   | -0.32                   |
| V2O5  | 3.35                  | 27.01                   | 3.12                    |
| V2O3  | 4.87                  | 38.77                   | 3.23                    |
| VO2   | 4.65                  | 37.27                   | 3.77                    |
| V3Si  | 5.77                  | 36.12                   | 0.54                    |
| V5Si3 | 5.29                  | 41.89                   | 0.96                    |
| VSi2  | 4.42                  | 36.12                   | 1.95                    |

<span id="page-5-1"></span>TABLE I. Theoretical scattering length densities

ratio V:Si in V3Si is 3:1, and the factor 4/3 appears because the ratio V:O in V2O<sup>3</sup> is 2:3, multiplied by 2 as each reacted SiO<sup>2</sup> formula unit releases 2 oxygen.

Using this chemical model to guide the fitting procedure, the four thickness parameters in the reflectivity model (dSiO<sup>2</sup> , d<sup>V</sup> , dV3Si and dV2O<sup>3</sup> ) are effectively constrained so that they all depend on a single free parameter (∆x) which greatly improves the reliability of the fitting. One additional variable was introduced to model a thin vanadium native oxide layer caused from ambient exposure prior to heating. However this was constrained to a thickness of < 2 nm and had a negligible effect on the fits overall.

To test the accuracy of the above model before, after and during the reactions, the data were fitted at multiple temperatures. Initially the time-independent dataset at 100 ◦C was fitted to determine the virgin state of the sample, with dSiO<sup>2</sup> = 195 ± 5 nm at ∆x = 0. Figure [2](#page-5-0) shows the fitted reflectivity patterns and SLD profile at 600 ◦C. This fit shows that the V and SiO<sup>2</sup> films are both present as individual layers at 600 ◦C, similar to at roomtemperature, with minimal reaction having occurred.

In contrast, Figure [3](#page-6-0) shows the reflectivity patterns and SLD profile at 750 ◦C indicating a modified chemical profile after several hours of reaction. Figure [4](#page-6-1) shows the fitted reflectivity patterns and SLD profile at 800 ◦C after the reaction had run to conclusion over several hours. In all cases the model produces an excellent description of the data. The SLD profiles also provide insights into the diffusion and roughness parameters at each temperature. For comparison, the theoretical SLDs for the single phases, assuming the bulk densities, are reported in Table [I.](#page-5-1) An interesting situation was observed near the midpoint of the reaction at 750 ◦C, as shown in Figure [3](#page-6-0) where the thickness of the SiO<sup>2</sup> and V2O<sup>3</sup> both became comparable and relatively thin films (60 nm) formed. This is fully described by the model above. After the reaction has completed at 800 ◦C, the SiO<sup>2</sup> layer was fully consumed, and best-fit to Sample B yielded a thick V3Si layer (265 ± 5 nm) and thick oxide layer (170 ± 5 nm,), consistent with the post-heat treatment TEM data presented later.

![](_page_5_Figure_7.jpeg)

**Caption:** Figure 2 shows neutron reflectivity data, with a logarithmic-linear representation by plotting reflectivity (R) vs. momentum transfer (Q) from in-situ annealing at 600°C. The data, along with the best-fit scattering length density (SLD) profile, reveal minimal changes under this temperature, indicating a lack of significant reaction progress. The descriptions provide baseline metrics for comparison against higher temperature transformations documented in subsequent figures.


<span id="page-5-0"></span>FIG. 2. a) Logarithmic-linear representation of the neutron reflectivity experimental data as a function of Q, obtained during the in-situ annealing of V samples at a temperature of 600 ◦C. In the plot, the solid circles represent the experimental data, and the red solid lines fits to the data. b) The best-fit SLD profile.

The approximate growth rates observed during the experiment, matching the time regions listed in Figure [1](#page-3-0) are reported in Table [II.](#page-6-2) Here we have used a linear fit to the thicknesses observed over these measured time windows. The table indicates the rate of the depletion (negative thickness "gain") of the SiO2, and the associated positive thickness-gain of the other layers. It is worth stating that the growth rate of a diffusion-mediated reaction is generally non-linear over long time spans, and the rate depends on the amount of SiO<sup>2</sup> which evolves over time. Thus, the latter rates are simply an indication of the growth observed at particular points during the experiment, provided as a guide for future work.

In general, while the chemical model describes the relative amounts of the different phases, there is still the question of whether those phases form discrete layers and the spatial position of each layer. Reflectivity addresses this question as trial-fitting showed only certain

![](_page_6_Figure_1.jpeg)

**Caption:** Figure 3 presents neutron reflectivity data collected during in-situ annealing at 750°C, translating changes in reflectivity patterns into SLD profiles. Such profiling enables real-time monitoring of structural shifts, elucidating interfacial reactions and layer development characteristic of various silicide and oxide phase formations within the studied temperature regime.


<span id="page-6-0"></span>FIG. 3. a) Logarithmic-linear representation of the neutron reflectivity experimental data as a function of Q, obtained during the in-situ annealing of V samples at a temperature of 750 ◦C. In the plot, the solid circles represent the experimental data, and the red solid lines fits to the data. b) The best-fit SLD profile.

SLD profiles fit the data well. In general, for T > 750 ◦C the reflectivity profiles only could be well-described with the profile where the vanadium oxide formed on top of the silicide as a discrete layer. This also concurs with the TEM data. The interpretation is that the different surface energy/surface tension of the two phases favours the latter configuration. At an earlier stage of the growth (650 ◦C and early times at 700 ◦C) there is some evidence, however, for mixed layer formation of a silicide/vanadium oxide above the SiO<sup>2</sup> interface, which warrants investigation in future work.

# B. XRD Analysis

X-ray diffraction (XRD) patterns for three samples, an un-annealed sample (Preheat), Sample A annealed at

![](_page_6_Figure_6.jpeg)

**Caption:** Figure 4 illustrates neutron reflectivity measurements conducted at 800°C, showing significant alteration in film structure, as interpreted through the SLD profile. Enhanced transformation attributes are noted, including complete consumption of the initial SiO2 layer, underscoring the extensive V3Si layer development and associated vanadium oxide formations. This transformation analysis validates findings from XRD and TEM characterizations.


<span id="page-6-1"></span>FIG. 4. a) Logarithmic-linear representation of the neutron reflectivity experimental data as a function of Q, obtained during the in-situ annealing of V samples at a temperature of 800 ◦C. In the plot, the solid circles represent the experimental data, and the red solid lines fits to the data. b) The best-fit SLD profile.

TABLE II. Observed film growth rates at different temperatures

<span id="page-6-2"></span>

| Temperature | SiO2  | V3Si                       | V2O3 |
|-------------|-------|----------------------------|------|
| ◦C)<br>(    |       | (nm/min) (nm/min) (nm/min) |      |
| 600         | 0     | 0                          | 0    |
| 700         | -0.11 | 0.15                       | 0.10 |
| 750         | -0.39 | 0.55                       | 0.34 |

750 ◦C, and Sample B annealed at 800 ◦C, are presented in Figure [5.](#page-7-0) The un-annealed sample displays diffraction peaks that are characteristic of metallic vanadium deposited on a silicon substrate that has a thermally oxidized SiO<sup>2</sup> layer. The diffraction pattern for Sample A, which underwent in-situ annealing during the neutron reflectometry study, confirms the formation of V3Si by
exhibiting all major peaks associated with this phase. Also, peaks indicative of V2O<sup>3</sup> and VO are observed. In contrast, the XRD pattern for Sample B, also prepared similarly, but annealed at 800 ◦C shows a predominance of V3Si peaks with a noticeable reduction in the intensity of the oxide peaks compared to the oxide peaks for Sample A. It is also observed that the SiO<sup>2</sup> peaks have disappeared, and an additional V5Si<sup>3</sup> phase has formed, indicating that the SiO<sup>2</sup> bottom layer was completely consumed during the reaction, exposing the underlying silicon substrate for further reaction. The reaction of remaining metallic vanadium with the silicon substrate is suggested to be the cause of the formation of this additional phase.

![](0__page_7_Figure_1.jpeg)

**Caption:** Figure 5 provides X-ray diffraction (XRD) patterns for an un-annealed sample (Preheat), along with Samples A and B, annealed at 750°C and 800°C respectively. The patterns reveal differences in V3Si peak intensity and presence of oxide phases, with a noticeable reduction in these peaks for Sample B due to the complete consumption of the SiO2 layer. This phase transformation and consumption of SiO2 indicate significant restructuring, a critical insight into the thermal treatment effects and V-Si reactions.


FIG. 5. X-ray diffraction patterns of an un-annealed sample (Preheat), sample A (750 ◦C) and sample B (800 ◦C)

## <span id="page-7-1"></span>C. TEM Analysis

Figure [6](#page-7-0) shows the cross-sectional TEM images, HAADF-STEM images and EDAX mapping of sample A (a-e) and sample B (f-j). Both samples exhibit a continuous V3Si layer with an oxide layer on top, formed due to the diffusion-controlled growth process. The thicknesses of the V3Si layers are approximately 90 nm for sample A, and 280 nm for sample B. The vanadium oxide layers on top are 320 nm thick for sample A, and 180 nm for sample B. It is to be noted that the bottom SiO<sup>2</sup> layer in sample B is fully consumed during the annealing process, leaving the V3Si layer directly on the Si substrate. This result is consistent with the reflectometry model presented in [III](#page-3-0)[.A](#page-3-1) for the sample B, which further validates its accuracy. Careful analysis of the V3Si morphology reveals a columnar grain structure in both the samples, characterized by elongated vertical grains growing perpendicular to the plane of the film surface.

The role of the columnar grains can be understood from the flux pinning mechanism at the grain boundaries. Columnar grain growths are reported in A15 superconductors like V3Ga [\[38\]](#page-12-0) and Nb3Sn [\[39\]](#page-12-1), where the superconducting phase formation is through a diffusion process. The average widths of grains are estimated to be 35 nm for sample A, and 60 nm for sample B in this study. It is clearly seen that the grain size increases with the annealing temperature from 750 ◦C to 800 ◦C, and the columnar grains become more prominent with the increase in temperature.

![](0__page_7_Figure_8.jpeg)

**Caption:** Figure 6 captures cross-sectional TEM images for Samples A and B, showing their respective V3Si silicide layers, columnar grain patterns, and oxides. High-angle annular dark-field STEM images (HAADF-STEM) and EDAX mappings elucidate compositional and grain-oriented differences correlating to annealing treatments. These structural insights aid in understanding diffusion-controlled growth and grain pinning mechanisms crucial for superconductor performance.


<span id="page-7-0"></span>FIG. 6. Cross-sectional TEM images of Sample A (a) and Sample B (f) reveal the V3Si silicide layer, its interfaces, the columnar grain structures, and the oxide layer atop the V3Si. HAADF-STEM images acquired in SEM for Sample A (b) and Sample B (g) further illustrate the structural composition. EDAX elemental mapping for oxygen, silicon, and vanadium, derived from the HAADF images, as shown for Sample A (ce) and Sample B (h-j), highlight the elemental distribution across the samples.

## D. Magnetization results

Figure [7](#page-8-0) shows the dependence of the DC magnetic moment on temperature, for the in-situ annealed samples A and B, respectively. Each sample was ZFC to 3 K, and then warmed under constant magnetic fields of 100, 500, 1000, and 10,000 Oe, during which the magnetic moment (m) was measured. For both the samples, the onset of superconductivity, Tc(onset) was determined to be ∼13 K. In this study, the onset temperature is considered to be the critical temperature. There are studies that considered the midpoint of the diamagnetic transition as Tc; but this approach has a few challenges [\[40\]](#page-12-2). It should be noted that while bulk single crystal V3Si has a T<sup>c</sup> of 17.1 K, that of V3Si thin films varies based on growth conditions such as the substrate material [\[22,](#page-11-0) [41\]](#page-12-3), deposition method [\[24,](#page-11-1) [42\]](#page-12-4), and annealing temperature [\[43\]](#page-12-5). V3Si films are commonly developed on substrates like silicon, quartz, graphite, magnesium oxide, sapphire, copper, and niobium. In reported results, the highest T<sup>c</sup> in V3Si thin films (<1 µm thick) ranges from 10 to 16 K, depending on the growth conditions. Such a decrease in the T<sup>c</sup> of thin films compared to the bulk, could be explained by the proximity effect induced by the substrate on the V3Si films [\[44\]](#page-12-6).

The magnetization in the superconducting state is saturated for both the samples at the lowest measured temperature. Below Tc, the diamagnetism is enhanced due to screening supercurrents. There is a decrease in flux expulsion as external magnetic field strength increases, which is a characteristic of the mixed state for type-II superconductors [\[45\]](#page-12-7), and this results in the suppression of a superconducting response. Hence, there is a significant reduction in superconducting volume that is observed with the increase in magnetic field. Notably, in sample B at 10 kOe, a paramagnetic Meissner effect or Wohlleben effect was observed [\[46,](#page-12-8) [47\]](#page-12-9). In both samples A and B, diamagnetic saturation is observed only below 5 K. This broad transition in the magnetization curves could be a result of strong vortex pinning, thermal fluctuations leading to flux creep and inhomogeneous pinning distributions in the film [\[48\]](#page-12-10). This indicates that there is irreversibility in the magnetization.

![](0__page_8_Figure_2.jpeg)

**Caption:** Figure 7 illustrates temperature-dependent magnetization measurements under various magnetic fields for both Sample A (a) and Sample B (b). Each graph presents data obtained while warming samples zero-field-cooled to 3 K, showing Tc(onset) around 13 K. The curves highlight essential thermal and field-dependent behavior of the superconducting state, emphasizing differences in pinning efficiency, as inferred from magnetization behaviors such as the Paramagnetic Meissner Effect and other reversible phenomena.


<span id="page-8-0"></span>FIG. 7. Temperature-dependent magnetization measured at various magnetic fieeds for (a) sample A and (b) sample B.

Figure [8](#page-9-0) illustrates the isothermal DC magnetic mo-

ment (m) as a function of the magnetic field Hext (measured in Oersteds (kOe), for samples A and B at temperatures ranging from 3 K to 15 K, in increments of 1 K. A complete magnetic field cycle, starting from 0 to 80 kOe, returning to 0, then dropping to -80 kOe, and finally reverting to 0, is recorded in the data. It is seen from the isothermal magnetic field dependence of the magnetic moment in both samples A and B that below Tc, irreversible magnetization hysteresis is observed. As the temperature increases, the width of the magnetic hysteresis loop decreases, resulting in the reduction of the loop area up to Tc. Since the screening currents are unable to fully expel the trapped flux lines while the magnetic field is decreased from 80 kOe to 0 Oe, the irreversible magnetic moment results. Even when the external field is reduced to zero, this trapping causes a residual magnetic moment.

The findings in this study are presented in terms of m, due to the challenges in accurately determining the volumes of the superconducting material in the samples. But if the volume of the superconducting phase of the films were known, the results could also be expressed in terms of magnetization (M = m/V). The observed hysteresis in the magnetization confirms that the V3Si film formed by this preparation route is a type II superconductor [\[49\]](#page-12-11). The diamagnetic response of the V3Si layer becomes evident at the start of each trace below Tc, as the field increases from 0 to 80 kOe, which is consistent with the m(T) measurement results. The magnetic moment in this region can be expressed as −m(Hext).

From Figure [7](#page-8-0) and [8](#page-9-0) it can be understood that below Tc, the magnitude of m is larger for sample A than that of sample B. This shows that sample A possess a higher superconducting volume fraction compared sample B, though the V3Si thickness in sample B is larger compared to sample A. Also, in Figure [8](#page-9-0) (b) for sample B, the magnetic moment curve starts to show a downward bend in high fields close to the critical field, with the increase in temperature. This downward bending of the magnetic moment is possibly due to a weaker superconducting volume fraction formed due to annealing at 800 ◦C. This allows the diamagnetic response from the underlying silicon substrate to become more pronounced due to the depletion of SiO<sup>2</sup> layer. The thicknesses of the V3Si layers and the depletion of the SiO<sup>2</sup> layer in sample B are confirmed through the reflectometry measurements and the cross-sectional TEM analysis discussed in sections [III.](#page-3-0)[A](#page-3-1) and [III.](#page-3-0)[C](#page-7-1) respectively.

Figure [9](#page-9-1) shows the temperature dependence of both critical fields, Bc1 and Bc2, for sample A. From the plot of magnetic moment m versus H shown in Figure [8\(](#page-9-0)a), the lower critical field, Bc1(0) = µ0Hc1, and the upper critical field, Bc2(0) = µ0Hc2, are derived. For type-II superconductors, Bc1 is commonly defined as a field at which the initial diamagnetic response on the m–H curve starts to deviate perfectly from the linear behaviour. As the temperature approaches Tc, Bc1 becomes exceedingly small, making its detection near the critical temperature almost impractical due to very subtle deviation, espe-

![](0__page_9_Figure_0.jpeg)

**Caption:** Figure 8 shows the DC magnetic moment curves as a function of the applied external field (Hext) for temperatures ranging 3–15 K, with plots for both Sample A (a) and Sample B (b). These curves illustrate significant findings, such as the lower critical field (Bc1) for Sample A, which is distinctly characterized below the transition temperature (Tc). Both samples demonstrate irreversible magnetization hysteresis below their respective Tc, a behavior typical of type-II superconductors. Insets provide virgin magnetic moment curves at 3 K, highlighting critical transitions near Bc1. These observations reveal Sample A's higher superconducting volume fraction compared to Sample B, which is confirmed through corresponding reflectometry data and cross-sectional TEM analyses.


<span id="page-9-0"></span>FIG. 8. The DC magnetic moment curves as a function of the applied external field Hext in the temperature range of 3 – 15 K for both (a) sample A and (b) sample B are depicted. The insets in both figures display the virgin magnetic moment curves at 3 K. For sample A, the lower critical field (Bc1) is distinctly defined at temperatures significantly below the transition temperature (Tc). Both samples demonstrate irreversible magnetization hysteresis behavior below Tc. A magnetization hysteresis is observed up to 4 T even at 10 K, nearing Tc.

cially in superconductors with high critical current densities [\[40\]](#page-12-2). Figure [9\(](#page-9-1)a) shows the temperature dependence of Bc1 (in blue solid circle symbol) with respect to temperature, and the inset depicts the m–H curves measured at low temperatures where the deviation from the linear behaviour of the lower critical field is clearly observable. The olive solid line represents a linear fit to the Bc1 data points at low temperatures, and the slope -dHc1/dT is calculated to be −0.11 T/K, from which Bc1(0) ≈0.65 T, can be estimated.

![](0__page_9_Figure_3.jpeg)

**Caption:** Figure 9 displays (a) lower critical field (Bc1) and (b) upper critical field (Bc2) temperature dependencies for Sample A. Bc1(T) is derived from the slope deviation of m-H curves, indicating superconducting transition aspects, while Bc2(T) fits involve empirical parameters. These insights into critical fields improve understanding of Ginzburg-Landau behavior and coherence length calculations for type-II superconductors.


<span id="page-9-1"></span>FIG. 9. (a) Temperature dependence of lower critical field, Bc1, and (b) upper critical field, Bc2, for the sample A

The temperature dependence of Bc2 (in blue solid circle symbol) for sample A is shown in Figure [9\(](#page-9-1)b). Bc2 at each temperature is estimated from the m–H plot where the transition from diamagnetic to paramagnetic state occurs, at the point where the curve intersects m=0. The empirical formula [\[50\]](#page-12-12), Bc2(T) = Bc2(0)(1−t 2 )/(1+ t 2 ), where t = T /Tc, is employed to fit the experimental data. The fit is depicted using the solid olive curve. From the GL fit, Bc2(0) was determined as 7.38 T.

Although the GL equations are expected to be applicable near the Tc; in this study, it is observed that they accurately hold even to temperatures as low as ∼0.2 Tc, which is characteristic of conventional BCS superconductors [\[51\]](#page-12-13). The GL fit shown in Figure [9](#page-9-1) (b) shows a perfect linear relationship with temperature from T<sup>c</sup> to ∼0.5 Tc, which aligns with the behavior typical of superconductors with a high Ginzburg-Landau parameter κ [\[52\]](#page-12-14).

Assuming that the breakdown of superconductivity at Bc2 is through the orbital pair breaking effect [\[53\]](#page-12-15), the superconducting coherence length ξ and penetration depth λ can be calculated using the GL equations, µ0Hc2 = ϕ0/2πξ<sup>2</sup> and µ0Hc1 = ϕ0/2πλ<sup>2</sup> , where ϕ<sup>0</sup> denotes the flux quantum.

The coherence length is estimated to be 6.7 nm and the penetration depth to be 22.5 nm, from the values of Bc1 = 0.65 T and Bc2 = 7.38 T. The Ginzburg-Landau parameter, κ = λ/ξ = 3.35, is then calculated. It has been reported that Bc2 = 25 T, ξ = 4.2 nm [\[54\]](#page-12-16) and Bc2 = 22 T, ξ = 3.8 nm, from the measurements of such parameters in bulk V3Si single crystals [\[55\]](#page-12-17). However values similar to the measurements in this study, Bc2 = 8.1 T and ξ = 6.27 nm, were recently reported [\[56\]](#page-12-18) from low temperature tunnelling experiments conducted on V3Si (100) single crystals, and it was proposed that the increase in coherence length could be due to multi-band s-wave superconductivity in V3Si. The sheet resistance of the films was measured using a 4-probe method at room temperature. Before annealing, the sheet resistance was 1.3 Ω/□ (ohm/square). After annealing, it increased to 3 Ω/□ for sample A and 5.8 Ω/□ for sample B, which shows that the sheet resistance increases with the increase in annealing temperature.

The analysis of the results from the magnetization measurements in the prepared polycrystalline V3Si film can be summarized as follows. It is a type-II, high-Tc, high-κ material in its superconducting state, known for its strong anisotropic properties [\[57\]](#page-12-19). Magnetic flux starts to penetrate the superconductor in distinct threads known as flux lines or vortices (fluxoids) when the superconductor is exposed to an external magnetic field that is stronger than the lower critical field Bc1, but weaker than the upper critical field Bc2. Each vortex carries one quantum of magnetic flux (ϕ0) and is surrounded by a circulating supercurrent. At the core of each vortex, the superconducting order parameter is significantly reduced, thereby effectively eliminating superconductivity locally within the core [\[58,](#page-12-20) [59\]](#page-12-21).

The vortex pinning strength of a type II superconductor [\[60\]](#page-12-22) is an important parameter since it ensures that the zero electrical resistance is maintained under the influence of an external magnetic field. A high pinning efficiency restricts the movement of vortices when a current is applied, thereby enhancing the critical current density Jc. Hence energy dissipation is minimized, there is stability in varying magnetic fields and superconducting properties are preserved.

The fluxoid spacing, a<sup>0</sup> = q Φ<sup>0</sup> B , where ϕ<sup>0</sup> is the magnetic flux quantum, refers to the average distance between adjacent vortices in a type II superconductor in the mixed state. This spacing is dependent on the applied magnetic field strength. Thus, a large grain size compared to the small fluxoid spacing leads to lower pinning efficiency at high magnetic fields. However, the increase in the size of the grain boundaries compared to the grain size can likely improve or maintain pinning efficiency by compensating for the larger grain size. This is achieved by providing ample pinning sites across the material due to the extensive influence of the grain boundaries.

The vortex pinning strength of these films can be estimated from the grain boundary density, which is inversely proportional to the grain size. For a given magnetic field, fluxoid spacing can be calculated. For example, at 4 T, for sample A, with a grain size of ∼33 nm, a<sup>0</sup> is calculated as 22 nm, which shows a slight decrease in pinning efficiency. Similarly, for sample B, with a grain size of ∼60 nm, the pinning efficiency further decreases at 4 T.

In reversible type-II superconductors, with an increase in the external magnetic field, the vortex density also rises, leading the vortices to organize into a highly ordered structure called the Abrikosov lattice [\[45\]](#page-12-7), which minimizes the system's free energy. This lattice is generally characterized by an hexagonal configuration of vortices in high quality single crystal V3Si samples [\[61\]](#page-12-23). However, in polycrystalline V3Si thin films, the Abrikosov lattice is disordered due to the presence of strong pinning centers such as grain boundaries, defects and impurities. Hence the typical hexagonal vortex arrangement is disrupted resulting in a metastable state with vortices trapped inhomogeneously within the sample, which causes irreversible behaviour in the magnetization of the superconductor. The Abrikosov unit cell, which is the smallest repetitive unit in this lattice, has an area that decreases as the magnetic field strength increases, indicating an inverse relationship between the two [\[62\]](#page-12-24).

The vortex core size can be approximated based on the coherence length ξ, with the Ginzburg-Landau theory being applicable near the critical temperature, Tc. The approximate area of the decay of the fluxoid core is given by πξ<sup>2</sup> , while πλ<sup>2</sup> represents the area through which the magnetic field penetrates without disrupting superconductivity, persisting up to an upper critical field [\[63\]](#page-12-25). Bc2 is estimated where the average spacing between the centres of adjacent vortices approximates the size of their cores. At this point, the superconducting regions connecting the fluxoids start to overlap significantly, and finally diminish. When reaching Bc2, the vortex cores expand to such an extent that the continuous superconducting paths for current flow are disrupted, rendering the entire sample non-superconducting. This phenomenon supports the report of a smaller coherence length in bulk V3Si single crystals.

The calculations in this study indicate a slightly large coherence length, which might be due to the increased size of the vortex core at low magnetic fields. This will result in weaker pinning strength and hence the degradation of the superconducting properties. The lower Bc2 measured in this study compared to the bulk, is likely due to the larger coherence length calculated at zero field, since Bc2 varies inversely with ξ. Vortex interactions are generally repulsive because their circulating currents generate magnetic fields that interact. The vortex density increases with magnetic field strength, decreasing the average distance between vortices and increasing the repulsive force between them. The superconducting material reduces the overlap of the vortex cores by decreasing each vortex's core area, in order to minimise the system's energy. Hence, the core size reduces at low fields to sustain an energetically optimal vortex configuration.

In the dirty limit of the microscopic theory, it has been established that the size of the vortex core in a conventional s-wave superconductor reduces as the magnetic field intensifies [\[64\]](#page-12-26). This effect was previously observed in V3Si single crystals [\[65\]](#page-12-27). Therefore, the extension of Ginzburg-Landau theory developed by Abrikosov that forms the foundation of type II superconductivity details the formation of the Abrikosov flux line lattice and its superconducting order parameters.

## IV. CONCLUSIONS

In conclusion, detailed investigations of the phase transformation during vanadium silicide thin-film formation are reported. A chemical model is proposed to evaluate the time dependency of the phase formation during the thin-film growth at high temperatures. Magnetization measurements have been performed on the annealed samples to assess superconductivity and the superconducting order parameters of the film were explored. This study provides a methodology for optimizing the growth conditions for superconducting thin films, which is an advancement towards developing future superconducting quantum devices that can operate well above liquid helium temperature. One of the future research avenues in this work is to optimize the chemical model for the growth of thin superconducting films. Another area worth exploring is the proximity effect induced by the substrate on the superconducting properties of these V3Si films.

## V. ACKNOWLEDGMENTS

This work was performed in part at the Australian Nuclear Science and Technology Organization. We acknowledge the support of the Australian Centre for Neutron Scattering, ANSTO and the Australian Government through the National Collaborative Research Infrastructure Strategy (NCRIS), in supporting the neutron research infrastructure used in this work via ACNS proposal 17337. M. Bose is also supported in part by ARC grants (DP200103233 and CE170100012).

- [1] M. Kjaergaard, M. E. Schwartz, J. Braum¨uller, P. Krantz, J. I.-J. Wang, S. Gustavsson, and W. D. Oliver, Superconducting qubits: Current state of play, [Annual Review of Condensed Matter Physics](https://doi.org/https://doi.org/10.1146/annurev-conmatphys-031119-050605) 11, 369 [\(2020\).](https://doi.org/https://doi.org/10.1146/annurev-conmatphys-031119-050605)
- [2] G. Burkard, M. Gullans, X. Mi, and J. Petta, Superconductor–semiconductor hybrid-circuit quantum electrodynamics, [Nature Reviews Physics](https://doi.org/10.1038/s42254-019-0135-2) 2, 129 (2020).
- [3] S. Bravyi, O. Dial, J. M. Gambetta, D. Gil, and Z. Nazario, The future of quantum computing with superconducting qubits, [Journal of Applied Physics](https://doi.org/10.1063/5.0082975) 132, [160902 \(2022\).](https://doi.org/10.1063/5.0082975)
- [4] C. H. Yang, R. Leon, J. Hwang, A. Saraiva, T. Tanttu, W. Huang, J. Camirand Lemyre, K. W. Chan, K. Tan, F. E. Hudson, et al., Operation of a silicon quantum processor unit cell above one kelvin, Nature 580[, 350 \(2020\).](https://doi.org/10.1038/s41586-020-2171-6)
- [5] L. Petit, H. Eenink, M. Russ, W. Lawrie, N. Hendrickx, S. Philips, J. Clarke, L. Vandersypen, and M. Veldhorst, Universal quantum logic in hot silicon qubits, [Nature](https://doi.org/10.1038/s41586-020-2170-7) 580[, 355 \(2020\).](https://doi.org/10.1038/s41586-020-2170-7)
- [6] G. F. Hardy and J. K. Hulm, Superconducting silicides and germanides, Phys. Rev. 89[, 884 \(1953\).](https://doi.org/10.1103/PhysRev.89.884)
- [7] G. F. Hardy and J. K. Hulm, The superconductivity of some transition metal compounds, [Phys. Rev.](https://doi.org/10.1103/PhysRev.93.1004) 93, 1004 [\(1954\).](https://doi.org/10.1103/PhysRev.93.1004)
- [8] E. Saur, Experiments on the critical behavior of typeii superconductors in high magnetic fields, in [Physics of](https://doi.org/10.1007/978-1-4899-5508-1_24) [Solids in Intense Magnetic Fields: Lectures presented at](https://doi.org/10.1007/978-1-4899-5508-1_24) [the First Chania Conference held at Chania, Crete, July](https://doi.org/10.1007/978-1-4899-5508-1_24) [16–29, 1967](https://doi.org/10.1007/978-1-4899-5508-1_24) , edited by E. D. Haidemenakis (Springer US, Boston, MA, 1969) pp. 454–466.
- [9] E. S. Greiner and J. Mason, H., Preparation of single crystals of V3Si, [Journal of Applied Physics](https://doi.org/10.1063/1.1713170) 35, 3058 [\(1964\).](https://doi.org/10.1063/1.1713170)
- [10] C. G. Shull and F. A. Wedgwood, Neutron-diffraction studies of electron-spin pairing in superconducting V3Si, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.16.513) 16, 513 (1966).
- [11] B. W. Batterman and C. S. Barrett, Low-temperature structural transformation in V3Si, [Phys. Rev.](https://doi.org/10.1103/PhysRev.145.296) 145, 296 [\(1966\).](https://doi.org/10.1103/PhysRev.145.296)
- [12] L. R. Testardi and T. B. Bateman, Lattice instability of high-transition-temperature superconductors. II. singlecrystal V3Si results, Phys. Rev. 154[, 402 \(1967\).](https://doi.org/10.1103/PhysRev.154.402)
- [13] J. E. Kunzler, J. P. Maita, H. J. Levinstein, and E. J. Ryder, Pronounced change in the low-temperature heat capacity of V3Si with stress, Phys. Rev. 143[, 390 \(1966\).](https://doi.org/10.1103/PhysRev.143.390)
- [14] H. King, F. Cocks, and J. Pollock, Further evidence of the low temperature phase transformation in Nb3Sn and V3Si, [Physics Letters A](https://doi.org/https://doi.org/10.1016/0375-9601(67)90109-0) 26, 77 (1967).
- [15] L. F. Mattheiss and W. Weber, Electronic structure of cubic Nb3Sn and V3Si, Phys. Rev. B 25[, 2248 \(1982\).](https://doi.org/10.1103/PhysRevB.25.2248)
- [16] O. Bisi and L. W. Chiao, Electronic structure of vanadium silicides, Phys. Rev. B 25[, 4943 \(1982\).](https://doi.org/10.1103/PhysRevB.25.4943)
- [17] L. R. Testardi, Structural instability and superconductivity in A-15 compounds, [Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.47.637) 47, 637 (1975).
- [18] S. B. Dierker, M. V. Klein, G. W. Webb, and Z. Fisk, Electronic raman scattering by superconducting-gap excitations in Nb3Sn and V3Si, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.50.853) 50, 853

[\(1983\).](https://doi.org/10.1103/PhysRevLett.50.853)

- [19] B. Mitrovi´c and J. P. Carbotte, Thermodynamic and farinfrared properties of V3Si calculated from tunneling results for the eliashberg function α <sup>2</sup>F and coulomb pseudopotential parameter µ ∗ , [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.33.591) 33, 591 (1986).
- [20] K. Hirota, L. Rebelsky, and G. Shirane, X-ray-scattering study of the cubic-to-tetragonal transition and its precursive phenomenon in V3Si, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.51.11325) 51, 11325 [\(1995\).](https://doi.org/10.1103/PhysRevB.51.11325)
- [21] S. Suga, S. Itoda, A. Sekiyama, H. Fujiwara, S. Komori, S. Imada, M. Yabashi, K. Tamasaku, A. Higashiya, T. Ishikawa, M. Shang, and T. Fujikawa, Recoil effects for valence and core photoelectrons in V3Si, [Phys. Rev.](https://doi.org/10.1103/PhysRevB.86.035146) B 86[, 035146 \(2012\).](https://doi.org/10.1103/PhysRevB.86.035146)
- <span id="page-11-0"></span>[22] K. Howard, M. U. Liepe, and Z. Sun, Thermal annealing of DC sputtered Nb3Sn and V3Si thin films for superconducting radio-frequency cavities, [Journal of Applied](https://doi.org/10.1063/5.0185404) Physics 134[, 225301 \(2023\).](https://doi.org/10.1063/5.0185404)
- [23] W. Zhang, A. T. Bollinger, R. Li, K. Kisslinger, X. Tong, M. Liu, and C. T. Black, Thin-film synthesis of superconductor-on-insulator A15 vanadium silicide, [Sci](https://doi.org/10.1038/s41598-021-82046-1)[entific Reports](https://doi.org/10.1038/s41598-021-82046-1) 11, 2358 (2021).
- <span id="page-11-1"></span>[24] W. Bangert, J. Geerk, and P. Schweiss, Tunneling and neutron scattering experiments on A15 V3Si, [Phys. Rev.](https://doi.org/10.1103/PhysRevB.31.6066) B 31[, 6066 \(1985\).](https://doi.org/10.1103/PhysRevB.31.6066)
- [25] A. Borghesi, A. Piaggi, G. Guizzetti, F. Nava, and M. Bacchetta, Optical properties of vanadium silicide polycrystalline films, Phys. Rev. B 40[, 3249 \(1989\).](https://doi.org/10.1103/PhysRevB.40.3249)
- [26] G. Oya, H. Inabe, Y. Onodera, and Y. Sawada, Superconducting transition temperatures of thin V3Si layers formed by the interaction of V films with thinly oxidized Si wafers, [Journal of Applied Physics](https://doi.org/10.1063/1.330524) 53, 1115 (1982).
- [27] P. A. Psaras, M. Eizenberg, and K. N. Tu, Sequential silicide formation between vanadium and amorphous silicon thin-film bilayers, [Journal of Applied Physics](https://doi.org/10.1063/1.333910) 56, 3439 [\(1984\).](https://doi.org/10.1063/1.333910)
- [28] E. Derunova, Y. Sun, C. Felser, S. S. P. Parkin, B. Yan, and M. N. Ali, Giant intrinsic spin hall effect in W3Ta and other A15 superconductors, [Science Advances](https://doi.org/10.1126/sciadv.aav8575) 5, [eaav8575 \(2019\).](https://doi.org/10.1126/sciadv.aav8575)
- [29] T. Vethaak, F. Gustavo, T. Farjot, T. Kubart, P. Gergaud, S.-L. Zhang, F. Lefloch, and F. Nemouchi, Superconducting v3si for quantum circuit applications, [Micro](https://doi.org/https://doi.org/10.1016/j.mee.2021.111570)[electronic Engineering](https://doi.org/https://doi.org/10.1016/j.mee.2021.111570) 244-246, 111570 (2021).
- [30] N. Torikai, Neutron reflectometry, in [Neutrons in Soft](https://doi.org/https://doi.org/10.1002/9780470933886.ch5) [Matter](https://doi.org/https://doi.org/10.1002/9780470933886.ch5) (John Wiley & Sons, Ltd, 2011) Chap. II.2, pp. 115–145.
- [31] G. Shirane, J. Axe, and R. Birgeneau, Neutron scattering study of the lattice dynamical phase transition in V3Si, [Solid State Communications](https://doi.org/https://doi.org/10.1016/0038-1098(71)90530-8) 9, 397 (1971).
- [32] G. S. Knapp, S. D. Bader, and Z. Fisk, Phonon properties of A−15 superconductors obtained from heat-capacity measurements, Phys. Rev. B 13[, 3783 \(1976\).](https://doi.org/10.1103/PhysRevB.13.3783)
- [33] M. Yethiraj, D. K. Christen, A. A. Gapud, D. M. Paul, S. J. Crowe, C. D. Dewhurst, R. Cubitt, L. Porcar, and A. Gurevich, Temperature and field dependence of the flux-line-lattice symmetry in V3Si, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.72.060504) 72, [060504 \(2005\).](https://doi.org/10.1103/PhysRevB.72.060504)
- [34] A. P. Le Brun, T.-Y. Huang, S. Pullen, A. R. J. Nelson, J. Spedding, and S. A. Holt, Spatz: the time-of-flight neutron reflectometer with vertical sample geometry at the OPAL research reactor, [Journal of Applied Crystal](https://doi.org/10.1107/S160057672201086X)lography 56[, 18 \(2023\).](https://doi.org/10.1107/S160057672201086X)
- [35] A. R. J. Nelson and S. W. Prescott, refnx: neutron and Xray reflectometry analysis in Python, [Journal of Applied](https://doi.org/10.1107/S1600576718017296) [Crystallography](https://doi.org/10.1107/S1600576718017296) 52, 193 (2019).
- [36] S. A. Holt, T. E. Oliver, and A. R. J. Nelson, Using refnx to model neutron reflectometry data from phospholipid bilayers, in [Membrane Lipids: Methods and Protocols](https://doi.org/10.1007/978-1-0716-1843-1_15), edited by C. G. Cranfield (Springer US, New York, NY, 2022) pp. 179–197.
- [37] C. Goodway, P. McIntyre, A. Sears, N. Belkhier, G. Burgess, O. Kirichek, E. Leli`evre-Berna, F. Marchal, S. Turc, and S. Wakefield, A fast-cooling mode for blue series furnaces, [Journal of Neutron Research](https://doi.org/10.3233/JNR-190128) 21, 137 [\(2019\).](https://doi.org/10.3233/JNR-190128)
- <span id="page-12-0"></span>[38] J. D. Livingston, Grain size in A-15 reaction layers, [phys](https://doi.org/https://doi.org/10.1002/pssa.2210440131)[ica status solidi \(a\)](https://doi.org/https://doi.org/10.1002/pssa.2210440131) 44, 295 (1977).
- <span id="page-12-1"></span>[39] A. Godeke, M. C. Jewell, C. M. Fischer, A. A. Squitieri, P. J. Lee, and D. C. Larbalestier, The upper critical field of filamentary Nb3Sn conductors, [Journal of Ap](https://doi.org/10.1063/1.1890447)plied Physics 97[, 093909 \(2005\).](https://doi.org/10.1063/1.1890447)
- <span id="page-12-2"></span>[40] R. B. Goldfarb, M. Lelental, and C. A. Thompson, Alternating-field susceptometry and magnetic susceptibility of superconductors, in [Magnetic Susceptibility of](https://doi.org/10.1007/978-1-4899-2379-0_3) [Superconductors and Other Spin Systems](https://doi.org/10.1007/978-1-4899-2379-0_3), edited by R. A. Hein, T. L. Francavilla, and D. H. Liebenberg (Springer US, Boston, MA, 1991) pp. 49–80.
- <span id="page-12-3"></span>[41] J. J. Hauser and H. C. Theuerer, Evidence for filaments in v3si, Phys. Rev. 129[, 103 \(1963\).](https://doi.org/10.1103/PhysRev.129.103)
- <span id="page-12-4"></span>[42] E. T. Croke, R. J. Hauenstein, and T. C. McGill, Growth of superconducting V3Si on Si by molecular beam epitaxial techniques, Applied Physics Letters 53, 514 (1988).
- <span id="page-12-5"></span>[43] F. Nava, O. Bisi, and K. N. Tu, Electrical transport properties of v3si, v5si3, and vsi<sup>2</sup> thin films, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.34.6143) 34, [6143 \(1986\).](https://doi.org/10.1103/PhysRevB.34.6143)
- <span id="page-12-6"></span>[44] J. Clarke, The proximity effect between superconducting and normal thin films in zero field, [Le Journal de](https://doi.org/10.1051/jphyscol:1968201) [Physique Colloques](https://doi.org/10.1051/jphyscol:1968201) 29, C2 (1968).
- <span id="page-12-7"></span>[45] A. A. Abrikosov, Nobel lecture: Type-ii superconductors and the vortex lattice, [Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.76.975) 76, 975 (2004).
- <span id="page-12-8"></span>[46] A. Geim, S. Dubonos, J. Lok, M. Henini, and J. Maan, Paramagnetic meissner effect in small superconductors, Nature 396[, 144 \(1998\).](https://doi.org/https://doi.org/10.1038/24110)
- <span id="page-12-9"></span>[47] P. Kosti´c, B. Veal, A. P. Paulikas, U. Welp, V. R. Todt, C. Gu, U. Geiser, J. M. Williams, K. D. Carlson, and R. A. Klemm, Paramagnetic meissner effect in Nb, [Phys.](https://doi.org/10.1103/PhysRevB.53.791) Rev. B 53[, 791 \(1996\).](https://doi.org/10.1103/PhysRevB.53.791)
- <span id="page-12-10"></span>[48] V. Vinokur, P. Kes, and A. Koshelev, Flux pinning and creep in very anistropic high temperature superconductors, [Physica C: Superconductivity](https://doi.org/https://doi.org/10.1016/0921-4534(90)90100-S) 168, 29 (1990).
- <span id="page-12-11"></span>[49] P. Chaddah, Studies of irreversible magnetization in superconductors—a review, [Pramana J. Phys](https://doi.org/https://doi.org/10.1007/BF02847211) 36, 353 [\(1991\).](https://doi.org/https://doi.org/10.1007/BF02847211)
- <span id="page-12-12"></span>[50] A. B. Karki, Y. M. Xiong, N. Haldolaarachchige, S. Stadler, I. Vekhter, P. W. Adams, D. Young, W. Phelan, and J. Y. Chan, Physical properties of the noncen-

trosymmetric superconductor Nb0.18Re0.82, [Physical Re](https://doi.org/10.1103/PhysRevB.83.144525)view B 83[, 144525 \(2011\).](https://doi.org/10.1103/PhysRevB.83.144525)

- <span id="page-12-13"></span>[51] C.-Y. Xia, H.-B. Zeng, Y. Tian, C.-M. Chen, and J. Zaanen, Holographic abrikosov lattice: Vortex matter from black hole, Phys. Rev. D 105[, L021901 \(2022\).](https://doi.org/10.1103/PhysRevD.105.L021901)
- <span id="page-12-14"></span>[52] Y. Li, J. Garcia, G. Franco, J. Lu, K. Lu, B. Rong, B. Shafiq, N. Chen, Y. Liu, L. Liu, B. Song, Y. Wei, S. S. Johnson, Z. Luo, and Z. Feng, Critical magnetic fields of superconducting aluminum-substituted Ba8Si42Al<sup>4</sup> clathrate, [Journal of Applied Physics](https://doi.org/10.1063/1.4921702) 117, [213912 \(2015\).](https://doi.org/10.1063/1.4921702)
- <span id="page-12-15"></span>[53] Y. Matsuda and H. Shimahara, Fulde–ferrell–larkin– ovchinnikov state in heavy fermion superconductors, [Journal of the Physical Society of Japan](https://doi.org/https://doi.org/10.1143/JPSJ.76.051005) 76, 051005 [\(2007\).](https://doi.org/https://doi.org/10.1143/JPSJ.76.051005)
- <span id="page-12-16"></span>[54] W. Wu, A. Keren, L. Le, G. Luke, B. Sternlieb, Y. Uemura, D. Johnston, B. Cho, and P. Gehring, Magnetic penetration depth in V3Si and LiTi2O<sup>4</sup> measured by µSR, [Hyperfine Interactions](https://doi.org/https://doi.org/10.1007/BF02068956) 86, 615 (1994).
- <span id="page-12-17"></span>[55] M. Yethiraj, D. Christen, D. M. Paul, P. Miranovic, and J. Thompson, Flux lattice symmetry in V3Si: Nonlocal effects in a high-κ superconductor, [Physical review letters](https://doi.org/10.1103/PhysRevLett.82.5112) 82[, 5112 \(1999\).](https://doi.org/10.1103/PhysRevLett.82.5112)
- <span id="page-12-18"></span>[56] S. Ding, D. Zhao, T. Jiang, H. Wang, D. Feng, and T. Zhang, Surface structure and multigap superconductivity of V3Si (111) revealed by scanning tunneling microscopy, [Quantum Frontiers](https://doi.org/https://doi.org/10.1007/s44214-023-00028-y) 2, 3 (2023).
- <span id="page-12-19"></span>[57] D. Christen, H. Kerchner, S. Sekula, and Y. Chang, Fluxline lattice anisotropy in v3si: Observation and interpretation, [Physica B+C](https://doi.org/https://doi.org/10.1016/0378-4363(85)90509-1) 135, 369 (1985).
- <span id="page-12-20"></span>[58] V. Moshchalkov, M. Baert, V. Metlushko, E. Rosseel, M. Van Bael, K. Temst, R. Jonckheere, and Y. Bruynseraede, Magnetization of multiple-quanta vortex lattices, [Physical Review B](https://doi.org/10.1103/PhysRevB.54.7385) 54, 7385 (1996).
- <span id="page-12-21"></span>[59] T. Maniv, V. Zhuravlev, I. Vagner, and P. Wyder, Vortex states and quantum magnetic oscillations in conventional type-II superconductors, [Reviews of Modern Physics](https://doi.org/10.1103/RevModPhys.73.867) 73, [867 \(2001\).](https://doi.org/10.1103/RevModPhys.73.867)
- <span id="page-12-22"></span>[60] A. Campbell and J. Evetts, Flux vortices and transport currents in type ii superconductors, [Advances in Physics](https://doi.org/10.1080/00018737200101288) 21[, 199 \(1972\).](https://doi.org/10.1080/00018737200101288)
- <span id="page-12-23"></span>[61] C. E. Sosolik, J. A. Stroscio, M. D. Stiles, E. W. Hudson, S. R. Blankenship, A. P. Fein, and R. J. Celotta, Real-space imaging of structural transitions in the vortex lattice of v3Si, Phys. Rev. B 68[, 140503 \(2003\).](https://doi.org/10.1103/PhysRevB.68.140503)
- <span id="page-12-24"></span>[62] B. Rosenstein and D. Li, Ginzburg-landau theory of type II superconductors in magnetic field, [Reviews of modern](https://doi.org/10.1103/RevModPhys.82.109) physics 82[, 109 \(2010\).](https://doi.org/10.1103/RevModPhys.82.109)
- <span id="page-12-25"></span>[63] P.-G. De Gennes, [Superconductivity of metals and alloys](https://doi.org/10.1201/9780429497032) (CRC press, 2018).
- <span id="page-12-26"></span>[64] A. Golubov and U. Hartmann, Electronic structure of the abrikosov vortex core in arbitrary magnetic fields, [Physical review letters](https://doi.org/10.1103/PhysRevLett.72.3602) 72, 3602 (1994).
- <span id="page-12-27"></span>[65] E. Boaknin, M. Tanatar, J. Paglione, D. Hawthorn, F. Ronning, R. Hill, M. Sutherland, L. Taillefer, J. Sonier, S. Hayden, et al., Heat conduction in the vortex state of NbSe2: Evidence for multiband superconductivity, [Physical review letters](https://doi.org/10.1103/PhysRevLett.90.117003) 90, 117003 (2003).