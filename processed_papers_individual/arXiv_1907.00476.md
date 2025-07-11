# arXiv:1907.00476

**Paper ID:** 62e053c73fb19e5172fcdad1dac8e486

**PDF Path:** apl/Superconductors/arXiv:1907.00476.pdf

**Processing Status:** complete

**Captions Added:** 15

**Generated:** 2025-06-24T13:44:27.286958

---

# **Grain-boundary structure and segregation in Nb3Sn coatings on Nb for high-performance superconducting radiofrequency cavity applications**

Jaeyel Lee1,2, Zugang Mao<sup>1</sup> , Kai He1,3,6 , Zu Hawn Sung<sup>2</sup> , Tiziana Spina<sup>2</sup> , Sung-Il Baik1,4 , Daniel L Hall<sup>5</sup> , Matthias Liepe 5 , David N Seidman1,4 , Sam Posen<sup>2</sup>

<sup>1</sup>Department of Materials Science and Engineering, Northwestern University, Evanston, IL, 60208, USA <sup>2</sup>Fermi National Accelerator Laboratory, Batavia, IL, 60510, USA

<sup>3</sup>Northwestern University Atomic and Nanoscale Characterization Experimental Center, Evanston, IL,

# 60208, USA

<sup>4</sup>Northwestern University Center for Atom-Probe Tomography (NUCAPT), Evanston, IL, 60208, USA

<sup>5</sup>Cornell Laboratory for Accelerator-Based Sciences and Education, Cornell University, Ithaca, NY,

# 14853, USA

<sup>6</sup>Department of Materials Science and Engineering, Clemson University, Clemson, SC, 29634, USA

## **Abstract**

We report on atomic-scale analyses of grain boundary (GB) structures and segregation in Nb3Sn coatings on Nb, prepared by the vapor-diffusion process, for superconducting radiofrequency (SRF) cavity applications, utilizing atom-probe tomography, high-resolution scanning transmission electron-microscopy and first-principles calculations. We demonstrate that the chemical composition of Nb3Sn GBs is correlated strongly with the diffusion of Sn and Nb at GBs during the coating process. In a sample coated with a relatively large Sn flux, we observe an interfacial width of Sn segregation at a GB of ~3 nm, with a maximum concentration of ~35 at.%. After post-annealing at 1100 <sup>o</sup>C for 3 h, the Sn segregated at GBs disappears and Nb segregation is observed subsequently at GBs, indicating that Nb diffused into the Nb3Sn GBs from the Nb substrate. It is also demonstrated that the amount of Sn segregation in a Nb3Sn coating can be controlled by: (i) Sn flux; and (ii) the temperatures of the Nb substrates and Sn source, which may affect the overall kinetics including GB diffusion of Sn and Nb. An investigation of the correlation between the chemical compositions of GBs and Nb3Sn SRF cavity performance reveals that the Nb3Sn SRF cavities with the best performance (high-quality factors at high accelerating electric-field gradients) do not exhibit Sn segregation at GBs. Our results suggest that the chemical compositions of GBs in Nb3Sn coatings for SRF cavities can be controlled by GB engineering and can be utilized to optimize fabrication of high-quality Nb3Sn coatings for SRF cavities.

**Keywords**: superconducting radio-frequency cavities; grain-boundary segregation; Nb3Sn; transmission electron microscopy; atom-probe tomography

| Nomenclature        |                                       | 𝜇𝑖         | chemical potential of an element, i |
|---------------------|---------------------------------------|------------|-------------------------------------|
|                     |                                       | 𝜈𝐺𝐵        | GB internal energy                  |
| A                   | area of GB plane                      | 𝜌          | atomic density of Sn in Nb3Sn       |
|                     |                                       |            | crystal                             |
| c                   | rotation axis of disorientation       | APT        | Atom-probe tomography               |
| 𝑑                   | average thickness of a Nb3Sn film     | ()<br>CSL | Coincident-site lattice             |
|                     | of each coating                       |            |                                     |
| 𝑑𝑔                  | diameter of a grain                   | EBSD       | electron backscatter diffraction    |
| Eacc                | accelerating electric field           | EDS        | energy disperse spectroscopy        |
| 𝐸𝑎𝑛𝑡𝑖𝑠𝑖𝑡𝑒           | anti-site substitutional energy       | FIB        | focused ion-beam                    |
| 𝐸𝑠𝑒𝑔𝑟𝑒𝑔𝑎𝑡𝑖𝑜𝑛        | segregation energy                    | Five DOFs  | five macroscopic degrees of         |
|                     |                                       |            | freedom of a GB                     |
| 𝐸𝑁𝑏3𝑆𝑛              | total internal energy of bulk Nb3Sn   | GB         | grain boundary                      |
| 𝐸𝑆𝑛<br>𝑖𝑛 𝐺𝐵        | Sn occupation energy in the GB        | GGA        | generalized gradient                |
|                     |                                       |            | approximation                       |
| 𝑇𝑂𝑇<br>𝐸𝐺𝐵          | total internal energy of GB structure | HAADF      | high-angle annular dark-field       |
| 𝑇𝑂𝑇<br>𝐸𝑆𝑛→𝑁𝑏       | total internal energy of GB structure | HA-GB      | high-angle GB                       |
|                     | with a Sn anti-site                   |            |                                     |
| 𝑇𝑂𝑇<br>𝐸𝑆𝑛<br>in 𝐺𝐵 | total internal energy of GB structure | HR-STEM    | high-resolution scanning            |
|                     | with an additional Sn in the core of  |            | transmission electron microscopy    |
|                     | GB                                    |            |                                     |
| G                   | geometric factor                      | LA-GB      | low-angle GB                        |
| n                   | normal vector perpendicular to GB     | PAW        | projector augmented wave            |
|                     | {hkl}<br>plane                        |            | potentials                          |
| n                   | the number of Sn atoms located at     | PBE        | Perdew-Burke-Ernzerhof              |
|                     | the GB                                |            |                                     |
| Rs                  | surface resistance                    | Q0-factor  | Quality-factor                      |
| 𝑡                   | time of Nb3Sn coating process         | SRF        | superconducting radio frequency     |
| Tc                  | critical temperature                  | SU         | Structural Unit                     |
| Гi                  | Gibbsian interfacial excess of an     | VASP       | Vienna ab initio simulation         |
|                     | element, i                            |            | package                             |
| θ                   | rotation angle around<br>c-axis       |            |                                     |

# **1. Introduction**

Significant progress has recently been made towards the application of Nb3Sn films for superconducting radio-frequency (SRF) cavities for particle-accelerator applications [1, 2]. Nb3Sn has an A15 structure and high-quality thin-films (~2-3 m) of Nb3Sn are successfully created on the surfaces of niobium (Nb) cavities by employing the Sn vapor-diffusion process. With the substantially higher critical temperature (T<sup>c</sup> ~18 K) of Nb3Sn compared to Nb [3-5], 1.3 GHz SRF cavities demonstrate a Q<sup>0</sup> (quality)-factor of ~10<sup>10</sup> at 4.2 K, where the Q0-factor is defined by a geometric factor (G) divided by the surface resistance (Rs) of a cavity (G/Rs), at useful accelerating electric fields; a value much higher, ~10 times, than has been achieved previously with Nb SRF cavities for the same experimental conditions (frequency, temperature, etc.) [6-8]. The performance of these Nb3Sn cavities has, however, been limited to an accelerating field of 14-17 MV/m, which corresponds to a peak surface magnetic field of ~70 mT. In some cases, SRF cavities display a significant Q0-slope (a degradation of the quality factor with increasing electric-field) starting at ~8 MV/m. The corresponding surface magnetic field is significantly smaller than the theoretical superheating field of Nb3Sn, Hsh ~400 mT, predicted on the basis of the RF superconductivity of an ideal surface [1, 9, 10]. This RF degradation is thought to be caused by crystalline imperfections in the surface layer of the Nb3Sn coating on the Nb cavity's internal surface [2].

The coherence length () of Nb3Sn is significantly smaller than that of metallic Nb ((Nb3Sn) ~3 nm versus (Nb) ~50 nm) [1, 8, 11], potentially making Nb3Sn more vulnerable to small-scale crystalline imperfections in the Nb3Sn layer. Several studies have been performed to investigate the structural imperfections in Nb3Sn coatings, such, as abnormally thin-grains [12-14], compositional variations[15-17], and Sn-deficient Nb3Sn phases [1, 8, 18], which may be responsible for the Q0-slope and quenching, indicating a loss of superconductivity. Crystallographic orientation relationships between the Nb substrates and the Nb3Sn films were measured and the Nb3Sn/Nb heterophase interfaces are demonstrated to play a crucial role in the formation of thin-grains and Sn-deficient regions [12]. Another candidate for causing degradation of the superconducting properties of Nb3Sn SRF cavities is grain-boundaries (GBs).

In superconducting magnets, imperfections are purposely engineered into Nb3Sn wires at GBs. Impurities, such as normal conducting copper [19] or insulating oxides [20, 21] may be added to degrade locally the superconducting properties (e.g., superconducting order-parameter and superconducting energy gap [11, 22]) associated with GBs, which may create an energetically favorable volume for the nucleation of vortices, where they become pinned. In contrast, the magnetic-field direction in SRF cavities changes on a nanosecond time-scale, leading to a significant dissipation of heat and a corresponding decrease in the quality factor if the flux penetrates into the superconductor (including penetration into the GBs) [23]. The potential importance of GBs in Nb3Sn SRF cavities has been extensively discussed [3, 8, 10, 24, 25], but the structure and chemical compositions of GBs in Nb3Sn, prepared by the vapor-diffusion process for SRF cavity applications are unknown, and, therefore, the exact roles played by GBs remains unclear.

The control of GB segregation has drawn scientific interest [26-29]. For practical applications, it affects the physical properties of alloys and ultimately provides a way to design materials with desired physical and mechanical properties. Especially for ordered alloys, such as Ni3Al [27, 30], Ni3Si [31], and CoAl [32], the overall composition is critical for segregation behavior at GBs, as stoichiometry away from the ideal concentrations of the alloys generates anti-site point defects, which tend to segregate at GBs to reduce the GB's total free-energy [27, 31]. This may provide opportunities for controlling the chemical compositions of GBs in ordered-alloy systems, Nb3Sn, under the non-equilibrium vapor-diffusion condition. GB engineering of structures and chemical compositions of Nb3Sn GBs can open the possibility of improving the superconducting properties of Nb3Sn SRF cavities [33-36]. Clarification of this problem provides the motivation for our current studies.

Herein, we report the results of atomic-scale analyses of the chemical compositions and crystallographic structures of GBs, utilizing high-resolution scanning transmission electron microscopy (HR-STEM), atomprobe tomography (APT), and first-principles calculations at 0 K. We find that the chemical composition of Nb3Sn GBs is highly correlated with diffusion of Sn and Nb at GBs and it can be controlled by optimizing growth procedures. The results reveal Sn segregation at GBs of some Nb3Sn coatings, with a measured Gibbsian interfacial excess of 10-20 Sn atom/nm<sup>2</sup> caused by a continuous supply of Sn atoms from an external source, which diffuse along GBs. After post-annealing at 1100 <sup>o</sup>C for 3 h, the Sn segregation disappears and Nb segregation is then observed at GBs because the segregated Sn atoms diffuse away from the GBs and concomitantly Nb diffuses into the GBs from the Nb substrate. The amount of Sn segregation is controlled by two factors: (i) flux of Sn atoms; and (ii) the temperatures of the Nb substrate and Sn sources. The level of Sn segregation is the result of a series of kinetic processes. A correlation between the chemical compositions of Nb3Sn GBs and the performance of Nb3Sn SRF cavities are investigated. Our results demonstrate that GB engineering of GBs in Nb3Sn can be utilized to fabricate high-quality Nb3Sn coatings on Nb for SRF cavities.

## **2. Methodologies**

# 2.1. Experimental procedure

The Sn vapor-diffusion process was utilized to coat the Nb substrate of either disk-shaped coupons (10 mm diameter with a 3 mm thickness) or the inner surface of 1.3 GHz cavities with Nb3Sn films [1, 3, 8]. Niobium substrates, a Sn source, and a SnCl<sup>2</sup> nucleating agent were placed in a vacuum furnace (< 10-6 Torr). Two different growth conditions were employed for the Nb3Sn coatings to control GB segregation in Nb3Sn films; processes A and B, **Fig. 1**. For process A, the furnace temperature was initially increased to 500 °C to create nucleation sites on a Nb surface and the Sn source was concurrently heated to 500 °C. Then the temperature of the Nb substrate was increased to 1100 °C to initiate formation of the Nb3Sn phase on a Nb surface. During the coating process, the Sn source was maintained at 1200 °C, so that sufficient Sn was provided continuously to the top surface of Nb3Sn. The Sn then diffused into the Nb sample. The temperature profile of the Sn source in **Fig. 1** was determined using a thermocouple close to the Sn source. The calibration procedure to estimate the actual temperature of the Sn source indicated that the real temperatures of the Sn source for processes A and B are ~1200 <sup>o</sup>C and 1220 <sup>o</sup>C, respectively [37]. The source contained 3.0 g Sn, which is more than the amount required for the desired film thickness of ~2 μm,

and hence a Sn vapor is supplied continuously during the coating process. In process B, the Nb3Sn coating was performed with both the Nb substrate and Sn source at an ~20 <sup>o</sup>C higher temperature, **Fig. 1**, which was achieved by placing the Nb substrates and Sn source close to the heating source of the furnace. Other differences are: (1) the source consisted of only 2.5 g Sn, so that together with the ~37% increase in the Sn evaporation rate, caused by elevating the temperature of the Sn source from 1200 to 1220 <sup>o</sup>C [38]; (2) all of the Sn source evaporated by the middle of the coating process and therefore, Nb3Sn coatings experienced some annealing at the end of the coating procedure [8]. Previous data on the evaporation rate of the Sn source suggest that under these conditions, Nb3Sn coatings may experience an annealing effect for 30 min to 60 min without an external supply of Sn.

The microstructural features of the Nb3Sn samples were systematically evaluated employing the correlative use of a scanning electron-microscope (SEM), TEM, and APT. A thin TEM foil and an APT nanotip were prepared using a 600i Nanolab Helios focused ion-beam (FIB) microscope; the samples were sputter thinned with a 30 kV Ga<sup>+</sup> ion-beam at 87 pA, and then slowly sputter thinned utilizing 5 kV Ga<sup>+</sup> ions at 28 pA, to remove the radiation-damaged layers caused by the 30 kV Ga<sup>+</sup> ion-beam. HR-STEM imaging in conjunction with energy dispersive X-ray spectroscopy (EDS) were performed using an aberrationcorrected JEOL ARM 200. For APT analyses, a Cameca LEAP4000X Si and a LEAP5000XS were utilized. Nanotips were irradiated with 30 pJ ultraviolet laser (wavelength = 355 nm) pulses at a pulse repetition rate of 250 to 500 kHz, with a detection rate of 0.5 to 1.0%. 3-D reconstructions and analyses were performed using Cameca's data analysis program, IVAS 3.8.1 (Cameca, Madison, WI).

## 2.2. Computational details

The first-principles calculations utilized in this research employ the plane-wave pseudopotential totalenergy method as implemented in the *Vienna ab initio simulation package* (VASP). We used projector augmented wave (PAW) potentials [39, 40] and the generalized gradient approximation (GGA) of Perdew-Burke-Ernzerhof (PBE) for exchange-correlation [41]. Unless otherwise specified, all structures were fully relaxed with respect to volume, as well as to all cell-internal atomic coordinates. We considered carefully

the convergence of results with respect to energy cutoff and k-points. A plane-wave basis set was employed with an energy cutoff of 600 eV to represent the Kohn-Sham wave functions. The summation over the Brillouin zone for the bulk structures was performed on a 12×12×12 Monkhorst-pack k-point mesh for all calculations. The magnetic spin-polarized method was applied in all the calculations. The lattice parameter of Nb3Sn was calculated to be 5.332 Å, which is in excellent agreement with an experimental value of 5.289 Å [42]. The [12̅0]-tilt Nb3Sn GB structures were constructed using the interface builder subroutine in the MedeA software package. The GB supercell had a total of 192 atoms, which contained two GBs.

## **3. Experimental results**

## 3.1. TEM/APT analyses of Nb3Sn grain-boundaries (GBs)

**Figs. 1(a,b)** display SEM and HAADF-STEM micrographs of typical ~2 μm thick Nb3Sn coatings on Nb, prepared by the vapor-diffusion process A. The Gibbsian interfacial excesses of solute atoms and disorientations of thirty GBs detected in six Nb3Sn samples were characterized systematically: using HR-STEM, EDS, APT, and transmission electron-backscatter diffraction (EBSD). The five macroscopic degrees of freedom (DOFs) of a GB are: (i) rotation axis, **c** (two DOFs), which is a unit vector; (ii) rotation angle, **θ**, about **c** (one DOF); and (iii) the GB plane as given by **n,** which is a unit normal vector to this (hkl) plane (two DOFs). These five DOFs were determined for four selected [12̅0] tilt-GBs among the thirty GBs observed to understand the effects of the five-macroscopic DOFs on the level of Sn segregation as determined by the Gibbsian excess, i. The different levels of Sn segregation, 10 to 20 atom/nm<sup>2</sup> , are summarized in **Table 1**.

The chemical compositions of the GBs were determined by APT and the results are displayed in **Fig. 2**. The mass-spectrum of an APT analysis for a 3-D reconstruction and the identification of each peak are displayed in **Fig. S1**. Three-dimensional (3-D) reconstructions of the elements of one of the Nb3Sn samples (A5), prepared by process A, is displayed in **Fig. 2(a)**, which presents a GB along the vertical direction. Iso-concentration surfaces for Nb = 70 at.%, Sn = 30 at.%, and N = 0.3 at.% are included in the 3-D reconstructions of each element [43], which indicate GB segregation of Sn and nitrogen (N). A 1-Dconcentration profile across the GB plane is presented in **Fig. 2(b),** which indicates a maximum Sn concentration at a GB of 35 at.% and a Nb concentration of 65 at.%, which yield a value of ГSn = 18.1 ± 1.7 atom/nm<sup>2</sup> for the Gibbsian interfacial excess of Sn [44, 45]. The N concentration profile in **Fig. 2(c)** indicates that N also segregates at the same GB with an interfacial excess of Г<sup>N</sup> = 1.6 ± 0.1 atom/nm<sup>2</sup> . The full-width of the segregated region is estimated to be 3.0 ± 0.2 nm, where the width of Sn segregated region is defined by the full-width of the Sn concentration profile assuming that the Sn concentration profile follows a Gaussian function. Additional correlative TEM-APT analyses were performed for a GB from another sample (A10) prepared by process A, **Fig. 3**, with the GB plane located along the horizontal direction. In this case, the Gibbsian interfacial excess of Sn (ГSn) has the value 11.4 ± 0.8 and Г<sup>N</sup> is 0.5 ± 0.1 atom/nm<sup>2</sup> at the GB; these values are slightly smaller than those of the GB in sample A5. The full-width of the segregated region of sample A10 is 2.7 ± 0.2 nm.

HR-STEM imaging of selected [12̅0]-tilt GBs reveal the detailed atomic-structure of a GB in Nb3Sn. **Fig. 4** presents a representative high-angle [12̅0]-tilt GB in Nb3Sn. Owing to the orientation relationships (ORs) of the Nb3Sn/Nb heterophase interfaces, some Nb3Sn grains form [12̅0] tilt-GBs with different tilt-angles [12]. The structures and chemical compositions of GBs in Nb3Sn coatings were also analyzed by atomicallyresolved HR-STEM imaging and EDS mapping. HR-STEM micrographs of the high-angle (HA) tilt-GBs were recorded along the [12̅0]-zone-axis, revealing their atomic-structures. For the high-angle GB (HA-GB) of **Fig. 4(a)**, the Nb3Sn grains are disoriented by θ = 46.6<sup>o</sup> about the [12̅0]-zone-axis (**c**), while the normal vector (**n**) to the GB plane was determined to be [215]. The coincident-site lattice (CSL) sites between two grains are indicated by yellow dotted circles, which appear every three (210) planes of the GB, implying a = 3 coincidence site-lattice (CSL) [46]. The filtered HR-STEM micrograph of the HA-GB in **Fig. 4(b)** reveals structural-units (SUs), which are the cores of GB dislocations, as indicated by yellow pentagons [46, 47]. The HR-STEM image of the same GB with a GB plane normal**, n** = [211̅], which is rotated by 90<sup>o</sup> from the previous GB plane, (215), is presented in **Fig. S2**: it is also a = 3 GB.

HR-STEM EDS measurements of Nb and Sn were also performed through the plane of the GB, revealing that the Sn concentration is 32 at.% Sn at the GB, **Fig. 4(b)**, compared to ~25 at.% Sn within the grains. The Gibbsian interfacial excess (ГSn) was estimated to be 14.5 ± 2.3 atom/nm<sup>2</sup> .

## 3.2 Effect of grain-boundary (GB) structure on segregation of Sn at a GB

Additional [12̅0]-tilt-GBs with different tilt angles (θ), were analyzed to investigate correlations with GBstructure and Sn-excesses at GBs. For a low-angle grain boundary (LA-GB), **Fig. 5(a),** the HAADF-HR-STEM micrograph displays two Nb3Sn grains tilted by θ = 4.7<sup>o</sup> about the [12̅0]-axis (**c**), with the normal vector (**n**) to the GB plane being [001]. EDS measurements across the GB plane reveal a Gibbsian interfacial excess of Sn equal to ГSn =16.0 ± 6.4 atom/nm<sup>2</sup> . Another [12̅0]-tilt-GB with θ = 17.6<sup>o</sup> is displayed in **Fig. 5(b**), for which ГSn =18.5 ± 7.8 atom/nm<sup>2</sup> as determined by an EDS measurement. We note that the Gibbsian interfacial excess of Sn at the GBs is similar for both the HA-GB (14.5 ± 2.3 atom/nm<sup>2</sup> ) and LA-GB (16.0 ± 6.4 atom/nm<sup>2</sup> ), F**igs 4 and 5**, within experimental error. This implies that there is another factor that determines the level of GB-segregation of Sn in addition to the five macroscopic DOFs and their atomic structures. Tin segregation was analyzed systematically at additional GBs and the results are summarized in **Fig. 5(c)** and **Table 1**. The disorientation angles and axes of the GBs were determined by HR-STEM or transmission EBSD [48-50], **Fig. 5(c)**, and the chemistry was analyzed by HR-STEM EDS. When the values for the Gibbsian interfacial excesses of Sn at the GBs (ГSn) are plotted as a function of the disorientation angle there isn't a clear correlation among the disorientation angles of the two grains at a GB and the level of Sn segregation.

3.3 Effects of the Sn-flux and temperatures of the Nb-substrates and Sn-source on Sn-segregation at GBs **Fig. 6** displays the changes of the interfacial excesses of Sn as a function of the Sn flux and temperatures of the Nb substrates and Sn source. We note that the value of the average Sn flux is employed because there isn't an accurate method to measure the Sn flux during the coating process. The average Sn flux was estimated from the average Nb3Sn film thicknesses and the coating times, and defined as ⁄, where is
the atomic density of Sn in a Nb3Sn crystal, is the average thickness of a Nb3Sn film for each coating, and is the time for the Nb3Sn coating process. This plot yields a qualitative trend of Sn segregation at GBs, which depends on the Sn-flux. We investigated excess Sn at GBs prepared by two different processes: **Process A**: Sn and SnCl<sup>2</sup> are co-deposited on Nb substrates at T = 1100 <sup>o</sup>C for 4 h with the Sn source at 1200 <sup>o</sup>C for the first 3 h. The 3.0 g Sn source is not completely evaporated during process A, as there is some residual Sn deposition during the last hour with the Sn source at 1100 <sup>o</sup>C; **Process B:** Nb3Sn samples are coated at a slightly elevated temperature, by 20 <sup>o</sup>C for both Nb substrates (T = 1120 <sup>o</sup>C), for 4 to 5 h and the Sn source (T = 1220 <sup>o</sup>C) for the first 3 h. We think that the 2.5 g Sn source is evaporated completely by the middle of the coating process due to the ~37% increase of the Sn evaporation rate in process B compared to process A [8]; therefore, annealing occurs in the absence of a flux from the Sn source, for 30- 60 min at the end of the coating process, **Fig.1(c)**.

Eight Nb3Sn samples were prepared by process A with different average Sn fluxes, which were analyzed by TEM. Tin excesses at the GBs were characterized utilizing EDS in the STEM mode for 1 to 4 GBs per sample. **Fig 6** presents the average values of Sn segregation at the GBs (Sn excess at GBs) of each sample, indicating that the Sn excesses are correlated with the average Sn flux. Tin segregation wasn't observed at the GBs in the low average Sn-flux samples, but there was evidence of an increase in the medium Sn-flux (100-150 Sn atom/nm<sup>2</sup> min) samples. The level of the Sn excess is saturated at ~18 atom/nm<sup>2</sup> , with a corresponding maximum value of the GB concentration of ~35 at.% Sn.

Three samples coated by process B are displayed as red-dots in **Fig. 6** and the chemistry of 3 to 4 GBs were analyzed per sample utilizing STEM-EDS. Tin segregation wasn't observed at GBs coated according to process B. Tin depletion (or Nb segregation) was, however, observed for some GBs. The causes for the differences in the chemical compositions at GBs among samples produced by processes A and B are discussed in Section 4.2 and 4.3.

3.4 Effects of annealing on the Sn segregations at GBs

To identify the effect of annealing for 30-60 min in process B, annealing experiments were conducted for a Nb3Sn sample (A5), which exhibited Sn segregation at GBs, **Fig. 2**. The Nb3Sn sample with Sn segregation (A5) was annealed at 1100 <sup>o</sup>C for 3 h in the same vacuum furnace (< 10-6 Torr) and two GBs of the annealed Nb3Sn sample were analyzed using HR-STEM EDS. Indeed, Sn segregation disappears and Nb segregates at GBs, probably due to the diffusion of Nb from the Nb substrates, **Figs. 7** and **S3**, along GBs. The level of Nb segregation is 20 to 22 ±3 Nb atom/nm<sup>2</sup> for each GB and the full-width is ~5 ±1 nm for both GBs, which is a higher level and a wider full-width than for Sn segregation at GBs. The difference of Sn concentrations between the grain-interiors and the GB is ~9 at.%, which implies that the concentration of Sn at GBs decreases to ~16 at.%.

## 3.5 Segregation energy for Sn at a GB in Nb3Sn

The GB structural model of a symmetrical Nb3Sn 46.6<sup>o</sup> /[12̅0] tilt-GB with the GB-plane (211̅), **Fig. S.1**, was constructed for the purpose of first-principles calculations and is displayed in **Fig. 8(b)**. The GB was fully relaxed and the GB internal energy, *GB*, is determined using:

$$\nu\_{\rm GB} = (E\_{\rm GB}^{\rm TOT} - 2E\_{\rm Nb\_3Sn}) / \text{2A} \tag{l}$$

Where *TOT <sup>E</sup>GB* is the total internal energy of the GB structure displayed in **Fig. 8**, *Nb Sn* 3 *E* is the total internal energy of one side of the bulk Nb3Sn, and A is the area of the GB. The calculated GB internal energy at 0 K is 536 mJ/m<sup>2</sup> .

For the case with an excessive amount of Sn (>25 at. %), the calculations were performed for three different locations of an extra Sn atom; two locations at the core GB sites with large free-volumes (labeled 1 and 2 in **Fig. 8(b)**), where the free-volume is defined as an excess volume attributed to any crystal defect [51], and one Nb atomic site in the bulk material to form an anti-site (labeled as number 3 in **Fig. 8(b)**) pointdefect.

First, the anti-site substitutional energy of Sn for site 3 is calculated employing:

$$E\_{\rm antite} = (E\_{\rm 3n \to Nb}^{\rm TOT} + \mu\_{\rm Nb}) - (E\_{\rm GB}^{\rm TOT} + \mu\_{\rm 3n}) \tag{2}$$

Where *TOT ESn Nb* → is the total GB energy with a Sn anti-site point-defect, and *Nb* and *Sn* are the chemical potentials for pure Nb and Sn atoms. The calculated Sn anti-site energy is 0.635 eV/atom. Next, the Sn occupation energy at core GB sites (sites 1 and 2) is calculated using: 

$$E\_{\text{Sn in GB}} = \left[ E\_{\text{Sn in GB}}^{\text{TOT}} - (E\_{\text{GB}}^{\text{TOT}} + \mu\_{\text{Sn}}) \right] / n \tag{3}$$

Where in *TOT ESn GB* is the total GB energy with a Sn atom situated in the core of a GB, site 1 or 2, as displayed in **Fig. 8**, and *n* is the number of Sn atoms located at a GB. The calculated Sn occupation energy at the core of a GB is -0.164 eV/atom at site 1 and -0.152 eV/atom at site 2. Tin occupation at a GB can stabilize the GB structure by decreasing the system's total energy at both the sites studied herein. Site 1 has a slightly more negative occupation energy than site 2 due to having a 24% larger free-volume, **Fig. 8**. Finally, the driving force for excess Sn segregation from the bulk to a GB can be determined by calculating the segregation energy as follows:

$$E\_{\text{negmentation}} = E\_{\text{Sr in GB}} - E\_{\text{anticite}} \tag{4}$$

The value of the segregation internal energy calculated at 0 K for our system is -0.799 eV/atom for site 1 and -0.787 eV/atom for site 2 at the GB, which are quite close to one another.

#### **4. Discussion**

### 4.1. Structures and chemical compositions of Nb3Sn GBs

TEM and APT analyses of the atomic structure and chemical compositions of GBs in Nb3Sn demonstrate that the full-width of the Sn segregated region at a GB is ~3.0 ±0.5 nm and the maximum Sn concentration is ~35 at.% (**Figs. 2-4**). As seen in **Fig. 4(b)**, periodic atomic arrays of structural units (SUs) (the cores of GB dislocations) are observed in the HR-STEM images. A possible sign of chemical ordering of segregated Sn at a GB is observed in a limited case, as indicated by the periodic bright-contrast effects of coincident sites at a GB of **Figs. 4(a)** and **S4**. This implies that there could be a higher concentration of Sn atoms in an

atomic column in the core of a GB dislocation, agreeing with the results of our first-principles calculations. There is, however, no clear evidence for the presence of chemical ordering or non-equilibrium phases [52- 55] at other GBs employing additional rigorous analyses using HR-STEM, and we conclude that most GBs form structurally sharp interfaces, **Fig. 5**.

Our first-principles calculations demonstrated that GBs are favorable sites for excess Sn in Nb3Sn. As the formation energy of a Sn-anti-site defect is high, 0.635 eV [12, 56], Sn-anti-site defects are most likely to segregate at GBs to reduce the grain-boundary energy of Nb3Sn. When a Sn-anti-site defect moves from the bulk (site 3) to a core GB site (1 and 2 in **Fig. 8(b)**), the energy is reduced by ~0.8 eV/atom, which is the segregation energy of Sn at the cores of GBs. Excess Sn atoms may first fill the core GB sites with large free volumes but since these sites are limited, it may then begin to occupy a Nb sublattice site near the GBs, within the ~3 nm width, after filling the GB core-sites. The segregation energy of a Sn atom at the Nb sublattice near GBs may be smaller than that at the core sites of a GB, because segregated Sn forms a Sn anti-site defect near a GB. This reduction in segregation energy means that after filling the core GB sites, Sn segregation at GBs becomes more costly energy-wise, which may explain the observation that saturation of Sn segregation occurs at 15~20 atom/nm<sup>2</sup> even with a high Sn-flux, **Fig. 6**. An estimated 15 atom/nm<sup>2</sup> of Sn excess in ~3 nm surrounding a GB generates about one Sn anti-site defect in every two unit-cells of Nb3Sn. This results in a substantial increase in internal energy, which is probably insufficient to overcome the nucleation barrier to transform to other phases.

### 4.2. Origin of Sn segregation at GBs in Nb3Sn

In this section, we consider the causes of Sn segregations at GBs based on thermodynamic and kinetic reasons. There are two thermodynamic factors: (i) GB structure; and (ii) stoichiometry of Nb3Sn associated with the formation energies of Sn and Nb antisites. The GB structures are highly meta-stable, so following J. W. Cahn's *ansatz* that the five macroscopic DOFs of a GB are thermodynamic state variables, which determine the structure of a GB, we studied GB structure and its relationship to segregation [52]. Cahn assumed that the three microscopic DOFs are relaxed to their equilibrium values, which may or may not be the case for our situation. For the kinetic origins, Sn (and Nb) diffusion with or without a Sn-flux during the coating process is considered.

Firstly, it is noteworthy that GB segregation for the vapor-diffused Nb3Sn layer is not correlated with the structures of the GBs as represented by the five macroscopic DOFs. This is in contrast to the situation for an equilibrium state, where the interfacial excesses of solute are determined by the structures of the GBs, represented by the five macroscopic DOFs [33, 57]. LA-GBs contain smaller dislocation densities than HA-GBs, and therefore, in general, should exhibit smaller values of the Gibbsian excesses of solute atoms [34]. We have, however, observed no detectable differences in the amount of Sn segregation at LA-GBs and HA-GBs, **Figs. 4** and **5**.

Another thermodynamic origin that affects segregation behavior at GBs in Nb3Sn is the stoichiometry of Nb3Sn associated with the formation energies of Sn and Nb antisite point defects. Nb3Sn is an ordered alloy, where the segregation behavior of solutes is affected by the stoichiometry of the ordered intermetallic phase [27, 30-32, 58, 59]. For ordered alloys, the chemical composition of the GBs is related to the formation of anti-site defects in the interior of grains and their segregation at GBs to reduce GB energies. First-principles calculations at 0 K demonstrate that excess Sn inside Nb3Sn grains has a high driving-force for segregation at GBs, ~0.8 eV/atom. If the Sn concentration in Nb3Sn is <25 at.% Sn, Nb is the excess element and the excess Nb atoms then segregate preferentially at GBs to reduce their energy. We tried to analyze the effect of the bulk composition of Nb3Sn on Sn segregation at GBs employing APT/STEM-EDS analyses and strong correlations aren't observed, **Fig. S5**: some Nb3Sn grains that exhibit Sn-deficient regions display Sn-segregation at GBs and some possible Sn-rich Nb3Sn grains with ~25.5 at.% Sn do not display Sn segregation at GBs.

Lastly, the role of kinetics is discussed. In **Fig. 6**, the level of Sn segregation at GBs changes with different values of the average Sn flux. Higher Sn fluxes cause larger levels of Sn segregation at GBs, **Fig. 6**. This indicates that Sn diffusion along GBs during the coating process, with an external Sn-flux, plays an important role in Sn segregation at GBs. With an external Sn source, there is a continuous Sn flux to Nb3Sn coatings from the top surface and GBs in Nb3Sn are the main paths for Sn diffusion into Nb3Sn and Nb substrates. Alternatively, annealing a Nb3Sn sample, without an external Sn source, induces diffusion of the segregated Sn atoms from the GBs and Nb diffusion into the GBs from the Nb substrates, which results in Nb segregation at GBs, **Fig. 7**. The outcome that Sn-segregation is controlled by annealing with or without a Sn-flux is strong evidence that the Sn segregation is caused by Sn-diffusion along GBs, due to the external Sn-flux, which is a kinetic effect.

Segregation at GBs in Nb3Sn in the current study (especially for process A) was observed in a nonequilibrium state, with a continuous supply of Sn atoms from the vapor phase. Short-circuit diffusion of Sn and Nb along GBs and interfacial reactions at the Nb3Sn/Nb interface continue until the termination of the coating period for both processes A and B. In this non-equilibrium state of the Sn-vapor diffusion coating process, the effects of kinetics on GB segregation in Nb3Sn films may dominate over thermodynamic factors and it may rationalize the absence of clear effects of GB structure and the stoichiometry of Nb3Sn on the level of Sn segregation, within the detection limit of STEM-EDS and APT analyses.

We note, however, that even though the kinetics is the primary origin that determines the chemical compositions of GBs in Nb3Sn during the coating process, the higher level of Nb segregation and the wider width of the Nb concentration profile at GBs (**Figs 7** and **S3**), compared to Sn segregation, is attributed partially to the thermodynamic origin: a smaller formation energy of a Nb anti-site defect (0.268 eV) compared to a Sn anti-site defect (0.637 eV). Nb3Sn has a wide range of composition, ranging from 17 to 26 at.% Sn, which allows for variations in the chemical compositions of GBs in Nb3Sn, especially on the Nb-rich side of the Nb-Sn phase diagram [60].

It is noteworthy that the level of Sn segregation is essentially a constant along the depth of GBs from the top-surface of Nb3Sn to the Nb3Sn/Nb heterophase interface, **Fig. 9**. This result indicates that GB diffusion of Sn is so rapid that there isn't a concentration gradient along a GB [61]. The short-circuit diffusivity of Sn along GBs (Dgb) is faster by more than three orders of magnitude than bulk diffusion in Nb3Sn (Db) at 1100 <sup>o</sup>C; the estimated root-mean-square (RMS) diffusion distance of Sn in bulk Nb3Sn and in GBs at 1100 <sup>o</sup>C in 1 h (√4) are ~200 nm and ~120 μm, respectively [12, 56]. The RMS bulk diffusion distance of Sn at 1100 <sup>o</sup>C, ~200 nm, is larger than the GB diameter (ϭ = 3 nm), but it is much smaller than the grain diameter (d<sup>g</sup> = ~2 μm). This implies that GB diffusion is the primary path for Sn diffusion in Nb3Sn coatings and Dgb for Sn in GBs in our Nb3Sn coatings is classified as type-B diffusion, according to Harrison's classification scheme, where ϭ<<√<<d<sup>g</sup> [62, 63], which is called the *leaky-pipe* model.

#### 4.3. Controlling the chemical compositions of Nb3Sn GBs by growth processes

The key finding of the current research is that Sn segregation at a GB can be controlled during annealing with or without a Sn-flux. Specifically, it is highly correlated with, and can be controlled by, two factors: (i) the Sn flux; and (ii) the temperature of the Nb substrate and Sn source as observed in processes A and B, **Fig. 6**. Tin segregation at GBs disappears and Nb segregates at the GBs of some Nb3Sn coated films utilizing process B. The causes of the different chemical compositions of GBs in the Nb3Sn coatings prepared by process B compared to process A are: (i) there isn't an external Sn flux during the last 30 to 60 min at the end of the coating period in process B; (ii) as a result, the Nb3Sn coatings are annealed at 1120 <sup>o</sup>C for an extra 30 to 60 min and the segregated Sn atoms have sufficient time to diffuse into the Nb substrate to form Nb3Sn. Longer annealing times, without an external Sn source, cause Nb diffusion from the Nb substrate into Nb3Sn GBs, which leads to Nb segregation at GBs, **Fig. 7**. Variations of GB segregation in the presence or absence of an external vapor source have also been reported for the Cu-Bi system [29], where segregation of Bi at GBs in Cu is induced by an external Bi vapor-source, where the level of Bi segregation at a GB achieves a stationary-state. Once the external Bi vapor-source is removed the segregated Bi atom concentration at the GBs decreases until an equilibrium Bi vapor-pressure is established. Another difference in process B compared to process A is that the elevated temperatures of the Nb substrate and Sn source probably increase the interfacial reaction rate at the Nb3Sn/Nb interface. Assuming the activation energy barrier for Nb3Sn growth is ~230 kJ/mole [64], then increasing the temperature from 1100 <sup>o</sup>C to 1120 <sup>o</sup>C accelerates the growth rate of Nb3Sn by ~34%, assuming the Nb3Sn film growth follows an Arrhenius expression [64-66]. This increase of the interfacial reaction rate at the Nb3Sn/Nb interface may consume ~34% more Sn atoms at Nb3Sn/Nb interfaces, which helps to reduce Sn segregation at GBs.

4.4. Grain-boundary engineering for high-performance Nb3Sn SRF cavities

The width of Sn segregation at a GB (~3 nm) is comparable to the superconducting coherence length of Nb3Sn [2, 10, 23]. This suggests that Sn segregation at GBs can indeed possibly provide a path for magnetic flux penetration from the surface. Our research demonstrates that the chemical compositions of Nb3Sn GBs for the Sn vapor-diffusion process can be fine-tuned utilizing an optimized growth-process procedure, **Fig. 6**. Thus, GB engineering of Nb3Sn coatings provides a route for improving the superconducting properties of Nb3Sn SRF cavities [33], which is the goal of our research.

One Nb3Sn sample was cut directly from a high-performance Nb3Sn cavity, Cornell cavity 2 (**Fig. 10(a)**) [67], to investigate the correlation between Sn segregation at GBs and surface resistance. When GBs from a high-performance Nb3Sn cavity were analyzed by HR-STEM EDS and APT, there were no signs of Sn segregation, although a small amount of Sn depletion (or Nb segregation) of ~4 atom/nm<sup>2</sup> was observed at one GB, **Fig. 10(b)**. Another Nb3Sn coated cavity, Fermilab cavity 3, was prepared using process B in an attempt to avoid Sn segregation. A witness sample was coated together with Fermilab cavity 3 by process B to simulate Nb3Sn films in a cavity. STEM-EDS analysis of the GBs of the witness Nb3Sn sample revealed a lack of Sn segregation at GBs. A Sn concentration profile through a GB is displayed in **Fig. 10(c)**. Furthermore, Fermilab cavity 3 demonstrated a high level of cavity performance, with a minimal Qslope until an accelerating gradient of 15 MV/m.

Alternatively, Fermilab cavities 1 and 2, which experience degradation of the Q-factor at ~8 MV/m, do exhibit Sn segregation at GBs in the witness samples of the cavities, which are coated using the same growth condition to Fermilab cavities 1 and 2 (numbers 1 and 2, **Fig. 6**). The Sn excesses at the GBs are equal to 13.9 ± 1.8 and 12.3 ± 6.6 atom/nm<sup>2</sup> for samples 1 and 2, respectively. Our present results suggest that it is reasonable to postulate a correlation between Sn segregation at GBs in Nb3Sn and the appearance of Qslope or quench in a Nb3Sn cavity. This could indicate that GB segregation of Sn increases the surface resistance in Nb3Sn cavities and it is possible that the smaller Sn segregation at GBs may mitigate the degradation of Nb3Sn cavity performances (Q-slope and quench of Nb3Sn cavities).

We also investigated the possible effects of Sn-deficient regions and surface chemistry and to date significant roles aren't observed [68]. It is extremely challenging to isolate the effect of GBs on the SRF cavity performance from other factors, such as Sn-deficient grains, surface roughness, and surface chemistry, and this will require further investigations. Theoretical evaluations on the correlation between Sn segregation at GBs and superconducting properties is ongoing using density functional theory (DFT) calculations of the change of the critical temperature of Nb3Sn for different Sn concentrations and a timedependent Ginzburg-Landau model to simulate the penetration of magnetic fluxes into the Sn segregated GB, which will be reported in separate articles [69-71]. For Nb3Sn magnet applications, there has been intensive research on the role of GBs as flux-pinning centers, demonstrating that GBs in Nb3Sn magnets are pinning-centers of magnetic flux owing to the short coherence length of Nb3Sn (~3 nm), which enhances the current density [72-76]. Present investigations suggest that GBs in Nb3Sn are probably relevant to SRF cavity applications as well, and they provide a pathway to control the detailed chemical compositions of GBs in Nb3Sn used for SRF cavity applications.

### **5. Conclusions**

This article describes the use of atom-probe tomography (APT), high-resolution scanning transmission electron microscopy (HR-STEM), and first-principles calculations at 0 K, to perform systematic analyses of the chemical compositions and structures of grain boundaries (GBs) in Nb3Sn for superconducting radiofrequency (SRF) applications. Our results indicate a series of relationships among the growth kinetics, chemical compositions of grain-boundaries (GBs) and Nb3Sn SRF cavity performance.

- Grain-boundary short-circuit diffusion of Sn and Nb play an important role on the chemical compositions of GBs in Nb3Sn, which can be controlled by optimizing the growth processes.
- Tin segregation is observed at GBs in some Nb3Sn coatings and it is attributed to Sn short-circuit diffusion along GBs, due to the continuous Sn flux from an external Sn source. APT and HR-STEM-EDS analyses reveal that 10-20 atom/nm<sup>2</sup> of excess Sn segregates at the GBs, within a ~3 nm full-width and a maximum Sn concentration of 35 at.% Sn.
- Tin segregation at GBs disappears after post-annealing at 1100 <sup>o</sup>C for 3 h and concomitantly Nb segregates at the GBs due to diffusion of the segregated Sn from the GBs and simultaneously Nb diffusion along the GBs from the Nb substrates.
- The amount of Sn segregation at a GB during the coating process is correlated with two factors: (1) Sn flux; and (2) the temperatures of the Nb-substrate and Sn-source. These parameters affect shortcircuit Sn diffusion along GBs and probably interfacial reactions at Nb3Sn/Nb interfaces, which determines the chemical composition of a GB.
- First-principles calculations at 0 K demonstrate that excess Sn in Nb3Sn segregates preferentially at GBs due to the high formation-energy of Sn-anti-site defects in bulk Nb3Sn (0.635 eV).
- An investigation of the correlation between Sn segregation at GBs and cavity performance demonstrates that high-performance Nb3Sn SRF cavities are probably achieved when *GBs in Nb3Sn coatings do not exhibit significant Sn or Nb segregation*.
- We have shown experimentally that studies of GBs in Nb3Sn are relevant to the Q<sup>0</sup> (quality)-factor of Nb3Sn SRF cavities as a function of the accelerating electric field (MV/m), **Fig. 10**, owing to the short coherence length (~3 nm) of Nb3Sn.
- Implementation of the optimum process may facilitate the fabrication of high-performance Nb3Sn cavities. *Our results help, therefore, to improve the fabrication of high-quality Nb3Sn coatings for SRF cavity applications*.

# **Acknowledgments**

We are grateful to Drs. Amir R. Farkoosh, Xiaobing Hu, Dieter Isheim, Shipeng Shu, Xingchen Xu, Xuefeng Zhou and Mr. Qingqiang Ren for valuable discussions, and Mr. Brad Tennis for assistance in processing Nb3Sn coatings. We also thank Profs. Thomas Arias, David A. Muller, and James P. Sethna for their valuable suggestions. This research is supported by the United States Department of Energy, Offices of High Energy. Fermilab is operated by the Fermi Research Alliance LLC under Contract No. DE-AC02-
07CH11359 with the United States Department of Energy. This work made use of the EPIC, Keck-II, and/or SPID facilities of Northwestern University's NU*ANCE* Center, which received support from the Soft and Hybrid Nanotechnology Experimental (SHyNE) Resource (NSF ECCS-1542205); the MRSEC program (NSF DMR-1121262) at the Materials Research Center; the International Institute for Nanotechnology (IIN); the Keck Foundation; and the State of Illinois, through the IIN. Cornell's Nb3Sn coating program is supported by United States Department of Energy grant DE-SC0008431. Atom-probe tomography was performed at the Northwestern University Center for Atom-Probe Tomography (NUCAPT). The LEAP tomograph at NUCAPT was purchased and upgraded with grants from the NSF-MRI (DMR-0420532) and ONR-DURIP (N00014-0400798, N00014-0610539, N00014-0910781, N00014-1712870) programs. NUCAPT received support from the MRSEC program (NSF DMR-1720139) at the Materials Research Center, the SHyNE Resource (NSF ECCS-1542205), and the Initiative for Sustainability and Energy (ISEN) at Northwestern University.

## **References**

[1] S. Posen, Understanding and overcoming limitation mechanisms in Nb3Sn superconducting RF cavities, Ph.D. thesis, Cornell University, Ithaca, NY, USA, 2015.

[2] S. Posen, N. Valles, M. Liepe, Radio Frequency Magnetic Field Limits of Nb and Nb3Sn, Phys. Rev. Lett. 115(4) (2015) 047001.

[3] G. Arnolds, D. Proch, Measurement on a Nb3Sn structure for linear accelerator application, IEEE Trans. Magn. 13(1) (1977) 500-503.

[4] B. Hillenbrand, H. Martens, H. Pfister, K. Schnitzke, Y. Uzel, Superconducting Nb3Sn cavities with high microwave qualities, IEEE Trans. Magn. 13(1) (1977) 491-495.

[5] P. Kneisel, O. Stoltz, J. Halbritter, Measurements of superconducting Nb3Sn cavities in the GHz range, IEEE Trans. Magn. 15(1) (1979) 21-24.

[6] S. Posen, M. Checchin, A. Crawford, A. Grassellino, M. Martinello, O. Melnychuk, A. Romanenko, D. Sergatskov, Y. Trenikhina, Efficient expulsion of magnetic flux in superconducting radiofrequency cavities for high Q 0 applications, J. Appl. Phys. 119(21) (2016) 213903.

[7] S. Posen, M. Liepe, Advances in development of Nb3Sn superconducting radio-frequency cavities, Physical Review Special Topics-Accelerators and Beams 17(11) (2014) 112001.

[8] S. Posen, D. Hall, Nb3Sn superconducting radiofrequency cavities: fabrication, results, properties, and prospects, Supercond. Sci. Technol. 30(3) (2017) 033004.

[9] M.K. Transtrum, G. Catelani, J.P. Sethna, Superheating field of superconductors within Ginzburg-Landau theory, Physical Review B 83(9) (2011) 094505.

[10] D.B. Liarte, S. Posen, M.K. Transtrum, G. Catelani, M. Liepe, J.P. Sethna, Theoretical estimates of maximum fields in superconducting resonant radio frequency cavities: stability theory, disorder, and laminates, Supercond. Sci. Technol. 30(3) (2017) 033002.

[11] A. Gurevich, To use or not to use cool superconductors?, Nat. Mater. 10(4) (2011) 255-259.

[12] J. Lee, S. Posen, Z. Mao, Y. Trenikhina, K. He, D.L. Hall, M. Liepe, D.N. Seidman, Atomic-scale analyses of Nb3Sn on Nb prepared by vapor-diffusion for superconducting radiofrequency cavity applications: A correlative study, Supercond. Sci. Technol. 32(2) (2018) 024001.

[13] D. Hall, J. Kaufman, M. Liepe, J. Maniscalco, Surface Analysis Studies of Nb3Sn Thin Films, 7th International Particle Accelerator Conference (IPAC'16), Busan, Korea, May 8-13, 2016, JACOW, Geneva, Switzerland, 2016, pp. 2316-2319.

[14] Y. Trenikhina, S. Posen, A. Romanenko, M. Sardela, J. Zuo, D. Hall, M. Liepe, Performance-defining properties of Nb3Sn coating in SRF cavities, Supercond. Sci. Technol. 31(1) (2017) 015004.

[15] F. Hellman, Surface‐to‐surface segregation during growth of polycrystalline thin films, Appl. Phys. Lett. 51(12) (1987) 948-950.

[16] F. Hellman, T.H. Geballe, Specific heat of thin-film A15 superconductors: An anomalous inhomogeneity discovered, Physical Review B 36(1) (1987) 107-120.

[17] F. Hellman, J. Talvacchio, T. Geballe, A. Marshall, A new look at the growth of thin films of Nb-Sn, Advances in cryogenic engineering materials, Springer1986, pp. 593-602.

[18] C. Becker, S. Posen, N. Groll, R. Cook, C.M. Schlepütz, D.L. Hall, M. Liepe, M. Pellin, J. Zasadzinski, T. Proslier, Analysis of Nb3Sn surface layers for superconducting radio frequency cavity applications, Appl. Phys. Lett. 106(8) (2015) 082602.

[19] M. Sandim, D. Tytko, A. Kostka, P. Choi, S. Awaji, K. Watanabe, D. Raabe, Grain boundary segregation in a bronze-route Nb3Sn superconducting wire studied by atom probe tomography, Supercond. Sci. Technol. 26(5) (2013) 055008.

[20] X. Xu, A review and prospects for Nb3Sn superconductor development, Supercond. Sci. Technol. 30(9) (2017) 093001.

[21] X. Xu, M.D. Sumption, X. Peng, Internally oxidized Nb3Sn strands with fine grain size and high critical current density, Adv. Mater. 27(8) (2015) 1346-1350.

[22] S. Graser, P.J. Hirschfeld, T. Kopp, R. Gutser, B.M. Andersen, J. Mannhart, How grain boundaries limit supercurrents in high-temperature superconductors, Nat. Phys. 6(8) (2010) 609-614.

[23] A. Gurevich, G. Ciovati, Dynamics of vortex penetration, jumpwise instabilities, and nonlinear surface resistance of type-II superconductors in strong rf fields, Physical Review B 77(10) (2008) 104501.

[24] A. Gurevich, Theory of RF superconductivity for resonant cavities, Supercond. Sci. Technol. 30(3) (2017) 034004.

[25] A. Sheikhzada, A. Gurevich, Dynamic transition of vortices into phase slips and generation of vortexantivortex pairs in thin film Josephson junctions under dc and ac currents, Physical Review B 95(21) (2017) 214507.

[26] Z. Jiao, C. Schuh, Nanocrystalline Ag-W alloys lose stability upon solute desegregation from grain boundaries, Acta Mater. 161 (2018) 194-206.

[27] D.A. Muller, S. Subramanian, P.E. Batson, J. Silcox, S.L. Sass, Structure, chemistry and bonding at grain boundaries in Ni3Al—I. The role of boron in ductilizing grain boundaries, Acta Mater. 44(4) (1996) 1637-1645.

[28] W. Xing, A.R. Kalidindi, D. Amram, C.A. Schuh, Solute interaction effects on grain boundary segregation in ternary alloys, Acta Mater. 161 (2018) 285-294.

[29] B. Kempshall, B. Prenitzer, L. Giannuzzi, Grain boundary segregation: equilibrium and nonequilibrium conditions, Scr. Mater. 47(7) (2002) 447-451.

[30] S. Subramanian, D.A. Muller, J. Silcox, S.L. Sass, Structure, chemistry and bonding at grain boundaries in Ni3Al—II. The structure of small angle boundaries, Ni-enrichment and its influence on bonding, structure, energy and properties, Acta Mater. 44(4) (1996) 1647-1655.

[31] S. Subramanian, D.A. Muller, J. Silcox, S.L. Sass, The role of chemistry in controlling the bonding and fracture properties of grain boundaries in L1<sup>2</sup> intermetallic compounds, Materials Science and Engineering: A 239 (1997) 297-308.

[32] V. Blum, L. Hammer, C. Schmidt, W. Meier, O. Wieckhorst, S. Müller, K. Heinz, Segregation in strongly ordering compounds: a key role of constitutional defects, Phys. Rev. Lett. 89(26) (2002) 266102.

[33] D. Raabe, M. Herbig, S. Sandlöbes, Y. Li, D. Tytko, M. Kuzmina, D. Ponge, P.-P. Choi, Grain boundary segregation engineering in metallic alloys: A pathway to the design of interfaces, Curr. Opin. Solid State Mater. Sci. 18(4) (2014) 253-261.

[34] J.D. Rittner, D.N. Seidman, Solute-atom segregation to< 110> symmetric tilt grain boundaries, Acta Mater. 45(8) (1997) 3191-3202.

[35] D. Dimos, P. Chaudhari, J. Mannhart, Superconducting transport properties of grain boundaries inYBa2Cu3O<sup>7</sup> bicrystals, Phys. Rev. B 41(7) (1990) 4038-4049.

[36] S. Lee, J. Jiang, J. Weiss, C. Folkman, C. Bark, C. Tarantini, A. Xu, D. Abraimov, A. Polyanskii, C. Nelson, Weak-link behavior of grain boundaries in superconducting Ba (Fe1− x Cox)2As<sup>2</sup> bicrystals, Appl. Phys. Lett. 95(21) (2009) 212505.

[37] Note that the temperature of the tin heater is approximately 100 <sup>o</sup>C higher than that of the crucible containing the tin. This is not measured directly due to observed degradation of the Pt-Rh thermocouples when exposed to high flux of tin vapor.

[38] F. Geiger, C. Busse, R. Loehrke, The vapor pressure of indium, silver, gallium, copper, tin, and gold between 0.1 and 3.0 bar, Int. J. Thermophys. 8(4) (1987) 425-436.

[39] P.E. Blöchl, Projector augmented-wave method, Phys. Rev. B 50(24) (1994) 17953.

[40] G. Kresse, D. Joubert, From ultrasoft pseudopotentials to the projector augmented-wave method, Phys. Rev. B 59(3) (1999) 1758.

[41] J.P. Perdew, K. Burke, M. Ernzerhof, Generalized gradient approximation made simple, Phys. Rev. Lett. 77(18) (1996) 3865.

[42] H. Devantay, J.L. Jorda, M. Decroux, J. Muller, R. Flükiger, The physical and structural properties of superconducting A15-type Nb-Sn alloys, J. Mater. Sci. 16(8) (1981) 2145-2153.

[43] O.C. Hellman, J.A. Vandenbroucke, J. Rüsing, D. Isheim, D.N. Seidman, Analysis of threedimensional atom-probe data by the proximity histogram, Microsc. Microanal. 6(5) (2000) 437-444.

[44] O.C. Hellman, D.N. Seidman, Measurement of the Gibbsian interfacial excess of solute at an interface of arbitrary geometry using three-dimensional atom probe microscopy, Materials Science and Engineering: A 327(1) (2002) 24-28.

[45] D. Isheim, M.S. Gagliano, M.E. Fine, D.N. Seidman, Interfacial segregation at Cu-rich precipitates in a high-strength low-carbon steel studied on a sub-nanometer scale, Acta Mater. 54(3) (2006) 841-849.

[46] A.P. Sutton, R.W. Balluffi, Interfaces in crystalline materials, Monographs on the Physice and Chemistry of Materials (1995) 414-423.

[47] J.D. Rittner, D.N. Seidman, 〈110〉 symmetric tilt grain-boundary structures in fcc metals with low stacking-fault energies, Physical Review B 54(10) (1996) 6999-7015.

[48] S.-I. Baik, M. Olszta, S. Bruemmer, D.N. Seidman, Grain-boundary structure and segregation behavior in a nickel-base stainless alloy, Scr. Mater. 66(10) (2012) 809-812.

[49] W. Bollmann, Crystal lattices, interfaces, matrices: an extension of crystallography, Bollmann1982.

[50] H.-J. Bunge, Texture analysis in materials science: mathematical methods, Elsevier2013.

[51] H. Aaron, G. Bolling, Free volume as a criterion for grain boundary models, Surf. Sci. 31 (1972) 27- 49.

[52] J.W. Cahn, Transitions and phase equilibria among grain boundary structures, Le Journal de Physique Colloques 43(C6) (1982) C6-199-C6-213.

[53] P.R. Cantwell, M. Tang, S.J. Dillon, J. Luo, G.S. Rohrer, M.P. Harmer, Grain boundary complexions, Acta Mater. 62 (2014) 1-48.

[54] S.J. Dillon, K. Tai, S. Chen, The importance of grain boundary complexions in affecting physical properties of polycrystals, Curr. Opin. Solid State Mater. Sci. 20(5) (2016) 324-335.

[55] Z. Yu, P.R. Cantwell, Q. Gao, D. Yin, Y. Zhang, N. Zhou, G.S. Rohrer, M. Widom, J. Luo, M.P. Harmer, Segregation-induced ordered superstructures at general grain boundaries in a nickel-bismuth alloy, Science 358(6359) (2017) 97-101.

[56] R. Besson, S. Guyot, A. Legris, Atomic-scale study of diffusion in A15 Nb3Sn, Phys. Rev. B 75(5) (2007) 054105.

[57] D.N. Seidman, Subnanoscale studies of segregation at grain boundaries: Simulations and experiments, Annu. Rev. Mater. Res. 32(1) (2002) 235-269.

[58] C.T. Liu, C. White, J. Horton, Effect of boron on grain-boundaries in Ni3Al, Acta Metall. 33(2) (1985) 213-229.

[59] A. Ruban, H.L. Skriver, J.K. Nørskov, Surface segregation energies in transition-metal alloys, Phys. Rev. B 59(24) (1999) 15990.

[60] J. Charlesworth, I. Macphail, P. Madsen, Experimental work on the niobium-tin constitution diagram and related studies, J. Mater. Sci. 5(7) (1970) 580-603.

[61] H. Müller, T. Schneider, Heat treatment of Nb3Sn conductors, Cryogenics 48(7-8) (2008) 323-330.

[62] L. Harrison, Influence of dislocations on diffusion kinetics in solids with particular reference to the alkali halides, Trans. Faraday Society 57 (1961) 1191-1199.

[63] Y. Mishin, C. Herzig, Grain boundary diffusion: recent progress and future research, Materials Science and Engineering: A 260(1-2) (1999) 55-71.

[64] A. Paul, T. Laurila, V. Vuorinen, Microstructure, diffusion and growth mechanism of Nb3Sn superconductor by bronze technique, Superconductor, InTech2010.

[65] H. Farrell, G. Gilmer, M. Suenaga, Grain boundary diffusion and growth of intermetallic layers: Nb3Sn, J. Appl. Phys. 45(9) (1974) 4025-4035.

[66] T. Laurila, V. Vuorinen, A. Kumar, A. Paul, Diffusion and growth mechanism of Nb3Sn superconductor grown by bronze technique, Appl. Phys. Lett. 96(23) (2010) 231910.

[67] D.L. Hall, New Insights into the Limitations on the Efficiency and Achievable Gradients in Nb3Sn SRF Cavities, Ph.D. thesis, Cornell University, Ithaca, NY, USA, 2017.

[68] Effects of Sn-deficient regions and surface chemistry on the performance of Nb3Sn cavities are investigated utilizing STEM-EDS and XPS, respectively. Their effects are not significant at least until the accelerating electric field of ~15 MV/m (*unpublished*).

[69] J. Carlson, A. Pack, M.K. Transtrum, S. Posen, J. Lee, D.N. Seidman, D.B. Liarte, N. Sitaraman, A. Senanian, T. Arias, J.P. Sethna, Analysis of magnetic vortex dissipation in Sn-segregated boundaries in Nb3Sn superconducting radiofrequency (SRF) cavities, *in preparation*.

[70] A.R. Pack, J. Carlson, S. Wadsworth, M.K. Transtrum, Role of inhomogeneities for vortex nucleation in superconductors within time-dependent Ginzburg-Landau theory, arXiv e-prints, 2019. (arXiv:1911.02132)

[71] N.S. Sitaraman, J. Carlson, A.R. Pack, R.D. Porter, M.U. Liepe, M.K. Transtrum, T.A. Arias, *Ab Initio* study of antisite defects in Nb3Sn: Phase diagram and impact on superconductivity, arXiv e-prints, 2019. (arXiv:1912.07576)

[72] D. Rodrigues, C. Thieme, D.G. Pinatti, S. Foner, Grain boundary compositions, transport and flux pinning of multifilamentary Nb3Sn wires, IEEE Transactions on Applied Superconductivity 5(2) (1995) 1607-1610.

[73] M. Suenaga, W. Jansen, Chemical compositions at and near the grain boundaries in bronze‐processed superconducting Nb3Sn, Appl. Phys. Lett. 43(8) (1983) 791-793.

[74] M. Suenaga, Optimization of Nb3Sn, IEEE Transactions on Magnetics 21(2) (1985) 1122-1128.

[75] E. Wallach, J. Evetts, The development of microstructure in multifilamentary bronze route A15 composites, Advances in Cryogenic Engineering Materials, Springer, 1986, pp. 911-923.

[76] R. Scanlan, W. Fietz, E. Koch, Flux pinning centers in superconducting Nb3Sn, J. Appl. Phys. 46(5) (1975) 2244-2249.

**Table 1**. Summary of Gibbsian interfacial excesses (Гi) and the five macroscopic degrees of freedom (DOF) of selected [12̅0]-tilt-GBs.

| Number | ГSn            | Five<br>macroscopic degrees |              | Methods          |
|--------|----------------|-----------------------------|--------------|------------------|
|        |                |                             |              |                  |
|        | (atom/nm2<br>) | of freedom<br>(DOFs)        |              |                  |
|        |                |                             |              |                  |
|        |                | Disorientation              | GB           |                  |
|        |                |                             |              |                  |
|        |                | (c, θ)                      | plane<br>(n) |                  |
|        |                |                             |              |                  |
| GB1-1  |                | [12̅0], 46.6o               | (215)        |                  |
|        |                |                             |              |                  |
| GB1-2  | 14.5 ± 2.3     | [12̅0], 46.6o               | (211̅)       |                  |
|        |                |                             |              |                  |
| GB2    | 16.0<br>± 6.4  | [12̅0], 4.7o                | (001)        | HR-STEM<br>/ EDS |
|        |                |                             |              |                  |
|        |                |                             |              |                  |
| GB3    | 14.6 ± 1.5     | [12̅0], 30.0o               | (212)        |                  |
|        |                |                             |              |                  |
| GB4    | 18.5 ± 7.8     | [12̅0],<br>17.5o            | (215)        |                  |
|        |                |                             |              |                  |

![](1__page_25_Figure_0.jpeg)

**Caption:** Figure 1 comprises (a) an SEM image of a Nb3Sn coating on Nb showing surface morphology tilted from 52°, (b) a HAADF-STEM cross-sectional view revealing layer deposition, and (c) a schematic outlining temperature profiles of Sn coating processes A and B. Process B's higher temperatures and absence of final Sn flux in coating stages highlight the significant effect of processing conditions on Sn distribution and grain boundary engineering.


**Figure 1**. (a) SEM micrograph of the surface of a Nb3Sn coating on Nb tilted from the optic axis of the electron beam by 52<sup>o</sup> ; (b) HAADF-STEM cross-sectional micrograph of a Nb3Sn coating on Nb; (c) Schematic temperature profiles of the Sn source and Nb substrates (furnace) for Nb3Sn coatings on Nb employing processes A or B. For process B, the temperatures of the Sn source and Nb substrates are 20 <sup>o</sup>C higher than for process A. Additionally, for process B, the external Sn source is depleted at 30 to 60 min prior to the end of the coating period and therefore the Nb3Sn coatings experience some annealing.

![](1__page_26_Figure_0.jpeg)

**Caption:** Figure 2 presents a 3D reconstruction of Nb, Sn, and N atomic distributions across a grain boundary (GB) in a Nb3Sn sample. Iso-concentration surfaces for Nb (70 at.%), Sn (30 at.%), and N (0.3 at.%) are shown, along with concentration profiles across the GB. The Sn and Nb profiles showcase Sn segregation with a measured full-width of the segregated zone at 3.0 ± 0.2 nm, indicating significant Sn presence at the GB, accompanied by Nb depletion.


**Figure 2**. (a) 3-D reconstructed Nb, Sn, and N atomic distribution-maps. Iso-concentration surfaces of Nb (70 at.% Nb), Sn (30 at.% Sn), and N (0.3 at.% N) are indicated in the 3-D reconstructed maps of each element, upper right-hand figure. (b) Concentration profiles across a GB plane displaying Sn segregation and concomitantly Nb depletion. The measured full-width of the segregated zone is 3.0 ± 0.2 nm: the segregated zone is defined as the full-width of the Sn segregation and Nb depletion profiles as indicated by the two vertical blue-lines; (c) Nitrogen concentration profile across a GB.

![](1__page_27_Figure_0.jpeg)

**Caption:** Figure 3 on page 27 includes (a) a BF-TEM image of a Nb3Sn nanotip prepared for atom probe tomography (APT) and (b) corresponding 3D reconstructions of Nb, Sn, and N distributions, indicating GB traces. Sn and N demonstrate substantial segregation at GBs in the 1D profile, underlined by a pronounced N concentration profile that aids in understanding compositional variances at the atomic level within the nanostructure .


**Figure 3** (a) BF-TEM image of a nanotip of Nb3Sn for the APT experiment and corresponding 3-D reconstruction of Nb, Sn, and N atoms: traces of the GB plane are also indicated. (b) 1-D concentration profiles of Nb, Sn, and N across a GB displaying segregation of Sn and N at this GB. (c) Nitrogen (N) concentration profile.

![](1__page_28_Figure_0.jpeg)

**Caption:** Figure 4 provides an atomically resolved HR-STEM image of a high-angle (HA) tilt grain boundary (GB) in Nb3Sn with θ = 46.6°, along with its periodic structural units visualized via filtered micrographs. EDS analysis demonstrates Sn concentration peaks at 32 at.% Sn at the GB, exceeding concentrations within grains, highlighting the compositional differences at atomic scale.


**Figure 4**. (a) Atomically resolved HR-STEM micrograph of a high-angle tilt-GB (θ = 46.6<sup>o</sup> , **c**= [12̅0]) in Nb3Sn, where the normal to the GB plane is **n** = [215]. CSLs at the GB are marked with yellow dotted-circles in the magnified image. (b) Filtered HR-STEM micrograph displaying a periodic array of structural units (cores of GB dislocations) of the HA-GB, indicated by yellow pentagons. (c) EDS concentration profiles

across the HA-GB demonstrating that the maximum Sn concentration at the GB is ~32 at.% Sn, which is higher than in the matrix, 25 at.% Sn. Note the depletion in the Nb concentration profile.

![](1__page_29_Figure_1.jpeg)

**Caption:** Figure 5 presents atomically resolved HR-STEM images of [12̅0]-tilt grain boundaries (GBs) in Nb3Sn with varying tilt angles: (a) 4.7° with a normal to the GB plane as [001], (b) 17.5° with a GB plane normal [215]. In both micrographs, the Gibbsian interfacial excesses of Sn (ГSn) are plotted as a function of GB disorientation angle in (c). Red squares denote [12̅0] rotation axes at different angles, with the black squares for random rotation axes. The transmission-EBSD color map inset helped determine disorientation angles. Key findings indicate no clear correlation between disorientation angles and Sn segregation, shedding light on factors affecting Sn excess at GBs.


**Figure 5**. Atomically resolved HR-STEM images of [12̅0]-tilt-GBs in Nb3Sn with different tilt angles (θ) and GB planes (n): (a) θ = 4.7<sup>o</sup> /**n** = [001] and (b) θ = 17.5<sup>o</sup> /**n** = [215]. (c) Plot of the Gibbsian interfacial excesses of Sn (ГSn) as a function of the disorientation angle of GBs. The red-squares are for a [12̅0] rotation axis with different tilt-angles and black-squares are for random rotation axes. A transmission-EBSD color map of a TEM sample is inset in this plot, which was used to identify the disorientation angles of the GBs.
![](2__page_30_Figure_0.jpeg)

**Caption:** Figure 6 demonstrates the Gibbsian interfacial excess of Sn at grain boundaries in Nb3Sn films, dependent on Sn flux and process conditions. Distinctions between Process A and Process B are shown: Process A involves continuous Sn vapor supply, while Process B increases substrate temperatures without a Sn flux towards the coating's end. This segregation behavior is crucial for optimizing coating techniques to control Sn distribution and achieve desired Nb3Sn properties.


**Figure 6**. Gibbsian interfacial excesses of Sn at GBs in Nb3Sn films coated using processes A or B, which differ with respect to the average Sn flux and the temperatures of the Nb substrates. For process A, Nb3Sn films were deposited on Nb substrates at 1100 <sup>o</sup>C with the Sn source at 1200 <sup>o</sup>C; the Sn vapor was supplied during the entire coating process. For process B, the temperature of both the Nb substrate and Sn source were increased by 20 <sup>o</sup>C during creation of the Nb3Sn films; there wasn't, however, a Sn flux for the last 30-60 min of this process.

![](2__page_31_Figure_0.jpeg)

**Caption:** Figure 7 features concentration profiles across a grain boundary and supporting HR-STEM images of Nb3Sn post-annealing at 1100°C for 3 hours, initially exhibiting Sn segregation. The annealing disrupts Sn concentration, resulting in Nb segregation with an estimated Gibbsian interfacial excess of 22 ±3 atoms/nm² across approximately a 5 ±1 nm region. This change conveys significant details on thermal effects influencing elemental distributions at grain boundaries.


**Figure 7**. Concentration profiles across a GB and the corresponding HR-STEM micrograph of a Nb3Sn sample, which originally exhibited Sn segregation at GB in the as-coated state (A5) and then was annealed at 1100 <sup>o</sup>C for 3 h. Tin segregation disappears and concomitantly Nb segregation is observed. The Gibbsian interfacial excess of Nb is estimated to be 22 ±3 atom/nm<sup>2</sup> and the full-width of the Nb segregated region is ~5 ±1 nm.

![](2__page_32_Figure_0.jpeg)

**Caption:** Figure S2 highlights filtered high-resolution HAADF-STEM images of a high-angle [12̅0] tilt grain boundary (GB) in Nb3Sn. The atomic planes are indexed, revealing a GB plane with a normal vector, n = [211̅], and showcasing potential chemical ordering through bright contrast effects observed at CSL sites via intensity profiles, suggesting possible compositional variations at these sites.


**Figure 8.** (a) Filtered atomic-resolution HAADF-STEM micrograph of a 46.6<sup>o</sup> /[12̅0]tilt-GB of Nb3Sn whose GB plane is (211̅ ). (b) The [12̅0] tilt GB structural model determined from first-principles calculations. Site 1 or 2 is the Sn occupation site with a larger free volume space, and site 3 is an Sn antisite in the Nb3Sn bulk. The GB structure was fully relaxed.

![](2__page_33_Figure_0.jpeg)

**Caption:** Figure 9 on page 33 describes the Sn interfacial excess profile along a depth direction of a Nb3Sn grain boundary, maintaining a consistent Gibbsian interfacial excess of Sn from surface to substrate. This uniformity implies rapid short-circuit Sn diffusion with no major Sn concentration gradient along the boundary, a crucial insight into the mechanism governing Sn diffusion within Nb3Sn films.


**Figure 9.** Gibbsian interfacial excesses of Sn at a Nb3Sn GB along the depth direction. The values of the Gibbsian interfacial excesses of Sn are approximately a constant along the GB from the surface (point 1) to the Nb substrate (point 4), indicative of fast short-circuit diffusion of Sn along this GB, in the absence of a significant Sn concentration gradient.

![](2__page_34_Figure_0.jpeg)

**Caption:** Figure 10 illustrates (a) the Quality Factor (Q0) versus Accelerating Electric Field (Eacc) for Nb3Sn superconducting radio frequency (SRF) cavities at 4.4 K. Tin concentration profiles are depicted for Cornell cavity 2 and Fermilab cavity 3 in panels (b) and (c), respectively, revealing the absence of Sn segregation at GBs. This provides a contrast to low-performance cavities where Sn segregation is prevalent, and signifies that controlling segregation could enhance cavity performance.


**Figure 10**. (a) Quality factor (Q0) versus accelerating electric field (Eacc) curves of Nb3Sn SRF cavities measured at 4.4 K. (b) and (c): Tin concentration profiles across GBs for a high-performance Nb3Sn SRF cavities measured by STEM-EDS; (b) Cornell cavity 2 and (c) Fermilab cavity 3. We note that the highperformance cavities, Cornell cavity 2 and Fermilab cavity 3 reveal no Sn segregation at GBs.

For the 3-D reconstructions in **Figs 2** and **3**, all the isotopes of Nb and Sn are selected including the oxide states for Nb-O<sup>+</sup> and Nb-O2+ . The full width of each peak is selected for the reconstructions except Nb2+ and Nb3+, which have wide peak spreads. For nitrogen, N<sup>+</sup> and N<sup>2</sup> + are selected for the 3-D reconstructions.

![](2__page_35_Figure_1.jpeg)

**Caption:** Figure S.1 displays the mass spectrum obtained from atom-probe tomographic analyses of a Nb3Sn sample labeled A5. The spectrum captures a range of mass-to-charge ratios indicative of different atomic species present, showcasing distinct isotopic peaks for Nb and Sn, along with their oxides, providing insights into atomic distribution and elemental composition critical for understanding segregation phenomena.


**Figure S.1** Mass spectrum for atom-probe tomographic analyses of a Nb3Sn sample (A5)

![](2__page_36_Figure_0.jpeg)

**Caption:** Figure S4 illustrates HAADF-STEM imaging of a high-angle (46.6°) [12̅0]-tilt GB in Nb3Sn. Contrast effects at CSL sites suggest possible ordering at the GB. The intensity profile exhibits high intensities at CSL sites, implicating a higher Sn concentration and providing evidence for the localized chemical environment's influence at GBs.


**Figure S.2** (a) Magnified filtered high-resolution high angle annular dark field (HAADF) scanning transmission electron microscopy (STEM) image of a high-angle (46.6<sup>o</sup> ) [12̅0] tilt GB of Nb3Sn. (b) The atomic planes are idexed and the GB plane is given by **n** = [211̅].

![](2__page_37_Figure_0.jpeg)

**Caption:** Figure S.3 depicts Nb and Sn concentration profiles across a grain boundary of an annealed Nb3Sn sample at 1100°C for 3 hours. The study shows how initial Sn segregation dissipates upon annealing, leading to Nb segregation with a Gibbsian interfacial excess of approximately 20 ± 3 atoms/nm² across a ~5 nm region. This replacement highlights the material's complex diffusion dynamics and the effects of thermal treatment on modifying alloy composition at grain boundaries.


**Figure S.3** Niobium (Nb) and tin (Sn) concentration profiles across another GB of an annealed Nb3Sn sample at 1100 <sup>o</sup>C for 3 h. This figure demonstrates that when Sn segregation disappears then Nb segregation (20 ± 3 atom/nm<sup>2</sup> ) appears at the same GB with a full width of ~5 nm.

![](2__page_38_Figure_0.jpeg)

**Caption:** Figure S.3 depicts Sn and Nb concentration profiles across a GB from an annealed Nb3Sn sample at 1100°C for 3 hours. As Sn segregation dissipates, concurrent Nb segregation emerges, with a significant interfacial excess and widened profile. This provides evidence for annealing effects altering elemental distribution at GBs.


**Figure S.4** (a) high resolution HAADF- STEM image and contrast profiles across a 46.6<sup>o</sup> /[12̅0]-tilt-GB of Nb3Sn. Bright contrast effects are observed at the coincident site lattice (CSL) sites of the GB, indicating possible chemical ordering at this Nb3Sn GB. (b) The intensity profile across CSL sites near the GB displays high intensities at the CSL.

![](2__page_39_Figure_0.jpeg)

**Caption:** Figure S.5 details the interior composition of Nb3Sn grains using EDS. Part (a) highlights varying Sn interfacial excess at GBs across different coatings using HR-STEM-EDS. Each data point is a distinct coat, averaging Sn segregation impacts across 1 to 3 GBs. Part (b) features HAADF-STEM images and Sn L-maps of partially Sn-deficient Nb3Sn grains, showing Sn segregation at GBs where grain interiors are <25% Sn at a GB, validated by a noticeable Sn excess in the EDS profile. This demonstrates significant Sn-atom drift affecting accurate measurement of Sn profiles, revealing larger Sn segregated regions in some cases.


**Figure S.5** (a) The compositions of interior Nb3Sn grains with different levels of interfacial excesses of Sn at GBs. Each data point represents each coating: Tin excess at GBs on the ordinate scale is an average of Sn segregation effects at 1 to 3 GBs in each coating, estimated by HR-STEM-EDS measurements. The composition of the interior Nb3Sn grains on the abscissa is an average of 1 to 3 Nb3Sn grains of each coating, estimated by APT analyses. The limitation of this plot is that it it is difficult to think that the composition of 2-3 Nb3Sn grains represents the composition of the whole Nb3Sn sample of each coating. It may provide, however, a clue indicating that there isn't a strong correlation between the composition of Nb3Sn grains and Sn segregation at GBs. (b) HAADF-STEM image and EDS Sn L-map of Nb3Sn grains with Sn-deficient regions. Note that a GB in partially Sn-deficient grains (<25 at.% Sn) displays Sn segregation at the GB employing an EDS measurement. A large width of the Sn segregated region (~8 nm) in the concentration profile is due to the drift of the TEM sample during data acquisition, which overestimates the Sn signal, but the existence of Sn segregation is clear.