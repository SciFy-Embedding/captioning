# arXiv:2302.07201

**Paper ID:** 91da227359c679ba7286f5b8b838a334

**PDF Path:** apl/Superconductors/arXiv:2302.07201.pdf

**Processing Status:** complete

**Captions Added:** 35

**Generated:** 2025-06-24T13:44:27.744500

---

# Development of a prototype superconducting radio-frequency cavity for conduction-cooled accelerators

G. Ciovati,1, 2, [∗](#page-0-0) J. Anderson,<sup>3</sup> S. Balachandran,<sup>1</sup> G. Cheng,<sup>1</sup> B. Coriton,3, [†](#page-0-1) E. Daly,<sup>1</sup> P. Dhakal,<sup>1</sup> A.

Gurevich,<sup>2</sup> F. Hannon,1, [‡](#page-0-2) K. Harding,<sup>1</sup> L. Holland,<sup>3</sup> F. Marhauser,1, [§](#page-0-3) K. McLaughlin,<sup>3</sup> D. Packard,<sup>3</sup> T.

Powers,<sup>1</sup> U. Pudasaini,<sup>1</sup> J. Rathke,<sup>4</sup> R. Rimmer,<sup>1</sup> T. Schultheiss,<sup>5</sup> H. Vennekate,<sup>1</sup> and D. Vollmer<sup>3</sup>

<sup>1</sup>Thomas Jefferson National Accelerator Facility, Newport News, VA 23606, USA

<sup>2</sup>Center for Accelerator Science, Department of Physics, Old Dominion University,

Norfolk, Virginia 23529, USA

<sup>3</sup>General Atomics, San Diego, CA 92121, USA

<sup>4</sup>TECHSOURCE, Inc., Los Alamos, NM 87544, USA

<sup>5</sup>TJS Technologies LLC, Commack, NY 11725, USA

(Dated: 3-22-2023)

The higher efficiency of superconducting radio-frequency (SRF) cavities compared to normalconducting ones enables the development of high-energy continuous-wave linear accelerators (linacs). Recent progress in the development of high-quality Nb3Sn film coatings along with the availability of cryocoolers with high cooling capacity at 4 K makes it feasible to operate SRF cavities cooled by thermal conduction at relevant accelerating gradients for use in accelerators. A possible use of conduction-cooled SRF linacs is for environmental applications, requiring electron beams with energy of 1 − 10 MeV and 1 MW of power. We have designed a 915 MHz SRF linac for such an application and developed a prototype single-cell cavity to prove the proposed design by operating it with cryocoolers at the accelerating gradient required for 1 MeV energy gain. The cavity has a ∼ 3 µm thick Nb3Sn film on the inner surface, deposited on a ∼ 4 mm thick bulk Nb substrate and a bulk ∼ 7 mm thick Cu outer shell with three Cu attachment tabs. The cavity was tested up to a peak surface magnetic field of 53 mT in liquid He at 4.3 K. A horizontal test cryostat was designed and built to test the cavity cooled with three Gifford-McMahon cryocoolers. The rf tests of the conduction-cooled cavity, performed at General Atomics, achieved a peak surface magnetic field of 50 mT and stable operation was possible with up to 18.5 W of rf heat load. The peak frequency shift due to microphonics was 23 Hz. These results represent the highest peak surface magnetic field achieved in a conduction-cooled SRF cavity to date and meet the requirements for a 1 MeV energy gain.

#### I. INTRODUCTION

The superconducting radio-frequency (SRF) technology is widely used in modern particle accelerator research facilities because of its higher efficiency compared to normal-conducting rf technology [\[1,](#page-20-0) [2\]](#page-20-1). SRF accelerators require liquid helium (LHe) cryoplants, often subcooled to 2 K, to cool the cavities well below the superconducting transition temperature. Whereas niobium has been the material of choice for SRF cavities, Nb3Sn has emerged as a viable alternative in recent years [\[3\]](#page-20-2). Since the critical temperature of Nb3Sn is nearly twice that of Nb, similar rf dissipation as that of Nb can be achieved at 4.3 K, instead of 2 K, at fields much smaller than the thermodynamic critical field, Bc, which reduces significantly the cost and complexity of the LHe cryoplant. The cooling power at 4 K from commercial closed-cycle refrigerators (CCR), also known as cryocoolers, has also been steadily increasing in the last few years, making it feasible to design SRF accelerators in which the cavities are cooled by thermal conduction. Proof of principle demonstration of conduction-cooled SRF cavities have been achieved in single-cell cavities at 650 MHz [\[4\]](#page-21-0), 1.5 GHz [\[5\]](#page-21-1) and 2.6 GHz [\[6\]](#page-21-2), up to a peak surface magnetic field B<sup>p</sup> ∼ 42 mT, with different types of cryocoolers and methods to connect the cryocooler's cold stage (stage 2) to the cavity.

Conduction-cooling greatly simplifies the overall design and cost of an SRF accelerator, making the SRF technology feasible for industrial accelerators. One among several possible applications of SRF industrial accelerators is environmental remediation, which requires continuous-wave electron beams with energies in the range 1 - 10 MeV and 1 MW beam power [\[7\]](#page-21-3). Conceptual designs for MW-class conduction-cooled SRF accelerators at 1 MeV and 10 MeV have been published [\[8,](#page-21-4) [9\]](#page-21-5). Given the high beam power requirement, the efficiency of the overall accelerator is dominated by the efficiency of the high-power rf source, particularly at low beam energy. 915 MHz magnetrons for industrial heating are the most efficient and cost effective commercial high-power rf sources, with power levels up to ∼ 100 kW, making them an ideal choice for high-power industrial accelerators [\[10\]](#page-21-6). R&D efforts are ongoing to develop efficient power-combining schemes as well as methods to achieve the amplitude and phase control required to drive SRF

<span id="page-0-0"></span><sup>∗</sup> [gciovati@jlab.org](mailto:gciovati@jlab.org)

<span id="page-0-1"></span><sup>†</sup> Current address: ITER, 13067 St. Paul-lez-Durance, France

<span id="page-0-2"></span><sup>‡</sup> Current address: Phasespace Tech, 23734 Bj¨arred, Sweden

<span id="page-0-3"></span><sup>§</sup> Current address: SCK CEN, 2400 Mol, Belgium

cavities [\[11–](#page-21-7)[18\]](#page-21-8). In this contribution we present the conceptual design of a conduction-cooled 915 MHz, 1 MeV, 1 MW cw SRF electron linac along with the development of a prototype cavity aiming at demonstrating achieving the Bp-value which corresponds to the accelerating gradient required for a 1 MeV energy gain. The article is organized as follows: Sec. [II](#page-1-0) presents the conceptual design of the 915 MHz accelerator, focused on the beam transport simulations and the SRF cavity design, Sec. [III](#page-4-0) presents the development of a multi-metallic 952.6 MHz single-cell prototype cavity and its performance in LHe at 4.3 K. Section [IV](#page-9-0) presents the results from cryogenic and mechanical tests of the Cu thermal strap used for the connection between the cavity and the cryocooler, Sec. [V](#page-10-0) presents the design of a horizontal test cryostat (HTC) for the prototype cavity with three cryocoolers, Sec. [VI](#page-12-0) presents the results from the rf tests of the prototype cavity cooled by conduction inside the HTC, as well as cavity microphonics and vibration measurements. The analysis and discussion of the experimental results are presented in Sec. [VII](#page-17-0) and a summary is given in Sec. [VIII.](#page-20-3)

# <span id="page-1-0"></span>II. DESIGN OF A 915 MHz, 1 MeV, 1 MW CW ELECTRON LINAC

The conceptual design for a 750 MHz, 1 MeV, 1 MW, cw, conduction-cooled SRF electron linear accelerator (linac) for environmental remediation was presented in Ref. [\[8\]](#page-21-4). Different options for the high-power rf source were discussed and it was estimated that ∼ 60% of the operational cost was due to electric power consumption, if either a klystron or multibeam inductive output tube were used. Magnetrons are the most efficient high-power rf sources and R&D efforts are ongoing to make them capable of the amplitude and phase regulation required to drive SRF cavities as well as efficiently combine the power from multiple magnetrons up to 1 MW. Choosing the accelerator frequency to match that of high-power industrial magnetrons lowers the capital cost of the highpower rf source, compared to a frequency for which only a few devices are commercially available.

#### A. Beam transport simulations

The accelerator design described in Ref. [\[8\]](#page-21-4) has been re-evaluated at 915 MHz and it consists of the same main components: an rf gridded thermionic gun, a focusing solenoid, a cryomodule with a conduction-cooled single-cell SRF cavity, another focusing solenoid, raster magnets and a beam-exit window. The beam transport simulations were carried out with with the particle tracking software General Particle Tracer (GPT) [\[19\]](#page-21-9) using a simplified geometry which included the gun, the focusing solenoid and the SRF cavity. The cathode emitting area was assumed circular and planar, with a radius of 16 mm. The center positions of the solenoid and the cavity are

<span id="page-1-1"></span>TABLE I. Main design parameters for the 915 MHz, highpower cw electron linac and the SRF cavity.

| Accelerator parameters                        |          |  |
|-----------------------------------------------|----------|--|
| Gun voltage (kV)                              | -85      |  |
| Solenoid peak magnetic field on axis (mT)     |          |  |
| SRF cavity off-crest phase                    |          |  |
| SRF cavity peak electric field on axis (MV/m) | 16       |  |
| Beam energy (MeV)                             | 1.0      |  |
| Beam current (A)                              | 1.0      |  |
| SRF cavity geometry                           |          |  |
| Equator diameter (mm)                         | 145.2    |  |
| Equator ellipse major axis (mm)               | 58.3     |  |
| Equator ellipse minor axis (mm)               | 34.3     |  |
| Wall angle (◦<br>)                            | 5        |  |
| Radius of iris circle (mm)                    | 4.9      |  |
| Iris radius (mm)                              |          |  |
| Cell length (mm)                              | 43<br>86 |  |
| SRF cavity electromagnetic parameters         |          |  |
| Ep/Eacc                                       | 2.61     |  |
| Bp/Eacc<br>[mT/(MV/m)]                        | 3.90     |  |
| G (Ω)                                         | 176.3    |  |
| R/Q (Ω)                                       | 26.2     |  |
|                                               |          |  |

18 cm and 60 cm from the cathode, respectively. The main design parameters are listed in Table [I,](#page-1-1) resulting in a final kinetic energy of 1 MeV, with a root-mean-square (rms) energy spread of 100 keV at the exit of the cavity. The transverse rms beam radius is less than 14 mm inside the cavity, whereas the longitudinal rms bunch length is less than 10 mm, as shown in Fig. [1.](#page-2-0) None of the electrons in the simulation hit the cavity inner surface.

### <span id="page-1-2"></span>B. Cavity design

According to the GPT simulation, the electric field onaxis which results in a beam energy of 1 MeV can be achieved with a 915 MHz single-cell cavity with an iris-toiris cell length of 86 mm, which corresponds to a geometric β of 0.53, where β is the speed of the electron relative to the speed of light. The cell shape was designed as a compromise between minimizing the peak surface fields, maximizing the cavity aperture, ease of surface treatments and mechanical stability. The parameterization of the cell shape followed the one described in Ref. [\[20\]](#page-21-10) and the final value of the geometric parameters are listed in Table [I.](#page-1-1) The diameter of the beam tube was enlarged to 148 mm on one side of the cavity to facilitate the propagation of higher order modes (HOMs). The main electromagnetic parameters obtained from a simulation with the electromagnetic field solver Superfish [\[21\]](#page-21-11) are listed in Table [I.](#page-1-1) E<sup>p</sup> and B<sup>p</sup> are the peak surface electric and magnetic fields, respectively, Eacc is the effective accelerating field, G is the geometry factor and R/Q is linac definition of the characteristic the shunt impedance. The value of the operational peak electric field on axis listed in Table [I](#page-1-1) corresponds to Eacc = 10.6 MV/m, B<sup>p</sup> = 41.5 mT and

![](_page_2_Figure_0.jpeg)

**Caption:** Figure depicts beam dynamics simulations for the 915 MHz cavity with (a) illustrating average kinetic energy versus position and (b) showcasing beam sizes both horizontally and longitudinally. The plot provides insights on optimization strategies for radio-frequency quadrupole and superconducting cavity designs to enhance energy efficiency and beam quality in particle acceleration applications.


<span id="page-2-0"></span>FIG. 1. Average beam energy (a), transverse rms beam size and longitudinal rms bunch length (b) along the beam axis for a 915 MHz, 1 MeV, 1 A linac obtained from a simulation with GPT. The schematic layout on top is not to scale.

E<sup>p</sup> = 27.8 MV/m. Multipacting (MP) simulations were carried out with the code FishPact [\[22\]](#page-21-12) and no stable electron trajectories with a final impact energy > 25 eV were found below Eacc = 12.2 MV/m, which is above the operational Eacc-value.

Two identical coaxial fundamental power couplers (FPCs) are mounted to side ports, 180◦apart, on the small-diameter beam tube of the cavity. Each FPC needs to provide 457 kW of rf power, which corresponds to an external-Q value of Qext = 2.7 × 10<sup>4</sup> at the target Eacc. The relatively low Qext-value is due to the high beam loading and results in a bandwidth of the cavity resonance of ∼ 34 kHz. This relatively large bandwidth

makes the cavity less susceptible to detuning due to microphonics. The distance to the cavity iris and the length of the inner conductor were set to achieve this Qext. The tip of the FPC's center conductor has a "pringle" shape to maximize the coupling to the electric field while minimizing the penetration (∼ 1 mm) inside the beam tube. The FPC design was scaled to 915 MHz from a well established one for a high-current SRF injector [\[23\]](#page-21-13).

HOM analyses were carried out using the threedimensional (3D) electromagnetic software package CST Studio Suite [\[24\]](#page-21-14) following the same methodology described in Refs. [\[8](#page-21-4) and [25\]](#page-21-15). Both wakefield and lossy Eigenmode simulations were carried out, in good agreement to each other, to assess the most parasitic beaminduced longitudinal and transverse HOM impedances. The simulations confirmed, by utilizing beamline absorbers placed outside the cryomodule, that the dipole HOM impedances are kept well below the corresponding threshold impedance of a single-pass beam breakup instability, while the dissipated power levels in the absorber, dominated by longitudinal HOMs, can be well handled by a conventional absorber.

#### <span id="page-2-1"></span>C. Cavity thermal analysis

The cavity is intended to be built from 4 mm thick high-purity Nb sheets. A few micrometers-thick Nb3Sn thin film would be formed on the cavity inner surface by the vapor diffusion method [\[3,](#page-20-2) [26\]](#page-21-16). The cavity flanges are made of bulk Nb and vacuum sealing with mating flanges is done by In wire.

The integrated cavity and cryomodule design described in Ref. [\[8\]](#page-21-4) used a thick high-purity Cu layer deposited on the cavity outer surface to improve the thermal stabilization of the system. An experimental proof of concept was achieved in a 1.5 GHz single-cell cavity in which a high-purity Cu layer of thickness ≥ 5 mm as well as an equatorial bolt-on ring were grown on the outer surface of a Nb/Nb3Sn cavity by electroplating, with a thin interface layer deposited by cold-spray [\[5\]](#page-21-1). Growing the structures to connect the cavity to the cryocoolers' 2nd stage by electroplating can be very time consuming, therefore we propose a different approach consisting of a two-steps electroplating. In the first step, a Cu layer ∼ 2 mm thick is electroplated onto the thin cold-sprayed Cu layer. The layer is machined on a lathe to create attachment posts. An equator ring and a beam-tube plate machined from oxygen-free high-conductivity (OFHC) Cu (C10100 alloy) are mounted to the attachment posts on the cavity and electroplated in place during the second electroplating step, adding ∼ 5 mm of Cu. A final lathe machining of the Cu structure is needed to achieve the final dimensions and surface finish of the cavity outer layer. The thickness of the equator ring and beam-tube plate tapers down towards the attachment posts, allowing the electroplated Cu to build-up in those areas. The thermal conductivity of the electroplated Cu was measured on

samples and corresponds to that of Cu with a residual resistivity ratio of ∼ 300 [\[27\]](#page-21-17).

Figure [2](#page-3-0) shows a 3D layout used for the thermal analysis of the cavity, FPCs, thermal shield and four cryocoolers with the finite element code Ansys [\[28\]](#page-21-18). The temperature-dependent thermal conductivities of stainless steel and niobium used in the thermal analysis were the same as those used in Ref. [\[8\]](#page-21-4). The thermal conductivity of Cu with RRR = 100 from Ref. [\[29\]](#page-21-19) was used for OFHC Cu. The thermal conductivity of the electroplated Cu was taken from Ref. [\[27\]](#page-21-17). The Nb3Sn thin film was not included in the model for simplification. Considering the thermal conductivity of Nb3Sn from Ref. [\[30\]](#page-21-20), the thermal conductance per unit area of 2 µm thick Nb3Sn at 4 K is comparable to that of 4 mm thick Nb with residual resistivity ratio (RRR) ∼ 300.

![](_page_3_Figure_3.jpeg)

**Caption:** This figure outlines the diagram of a multi-stage cooling system designed to maintain the thermal characteristics of a superconducting cavity. Stages 1-1 through 2-4 illustrate different cooling stations interconnected around the cavity, which are integral to supporting conduction-cooling techniques pivotal for maintaining superconductive properties consistently during operation.


<span id="page-3-0"></span>FIG. 2. 3D model used for the thermal analysis. Each color represents a different material: cyan indicates Nb, blue indicates stainless steel, orange represents electroplated Cu, and fuchsia represents Cu with RRR=100. The nomenclature "Stage 1-3" means stage 1 of cryocooler number 3.

The cryocoolers are Gifford-McMahon (GM) type (model RDE-418D4, Sumitomo Cryogenics of America, Allentown, PA) with a nominal cooling power of 2 W at 4 K. The full cryocooler heat capacity map provided by the vendor was used in the thermal analysis and a conservative 2% derating factor was applied to the cryocooler mounted upside-down, as suggested by the vendor. The stage 2 temperature as a function of the heat load applied to the same stage was measured at Jefferson Lab for the same cryocooler model and it is shown in Fig. [3.](#page-3-1) The temperature of the stage 2 does not change significantly for a heat load of up to ∼ 80 W applied to stage 1 and

![](_page_3_Figure_6.jpeg)

**Caption:** This caption highlights the results from frequency spectrum analysis using Fast Fourier Transform (FFT) showing vertical acceleration impacts on the cavity. Vibration analysis from both the HTC support cart and CCR1 peak frequency data elucidates microphonic influence on cavity operation which can lead to detuning and frequency shifts.


![](_page_3_Figure_7.jpeg)

**Caption:** This figure presents the correlation between heat load variations applied to stage 2 and the corresponding stage 2 temperature changes. It includes conditions of applying external heat loads of 5 W and 55 W on stage 1. These performance measurements are vital to understanding the efficiency of thermal management systems used in conduction-cooled SRF applications and determining critical operational limits to prevent thermal instability.


<span id="page-3-1"></span>FIG. 3. Stage 2 temperature as a function of the stage 2 heat load for two different stage 1 heat load values for a GM cryocooler model RDE418-D4 measured at Jefferson Lab.

Copper thermal straps were envisioned to connect stage 2 of the cryocoolers to the cavity. A temperature-independent surface contact conductance of 172.4 kW/(m<sup>2</sup> K), derived from Ref. [\[31\]](#page-22-0), was used in the thermal analysis. The temperatures of the outer surface of the beam tubes and FPCs were fixed at 300 K. A static heat load of 1 W was applied uniformly to the cavity outer surface. The rf power density applied to the cavity's inner surface is given by <sup>1</sup> <sup>2</sup>R<sup>s</sup> (B/µ0) 2 , where B is the magnetic field over the cavity surface for the TM<sup>010</sup> accelerating mode and R<sup>s</sup> is the surface resistance of Nb3Sn. Nb3Sn cavities often display a significant field dependence of the surface resistance. For the thermal analysis, we considered Rs(Bp, T) = a<sup>0</sup> +a1B<sup>p</sup> +a2B<sup>2</sup> <sup>p</sup> +a3B<sup>3</sup> <sup>p</sup> +a4exp(−a5/T), where the 3rd order polynomial term is an empirical fit of the Rs(Bp) measured at 4.3 K in the prototype Nb3Sn/Nb cavity discussed in Sec. [III B](#page-6-0) and the last term is a least-squares fit of the Bardeen-Cooper-Schrieffer (BCS) surface resistance computed numerically with the code developed by Halbritter [\[32\]](#page-22-1). Figure [4](#page-4-1) shows the empirical Rs(Bp) at three different temperatures used in the analysis. Finally, the rf heat load from the two FPCs, each with 474 kW of input power [\[33\]](#page-22-2), is included in the model. A target Bp-value of 47 mT was chosen for the thermal analysis, corresponding to a ∼ 13% margin above the operational value. The thermal conductance of the thermal strap is used as a parameter to obtain convergence in the iterative simulation at the target B<sup>p</sup> = 47 mT.

The finite element solution was solved with a two loop iterative procedure. The inner loop required setting of the CCR stage temperatures and iterating temperatures and rf heat loads determined by rf resistance and fields. The outer loop consists of reading the CCRs' heat load

![](_page_4_Figure_1.jpeg)

**Caption:** This figure shows the setup layout for measurement of Q0, the cavity's intrinsic quality factor, with variations according to differing cool-down conditions. This highlights the influence of ambient magnetic fields resulting from cooling cycles, especially on achieving the thermal stability required for efficient superconducting performance .


<span id="page-4-1"></span>FIG. 4. Empirical field dependence of the surface resistance used in the thermal analysis calculated at three different temperatures.

reactions and looking up their temperatures from the CCR's capacity map for the next iteration. Small decreasing cryocooler temperature differences from one iteration to the next indicated a converging solution. Increasing differences from one iteration to the next signified a diverging solution. Fields were increased until we could not get a converged solution. Figure [5](#page-4-2) shows the temperature profiles the cavity, obtained from the thermal analysis for B<sup>p</sup> = 47 mT and with a thermal strap thermal conductance of 17.5 W/K at 4 K and 39.7 W/K at 10 K. The total heat load into stage 1 of the cryocoolers is 75 W. The total heat load into stage 2 of the four cryocoolers is 13.2 W, which is the sum of 0.7 W of radiant heat from the FPCs' inner conductors and pringle tip, 3.3 W of rf heat load from the Nb3Sn, 1 W of static heat load distributed on the cavity outer surface and 8.2 W of ambient conduction from the beam pipes and FPCs' outer conductors.

#### <span id="page-4-0"></span>III. 952.6 MHz PROTOTYPE CAVITY

The funding available for the project did not allow fabricating a prototype single-cell cavity of the same shape as the one discussed in Sec. [II B,](#page-1-2) however a 952.6 MHz Nb single-cell cavity, which was designed and built as a prototype for the proposed Jefferson Lab Electron Ion Collider [\[34\]](#page-22-3) was available after it had successfully completed the surface processing and cryogenic rf test [\[35\]](#page-22-4). The cavity frequency is sufficiently close to 915 MHz for the cavity to serve as a valid prototype to prove the concepts for conduction-cooled SRF cavities discussed in Sec. [II.](#page-1-0) The cell length is 141 mm, the shape of the equator is circular with 54.8 mm radius, the shape of the iris is elliptical

![](_page_4_Figure_6.jpeg)

**Caption:** This figure displays the temperature distribution across a prototype SRF cavity at operating conditions calculated using Ansys for a magnetic field of B<sup>p</sup> = 47 mT at a radio frequency (RF) heat load of 16.9 W. The temperature map uses a color gradient scale, illustrating areas most susceptible to heat loading, which are key to identifying regions that may benefit from additional upgrades to thermal management systems.


<span id="page-4-2"></span>FIG. 5. Temperature distribution for the cavity with ∼ 1 MW of rf input power and operating at B<sup>p</sup> = 47 mT.

with 21.8 mm major axis and 15.9 mm minor axis, with a flat straight segment tangent to both the iris ellipse and equator circle. The iris and equator inner diameters are 110 mm and 276.2 mm, respectively. The cavity design followed the same rationale as that described in Ref. [\[36\]](#page-22-5). The main electromagnetic parameters, computed with CST Studio Suite, are listed in Table [II.](#page-5-0)

The cavity was fabricated at Jefferson Lab with conventional cavity fabrication methods, starting from 4 mm thick, high-purity Nb sheets. The beamline flanges, with 165 mm outer diameter and 11 mm thickness were made from reactor-grade Nb. Indium wire (99.99% pure) 1.52 mm in diameter was used for the vacuum sealing of the cavity flanges to 316 stainless steel circular plates with rf feedthroughs for the input and transmitted power antennas and with a pump-out port. The best cavity performance was achieved after 30 µm removal from the inner surface by electropolishing (EP), degreasing in ultrapure water with a detergent and ultrasonic agitation, high-pressure rinse (HPR) with ultra-pure water in an ISO 4 clean room, assembly of the test hardware, evacuation to ∼ 10<sup>−</sup><sup>8</sup> mbar and baking at 120 ◦C for 12 h. The results from the cryogenic rf test in a vertical cryostat filled with superfluid He at 2.0 K are shown in Fig. [6.](#page-5-1) The cavity was limited by quench at B<sup>p</sup> = 184 mT. MP was processed at B<sup>p</sup> ∼ 52 mT and 96 mT. The highest x-ray dose rate, measured at the maximum rf field on the cryostat's top plate was 65 µSv/h, which is a relatively low value.

In order to verify experimentally the conductioncooling design concept discussed in Sec. [II C,](#page-2-1) the cavity will have to be coated with a Nb3Sn film on the inner surface and a thick copper outer layer. The Nb3Sn coating process and the results from the cryogenic rf tests following the coating are discussed in Sec. [III B.](#page-6-0) The Cu coating process follows the concepts discussed in Sec. [II C:](#page-2-1)

![](_page_5_Figure_1.jpeg)

**Caption:** This figure presents the quality factor, Q<sub>0</sub>, as a function of the accelerating electric field, E<sub>acc</sub>, for a 952.6 MHz prototype cavity. The plot includes steady-state measurements showing the cavity's performance, where increased E<sub>acc</sub> indicates declining Q<sub>0</sub>, suggesting thermal instabilities might limit the operational efficiency under high power conditions.


<span id="page-5-1"></span>FIG. 6. Results from the rf test in superfluid He at 2.0 K of the 952.6 MHz prototype single-cell Nb cavity. A picture of the cavity is shown in the inset.

deposition of a thin Cu layer by cold-spray, deposition of ∼ 2 mm thick Cu layer by electroplating, assembly of an equator ring and a beam-tube attachment plate, machined from OFHC Cu. A second electroplating step, growing a ∼ 5 mm thick Cu layer which bonds the equator ring and attachment plate to the cavity, completes the formation of the Cu outer shell of the cavity. The fabrication process of the Cu outer layer is discussed in details in Sec. [III C.](#page-7-0) Figure [7](#page-5-2) shows a cross-section layout of the cavity. The choice of the thickness of the electroplated Cu layer was dictated by the need to deposit sufficient material to bond the attachment ring and plate to the cavity and to have some safety margin, in case the thermal conductivity of the Cu was not as good as that measured on the small sample. The Cu is expected to be thinner at the irises because of the reduced current density in those regions, leading to a lower plating rate. The purpose of the attachment plate on the beam tube was to mimic the one that would be required on the 915 MHz cavity to intercept the heat from the FPCs, as shown in Fig. [2.](#page-3-0)

#### A. Engineering analysis

A mechanical analysis of the cavity was done using Ansys to determine the cavity stiffness and tuning sensitivity as well as the stress distribution in the Nb resulting from the cool-down to 4 K, given the different coefficient of thermal expansion (CTE) between Cu and Nb. The stiffness analysis was done by fixing the axial position of the face of one flange, applying a 1 mm axial displacement to the face of the other flange and calculating the change in the cavity frequency and the force on the flange. The analysis was done for two cases, us-

![](_page_5_Figure_6.jpeg)

**Caption:** The cross-sectional layout of the 952.6 MHz prototype cavity is depicted here to demonstrate conduction-cooling strategies, emphasizing the importance of geometrical and compositional adjustments for optimized RF performance, consistency in cooling efficiency, and thermal management.


<span id="page-5-2"></span>FIG. 7. Cross-section layout of the 952.6 MHz prototype cavity for conduction-cooling demonstration.

<span id="page-5-0"></span>TABLE II. Electromagnetic parameters and mechanical properties at 295 K for the 952.6 MHz prototype cavity, with and without the Cu structure shown in Fig. [7.](#page-5-2)

| Electromagnetic parameters |             |                |  |
|----------------------------|-------------|----------------|--|
| Ep/Eacc                    |             | 1.90           |  |
| Bp/Eacc                    |             | 4.00 mT/(MV/m) |  |
| G                          |             | 176.3 Ω        |  |
| R/Q(β = 1)                 |             | 107.99 Ω       |  |
| Mechanical properties      |             |                |  |
|                            | Nb          | Nb/Cu          |  |
| Tuning sensitivity         | 1470 kHz/mm | 1313 kHz/mm    |  |
| Stiffness                  | 13411 N/mm  | 95089 N/mm     |  |

ing material properties at 295 K and 4 K. The case at room temperature was analyzed for the cavity with and without the Cu outer structure. The material properties used for Nb were: Young's modulus E = 105 GPa at 295 K, E = 118 GPa at 4 K and Poisson's ratio ν = 0.38 at both temperatures. The material properties used for Cu were: Young's modulus E = 127.2 GPa at 295 K, E = 151.8 GPa at 4 K and Poisson's ratio ν = 0.34 at both temperatures. The results from the analysis at 295 K are listed in Table [II.](#page-5-0) The linear stiffness of the Nb/Cu cavity increases to ∼ 109 kN/mm at 4 K and, assuming a typical tuning range of 200 kHz at 4 K, a cold tuner would need to apply a force up to ∼ 16.6 kN, which is well within the range of cold tuner designs already in use for SRF cavities [\[37](#page-22-6)[–40\]](#page-22-7).

To perform the analysis of the cool-down stress, a coefficient of linear thermal expansion α<sup>L</sup> = 5 ppm/K for Nb [\[41\]](#page-22-8) and α<sup>L</sup> = 11.4 ppm/K for Cu [\[29\]](#page-21-19) were used. The same E and ν values used for the stiffness analysis were assigned to Nb. Cu was modeled with a bi-linear stress versus strain curve based on the data from Ref. [[\[29\]](#page-21-19)] for OFHC and electroformed Cu: yield strength Y S(295 K)=52 MPa, Y S(28 K)=61 MPa , tangent modulus TM(295 K)=1.7 GPa, TM(28 K)=1.5 GPa for OFHC Cu; Y S(295 K)=105 MPa, Y S(28 K)=145 MPa, TM=3.3 GPa for electroformed Cu. E = 114 GPa and ν = 0.34 were assigned to both types of Cu, independent

of temperature. Unlike both Cu and Nb, the Nb3Sn softens considerably as the temperature decreases to 4 K. Temperature dependent E and ν values from Ref. [\[42\]](#page-22-9) were used in the model.

The results from the simulations indicate that the maximum stress in Nb remains well below the yield strength. Local yielding in the iris region is predicted in the Cu layer at 4 K. Figure [8](#page-6-1) shows a plot of the average and maximum equivalent stress in the Nb3Sn layer as a function of temperature as well as the equivalent strain at 4 K. The non-monotonic temperature dependence of the equivalent stress in Nb3Sn could result from the decrease of E by a factor of ∼ 3 between 300 K and 4 K. Tensile strain is predicted in the Nb3Sn film at the equator.

![](_page_6_Figure_2.jpeg)

**Caption:** Figure shows (a) the temperature dependence of the maximum and average equivalent von Mises stress in the Nb3Sn layer of a prototype cavity with an outer Cu structure. (b) Displayed is the equivalent strain at 4 K. This figure helps illustrate the mechanical stress distribution and strain characteristics in the cavity under operating conditions, highlighting the stress resilience of the Nb3Sn thin film and copper layer under thermal variations and contributing to its RF performance.


<span id="page-6-1"></span>FIG. 8. (a) Maximum and average equivalent (von Mises) stress as a function of temperature and (b) equivalent strain at 4 K in the Nb3Sn layer for the prototype cavity with the Cu structure shown in Fig. [7](#page-5-2) on the outer surface.

A thermal analysis of the cavity cooled by three RDE-418D4 CCRs was done with Ansys in order to verify achieving the target B<sup>p</sup> = 47 mT. The same material properties, cryocooler heat map, interface thermal conductance and Rs(Bp) used for the analysis of the 915 MHz cavity discussed in Sec. [II C](#page-2-1) were used for the

simulation of the 952.6 MHz prototype cavity. A static heat load of 1 W was applied uniformly to the cavity outer surface and 98% of the heat capacity was used for the CCR with upside-down orientation. The results showed that the cavity should be thermally stable at B<sup>p</sup> = 47 mT with up to 6 W of additional heat applied to the beam tube flange on the side with the CCR. Under these conditions, the cavity temperature would be ∼ 5.3 K, the rf heat load would be 3.5 W and the total heat load to the CCRs' stage 2 would be 10.25 W.

## <span id="page-6-0"></span>B. Nb3Sn coating

Following the rf test of the bare Nb cavity, the endflanges were disassembled and the cavity was degreased, HPRed and Nb plates were mounted on the cavity flanges inside the clean room. The bottom plate had a crucible where 99.999% pure Sn granules and 99.99% pure SnCl<sup>2</sup> powder were placed. Another crucible with Sn and SnCl<sup>2</sup> was placed inside a Nb shelf welded to the top cover plate. The cavity was mounted on a vertical stand outside the clean room, which is inserted in the vacuum furnace [\[43\]](#page-22-10). The vapor diffusion process consisted of nucleation at 500 ◦C for 1 h followed by coating at 1200 ◦C for 6 h. A visual inspection of the cavity inner surface after the thin-film coating showed large areas with shiny appearance at the equator, which have been associated with "patchy" regions with significantly thinner coating layer than non-shiny areas [\[44,](#page-22-11) [45\]](#page-22-12). After the coating process, the cavity was degreased, HPRed and assembled with the test hardware and evacuated on a vertical test stand. The standard instrumentation mounted to the cavity consisted of three single-axis cryogenic flux-gate magnetometers (FGMs) directed along the cavity axis and mounted at the equator, ∼ 120◦apart, and Cernox temperature sensors mounted at the iris and equator.

The cavity was cooled with LHe to 4.3 K inside a vertical cryostat at Jefferson Lab, minimizing the temperature gradient across the iris near the Nb3Sn transition temperature T<sup>c</sup> ∼ 18 K. The Q<sup>0</sup> at B<sup>p</sup> = 3.9 mT was 1.3 × 10<sup>10</sup> and it decreased exponentially with increasing B<sup>p</sup> to 2.4 × 10<sup>9</sup> at 50.8 mT, where MP occurred and could not be processed. The cavity was kept under vacuum on the vertical test stand for about two months and it was cooled down again to 4.3 K for a re-test to try processing the MP. However, the Q<sup>0</sup> at 3.9 mT had degraded to 2.6×10<sup>9</sup> , decreasing exponentially to 8.2×10<sup>8</sup> at 14.3 mT, in spite of a uniform cool-down across Tc. The frequency of the cavity was ∼ 3.2 MHz lower than after the first test. After warm-up to room temperature, the cavity shape was inspected with a 3D laser coordinate measuring machine, while still under vacuum on the vertical test stand. It was found that the cavity shape had become "re-entrant" as the irises were pushed inward by ∼ 0.7 mm on each side as a result of the 1 atm differential pressure and material creep due the softening of the Nb by the annealing at 1200 ◦C.

After disassembly, the cavity was mechanically tuned back to the initial frequency, then ∼ 20 µm were removed from the inner surface by EP, followed by degreasing, HPR and assembly with Nb end-plates. In order to achieve a more uniform coating, the required amount of Sn and SnCl<sup>2</sup> granules were distributed in three crucibles: one on the top plate, one at the bottom plate and one at at approximately mid-height, hanging from the top plate. Nb witness samples were placed on the top and bottom plates. The cavity was coated with the same temperature profiles as for the first coating. A visual inspection of the inner cavity surface showed a uniform, "matte" finish. Figure [9](#page-7-1) shows an image of the Nb3Sn thin film formed on the witness sample at the top plate, measured with a secondary electron microscope (SEM). The thin-film was uniform, with a Sn concentration of (24.75 ± 0.30) at.%, close to that of stoichiometric Nb3Sn, measured by energy dispersive x-ray spectroscopy. However, nanometersize Sn residues were found to be distributed over the entire surface. Similar results were observed on the witness sample placed on the bottom plate. The film thickness was measured from the witness samples using SEM images taken from the cross-section prepared with a focused ion beam and it was (2.18±0.48) µm and (2.87±0.29) µm for the 1st and 2nd coating, respectively.

The cavity was degreased, HPRed, assembled with the test hardware, mounted on a vertical test stand with restraining plates to prevent deformation due to differential pressure after evacuation, evacuated to ∼ 10<sup>−</sup><sup>8</sup> mbar and cooled in LHe at 4.3 K. The cavity was tested three times, each with different cool-down conditions in terms of residual dc magnetic field at the cavity and temperature gradient across the cell. Figure [10](#page-8-0) shows a summary of the cavity performance in LHe at 4.3 K after forming the Nb3Sn film on the inner surface. The iristo-iris axial temperature gradient, dT /dz, and the residual magnetic field, Ba, at T<sup>c</sup> for the test after the 2nd coating shown in Fig. [10](#page-8-0) were 0.64 K/m and 2.5 µT, respectively. The cavity reached up to 56.4 mT and was limited by strong MP at 52 mT. The x-ray dose rate at the maximum rf field was 21 µSv/h. The residual resistance, Rres, increased linearly with increasing axial temperature gradient and residual magnetic field at the rates dRres/d(dT /dz) = (9 ± 4) nΩ/(K/m) and dRres/dB<sup>a</sup> = (7 ± 5) nΩ/µT, respectively. These values are consistent with those mentioned in Ref. [\[46\]](#page-22-13) for a 1.3 GHz cavity.

The resonant frequency and Q<sup>0</sup> were measured as a function of the cavity temperature with a vector network analyzer (VNA) during warm-up from 4.3 K to obtain the temperature dependence of the change in rf penetration depth, ∆λ [\[47\]](#page-22-14), and of the surface resistance averaged over the inner surface of the cavity, R<sup>s</sup> = G/Q0. The data between 15 K and T<sup>c</sup> were fitted using the two-fluid model for ∆λ(T) and Halbritter's code for Rs(T). The London penetration depth, λ<sup>L</sup> = 80 nm and the BCS coherence length, ξ<sup>0</sup> = 6 nm were considered material constants [\[48\]](#page-22-15). The values of the parameters obtained from

![](_page_7_Figure_3.jpeg)

**Caption:** Figure shows scanning electron micrographs of a Nb/Cu interface after the cold-spray deposition of Cu. Image (a) presents the overall surface morphology at a lower magnification (x350), identifying the surface texture and uniformity. Image (b) displays a more detailed look (x3.5k magnification) at the grain structure of the Cu layer, demonstrating the microstructural evolution vital for understanding the mechanical and thermal properties imparted during the cold-spray process.


<span id="page-7-1"></span>FIG. 9. (a) low (x350) and (b) high magnification (x3.5k) SEM images of the Nb3Sn film on a Nb witness sample after the second coating.

the least-square fits were: T<sup>c</sup> = (17.79 ± 0.05) K, energy gap ∆/k<sup>B</sup> = (41.6 ± 0.2) K, normal electrons' mean free path ` = (3.78 ± 0.08) nm and rf penetration depth at 0 K λ<sup>0</sup> = (129 ± 2) nm. The uncertainty on the Tc-value includes both statistical and systematic uncertainties.

#### <span id="page-7-0"></span>C. Copper coating

Following the rf tests after the 2nd coating, the cavity was disassembled, degreased, dried and aluminum endcaps with an O-ring seal were mounted to the cavity flanges. The cavity was shipped to Concurrent Technologies Corporation (CTC), Johnstown, PA to deposit a thin layer of copper on the cavity outer surface by the cold-spray method, which provides good adhesion of Cu to Nb. 99.9% pure Cu powder sized at -325 mesh was fed into a de Laval nozzle at a rate of 150 standard liter per minute, along with He gas at 4.13 MPa and 400 ◦C to achieve a high impact velocity of the Cu powder onto the Nb. The cavity temperature only increased by a few

![](_page_8_Figure_0.jpeg)

**Caption:** The figure illustrates results from RF tests on a prototype Nb3Sn-coated cavity in liquid helium (4.3 K). It shows Q<sub>0</sub> as a function of the RF field (MP). Highlights of this test include no detectable x-ray emissions and less pressure sensitivity (df/dP) compared to a cavity without the Cu structure, shedding light on enhanced performance upon Cu application .


<span id="page-8-0"></span>FIG. 10. Results from the rf tests in LHe at 4.3 K of the 952.6 MHz prototype single-cell cavity after coating the inner surface with Nb3Sn. The inset shows a picture of the inner surface after forming the Nb3Sn film.

degrees Celsius above room temperature during deposition. The layer thickness was obtained from the difference between cavity dimensions before and after coating, measured with calipers, and it was between 178−330 µm. The deposition process took about ∼ 20 min.

Electroplating of the 1st and 2nd Cu layers was done at AJ Tuck Co., Brookfield, CT, using a proprietary recipe resulting in > 99.99% pure Cu. Machining of the cavity outer surface to the required dimensions and surface finish after each electroplating step was done at Jefferson Lab using both computerized numerical control lathe and milling machines. The areas with the bolt-holes were machined to a 0.4 µm average surface roughness. The beampipe flanges were kept sealed with end-caps throughout all of the steps required to produce the Cu structure on the cavity. Figure [11](#page-8-1) shows images of the cavity through different steps of the Cu deposition. Figure [12](#page-8-2) shows an SEM image of the cross-section of a Nb/Cu sample, in which the Cu was deposited in a similar way as for the cavity. Post-processing analysis of the SEM images, acquired over a sample length of ∼ 5.9 mm, showed a total interface length of ∼ 48 mm and the debonded length was only 2%.

After the final machining of the outer surface, the cavity was degreased, HPRed, assembled with the test hardware. An all-metal valve and a burst disk were added to the pump-out port on the stainless steel end-cap on one side of the cavity. The cavity was evacuated on a vertical test stand to ∼ 10<sup>−</sup><sup>8</sup> mbar and inserted in a vertical cryostat at Jefferson Lab. The cavity was cooled to 4.3 K at a rate of < 2 K/min in the range 30 −300 K. Two temperature cycles above T<sup>c</sup> were done to minimize the temper-

![](_page_8_Picture_5.jpeg)

**Caption:** This figure (a) shows the fabrication of a Cu structure on an outer cavity surface using a cold-spray technique, (b) depicts the fitting of the equator ring on the Cu layer after the first electroplating and machining step, and (c) captures the final machining of the assembly. These processes are integral to constructing a superconducting radio-frequency (SRF) cavity enhanced with a copper outer layer that provides improved mechanical stability and thermal conductivity, crucial for minimizing RF losses and ensuring effective cooling in SRF applications .


FIG. 11. Pictures taken during the fabrication of the Cu structure on the outer cavity surface: during cold-spray (a), during fitting of the equator ring on the Cu layer after the 1 st electroplating and machining (b) and during the final machining (c).

<span id="page-8-1"></span>![](_page_8_Figure_7.jpeg)

**Caption:** A Scanning Electron Micrograph (SEM) shows a layered cross-section of a Nb/Cu interface, differentiating between cold-sprayed Cu and electroplated Cu. This critical interface is central to understanding the mechanical and thermal advantages imparted by different fabrication methods on SRF cavity performance .


<span id="page-8-2"></span>FIG. 12. SEM image of the cross-section of a Nb/Cu sample with the interface Cu layer between Nb and electroplated Cu being deposited by cold-spray.

ature gradient across the cavity. The cavity performance in LHe at 4.3 K is shown in Fig. [13,](#page-9-1) after cool-down with dT /dz = 1.4 K/m and B<sup>a</sup> = 0.6 µT at Tc, limited by MP at B<sup>p</sup> = 53 mT. No x-rays were detected above background during the test. The pressure sensitivity was df /dP = −24 Hz/mbar, compared to −233 Hz/mbar measured for the cavity without the Cu outer structure. The ambient magnetic field at the cavity equator was measured during the rf test by three FGMs distributed at ∼ 120◦along the circumference. The magnetic field
measured by one sensor increased linearly with the power dissipated in the cavity at a rate of ∼ 85 µT/mW. The magnetic field measured by another sensor jumped by ∼ 0.6 µT during a MP-induced quench.

![](0__page_9_Figure_2.jpeg)

**Caption:** This figure displays results from an RF test on a 952.6 MHz single-cell cavity in liquid helium at 4.3 K after Cu coating on the outer surface. The cavity performance demonstrates improved thermal management, crucial for RF stability and efficiency, with Q<sub>0</sub> measurements validating this under the test conditions.


<span id="page-9-2"></span>FIG. 13. Results from the rf test in LHe at 4.3 K of the 952.6 MHz prototype single-cell cavity after coating the outer surface with Cu. The inset shows a picture of the fully assembled cavity.

The resonant frequency and Q<sup>0</sup> were measured with a VNA and rf amplifiers during warm-up between 15 K and 295 K. The change in rf penetration depth as a function of temperature below T<sup>c</sup> was fitted with the two-fluid model, resulting in λ<sup>0</sup> = (138 ± 1) nm, ` = (3.1±0.1) nm and T<sup>c</sup> = (17.73±0.08) K. The Q0(T) was measured at B<sup>p</sup> = 1.2 mT between 4.3 − 8.4 K with the same self-excited loop rf system used for the high-power test and Rs(T) = G/Q0(T) was fitted with Rs(T) = RBCS(T) + Rres, where RBCS was calculated numerically with Halbritter's code with fixed T<sup>c</sup> = 17.73 K, ` = 3.1 nm, λ<sup>L</sup> = 80 nm, ξ<sup>0</sup> = 6 nm. The fit parameters were ∆/k<sup>B</sup> = (39.0 ± 0.1) K and Rres = (12.7 ± 0.7) nΩ. Figure [14](#page-9-0) shows a plot of Rs(T)/Rs(14.5 K) between 14.5 − 19.5 K measured before and after growth of the Cu structure on the cavity outer surface showing a possible broadening of the transition by ∼ 0.1 K for the cavity after Cu deposition.

The frequency shift of the cavity under vacuum between 295 K and 4.3 K was ∼ 3.23 MHz, compared to ∼ 1.40 MHz before Cu deposition. This difference is due to the different CTE between Cu and Nb: Fig. [15](#page-10-0) shows the measured ∆f /f = [f(295 K) − f(T)] /f(295 K) for the Cu/Nb/Nb3Sn cavity, showing good agreement with ∆L/L = [L(T) − L(295 K)] /L(295 K) for Cu from Ref. [\[29\]](#page-21-0).

The surface resistance as a function of temperature was also measured in the normal state and showed a minimum at ∼ 50 K. To understand the mechanism of this minimum, consider the surface impedance Z = −iωE(0)µ0/B

![](0__page_9_Figure_7.jpeg)

**Caption:** This figure shows the impact of superconducting Nb3Sn thin film on surface resistance (R<sub>s</sub>) as temperature varies. This data indicates material behavior in temperature transition phases and is essential for evaluating potential performance enhancements through Cu outer layer deposition and RF conditioning.


<span id="page-9-0"></span>FIG. 14. Rs(T)/Rs(15 K) between 14.5 − 19.5 K measured before and after growing the Cu structure on the cavity outer surface.

of a thin film of thickness d and conductivity σ1(T) deposited on a substrate with conductivity σ2(T), exposed to a parallel rf field:

<span id="page-9-1"></span>
$$Z = (1 - i)\sqrt{\frac{\omega\mu\_0}{2\sigma\_1}} \frac{\beta + \tanh[(1 - i)q]}{\beta \tanh[(1 - i)q] + 1},\qquad(1)$$

where β = p σ2/σ<sup>1</sup> and q = d p µ0σ1ω/2 is the ratio of the film thickness divided by the skin depth in the film. Figure [16](#page-10-1) shows a least-square fit of the real part of Eq. [\(1\)](#page-9-1) to the experimental data using the temperaturedependent dc conductivities measured on a bulk Nb sample and a thin film Nb3Sn sample on sapphire. The parameters obtained from the curve fitting were: d = (5.1 ± 0.2) µm, the RRR of Nb3Sn, RRR<sup>1</sup> = (21 ± 63) and that of the Nb layer near the interface with the thin film, RRR<sup>2</sup> = (100 ± 56) and the room temperature resistivity of Nb3Sn, ρ1(295 K) = (129 ± 12) µΩ cm. As one can see, Eq. [\(1\)](#page-9-1) captures the observed behavior of Rs(T) and shows that the minimum in Rs(T) at ∼ 50 K in Fig. [16](#page-10-1) reflects a crossover from RN b3Sn s (T) at low T, for which the skin depth in Nb3Sn is smaller than d, to RN b s (T) at high T, for which the skin depth in Nb3Sn becomes much larger than the layer thickness and the contribution of Nb3Sn to E<sup>s</sup> is negligible.

#### IV. THERMAL LINK ASSEMBLY

An essential component of a conduction-cooled SRF cryomodule is the thermal link assembly (TLA) connecting stage 2 of a CCR to the cavity. Besides maximizing the thermal conductance between the two, the TLA needs to have sufficient flexibility to allow for misalignments between the cavity and stage 2 of the CCR, as well as providing compliance for thermal contraction during

![](0__page_10_Figure_0.jpeg)

**Caption:** This figure provides an analysis of frequency shift (Δf/f) and relative length change (ΔL/L) against temperature (T), crucial for assessing the thermal crest behavior of the SRF cavity materials. These data help in understanding intrinsic material responses to temperature fluctuations which impact RF performance and cavity stability.


<span id="page-10-0"></span>FIG. 15. Normalized shift in resonance frequency of the Cu/Nb/Nb3Sn cavity along with the normalized thermal expansion of Cu as a function of temperature.

![](0__page_10_Figure_2.jpeg)

**Caption:** Displayed is the normalized surface resistance, R<sub>s</sub>(T)/R<sub>s</sub>(15 K), measured within the temperature range of 14.5 K to 19.5 K, both before and after the Cu structure was deposited on the outer cavity surface. The data suggests a shift in the superconducting transition, connected to thermal and structural changes imparted by the Cu outer layer.


<span id="page-10-1"></span>FIG. 16. Rs(T) in the normal state measured for the Cu/Nb/Nb3Sn cavity. The solid line is a least-square fit with the real part of Eq. [\(1\)](#page-9-1).

cool-down. The specifications in terms of the required thermal conductance, hT LA, were determined from the thermal analysis described in Sec. [II C:](#page-2-0) 17.5 W/K at 4 K and 39.7 W/K at 10 K. The TLAs were manufactured by Absolut System, Seyssinet-Pariset, France, by stacking ∼ 85, 100 µm thick Cu foils, C10300 alloy, of ∼ 113 mm length and 152 mm width. The foil stack was bent into a "double S" shape and ∼ 25 mm long sections at the ends were press-welded in an inert gas atmosphere. A 74 mm wide central section was also press-welded and bolt-hole patterns were machined in the press-welded regions. Press-welding is a cost-effective method to fabricate TLAs with no internal thermal interfaces, which must be avoided in order to achieved the required thermal conductance. The TLAs were cleaned and baked at the vendor, as discussed in Ref. [\[49\]](#page-22-0). Figure [17](#page-10-2) shows a picture of the TLA. One TLA was mounted to stage 2 of a Sumitomo RDE418-D4 CCR through an adapter OFHC Cu cylinder and calibrated Cernox resistancetemperature devices (RTDs) were mounted to the following locations: one sensor on stage 2, one sensor on the cylinder adapter, two sensors on the wide presswelded area, one sensor each on the press-welded end sections. Two heaters were also mounted to the end sections. The setup was inserted in a vacuum vessel and cooled to ∼ 2.4 K with the CCR. The difference between the temperature near the heater and that on the cold side area of the TLA, ∆T, was measured as a function of the heater power, Ph, which was the same on each end of TLA. The TLA thermal conductance was calculated as hT LA = 2Ph/(∆T<sup>1</sup> + ∆T2), where ∆T<sup>1</sup> and ∆T<sup>2</sup> are the ∆T from each side of the TLA. Figure [18](#page-11-0) shows hT LA as a function of the average temperature. Each data point is the average of 3 measurement cycles. The measured thermal conductance was 23 W/K at 4 K and 77 W/K at 10 K, exceeding the specifications.

![](0__page_10_Picture_6.jpeg)

**Caption:** The figure shows a copper component used in superconducting radio-frequency (SRF) cavities, specifically spotlighting the prepared shape of copper used to enhance mechanical stability and improve thermal conductivity. This structural enhancement plays a crucial role in supporting the superconductive performance and overall integrity of the SRF cavity by reducing thermal gradients and RF losses during operation.


FIG. 17. Picture of the TLA used to connect the cavity to stage 2 of the CCR.

<span id="page-10-2"></span>The stiffness of the TLA was measured at room temperature by measuring the force as a function of displacement for three orthogonal directions. The force per unit length was found to be 18 N/mm in the direction normal to the plane with the circular bolt-hole pattern, 22 N/mm in the direction perpendicular to the plane with the linear bolt-hole pattern, and 14 N/mm in the direction along the plane with the linear bolt-hole pattern.

# V. HORIZONTAL TEST CRYOSTAT DESIGN AND ASSEMBLY

A horizontal test cryostat (HTC), which allows rf testing of the prototype cavity cooled by 3 CCRs, was designed and procured. It has a cylindrical grade 304 stainless steel vacuum vessel with two dome-shaped, O-ring sealed end caps. The overall dimensions are 150 cm in

![](0__page_11_Figure_0.jpeg)

**Caption:** This shows normalized frequency shift as a function of temperature for a Cu/Nb/Nb3Sn composite. The measurement data highlight how compositional modifications, such as adding a Cu layer, impact thermal expansion properties and subsequent RF performance through resonant frequency stability.


<span id="page-11-0"></span>FIG. 18. Thermal conductance of the Cu TLA shown in Fig. [17](#page-10-2) as a function of temperature. The inset shows a picture of the TLA mounted to stage 2 of the CCR at Jefferson Lab.

length and 1.1 m outer diameter. Two closely-spaced CCRs are mounted at the top, the third one being mounted upside-down at the bottom, using dedicated insertion fixtures. The vacuum flanges of one of the CCRs at the top and the one at the bottom were mounted to a bellows assembly, bolted to weldments on the vacuum vessel, allowing adjusting the position of their stage 2 inside the vessel. The cavity is held inside the vessel by 4 pairs of nitronic rods and it is thermally isolated from them by G10 washers. TLAs connect stage 2 of each CCR to the cavity. A286 stainless steel bolts with hardtempered silicon-bronze nuts were used to make the connections, along with stainless steel blocks on both sides of each joint to distribute the pressure uniformly over the contact area. The Cu contact surfaces were machined to an average roughness of 0.4 µm and a thin layer of Apiezon N grease was applied over the contact areas. The clamping pressure estimated for the torque on the bolts is ∼ 53 MPa.

An outer magnetic shield, a thermal shield and an inner magnetic shield are installed between the vacuum vessel shell and the cavity. The inner and outer magnetic shields were made by Ad-vance Magnetics, Inc., Rochester, IN, USA from 1 mm thick Cryoperm 10 and Ad-Mu-80, respectively. The thermal shield was made out of 2 mm thick OFHC copper, and it is attached to the stage 1 flanges of one of the top and the bottom CCRs. Both the vacuum vessel and the thermal shield were manufactured by Master Machine & Tool, Inc., Newport News, VA, USA. The parts for each of the shields consisted of 6 side panels and 4 end caps, all fastened together during the assembly, split between an upper and a lower sub-assembly. The design of the magnetic shields was validated by a magnetostatic analysis using Ansys, predicting an ambient magnetic field of less than 0.5 µT anywhere at the cavity surface. The biggest challenge with the mechanical design of the HTC was to ensure that the mechanical loads on both stages of the cryocoolers were below the manufacturer's specifications: 49 N, 294 N and 981 N for horizontal, tensile and compressive loads on stage 2, respectively and 98 N, 294 N, 1961 N for horizontal, tensile and compressive loads on stage 1. The top sub-assembly of the thermal shield is hung from the top sub-assembly of the outer magnetic shield by stainless steel wires. The bottom sub-assembly of the thermal shield is mounted to a Cu spool-piece, attached to stage 1 of the bottom cryocooler. The top and bottom thermal shield sub-assemblies are allowed to slide over each other during cool-down. A thermal and mechanical analysis of the HTC was done using Ansys. The TLAs were modeled as an orthotropic material and the material properties were adjusted to match the measured stiffness and thermal conductance. The nitronic rods's pretension was set to achieve null displacement at the cavity flanges and interfaces with the TLAs at 300 K. The temperature distributions from the thermal analysis were used to determine the stresses at the cryocoolers' heat stations. The results validated the design as all mechanical loads to the cryocoolers after cool-down were within the specifications. Figure [19](#page-11-1) shows a cross-section of the 3D model of the HTC.

![](0__page_11_Figure_6.jpeg)

**Caption:** Displayed is the thermal conductance (h<sub>T</sub>) of the Cu thermal link assembly across a range of temperatures. This evaluation is key for understanding the efficiency and capacity of thermal linking in superconducting settings, where consistent heat transfer ensures optimal thermal gradient minimization and improves cavity performance.


<span id="page-11-1"></span>FIG. 19. Cross-section view of the 3D layout of the HTC on a support cart.

The vacuum vessel has three side-ports: one for the rf and instrumentation panel, one for evacuation and one for a vacuum gauge and a burst disk. A flexible rf input cable (TFlex-405, Times Microwave Systems, Inc., Wallingford, CT, USA) and a semi-rigid rf pick-up cable (ULT-05, Keycom, Tokyo, Japan) are connected between the vacuum-side of an rf feedthrough with two Subminiature A (SMA) connectors on the instrumentation panel and the N-type rf feedthroughs mounted on the cavity. These cables were selected for their low thermal conductivity and both cables have a thermal intercept mounted to the thermal shield [\[50\]](#page-22-1). The instrumentation consisted of 16 Cernox CX-1050 RTDs and 3 cryogenic FGMs. The RTDs were distributed as follows: one on stage 2 of each CCR, one on each TLA, one each on the top and bottom thermal shield sub-assemblies and eight on the cavity assembly at the approximate locations shown in Fig. [20.](#page-12-0) A section of the leads of the RTDs attached to the cavity was wrapped around copper bobbins mounted to a Cu plate, bolted to stage 1 of CCR 1 to intercept the heat from the warm end of the wires. The 3 FGMs were attached with Kapton tape to the Cu equator ring at the approximate locations shown in Fig. [20,](#page-12-0) each aligned to three orthogonal components of the ambient field.

In addition, 7 low-inductance ceramic heaters (HTRs) were distributed as follows: one each on the top and bottom thermal shield sub-assemblies, one on the Cu bobbins mount plate and 4 on the cavity at the approximate locations shown in Fig. [20.](#page-12-0) The heaters wires consisted of Cu twisted pairs and a section of the wires of the heaters attached to the cavity assembly was also epoxied to Cu bobbins mounted to the same Cu plate which has the bobbins for the RTDs' leads. Figure [21](#page-12-1) shows a picture of the prototype cavity being assembled inside the HTC at General Atomics. The cavity and stage 2 of the CCRs were wrapped with 15 layers of multi-layered insulation (MLI). No MLI was wrapped around the thermal shield.

![](0__page_12_Figure_2.jpeg)

**Caption:** This figure depicts the SRF cavity configuration with various components labeled, focusing on the positioning of RTDs, heaters, and fundamental power couplers. This schematic significantly aids in understanding the thermal and magnetic management in cavity design, essential for experimental precision and optimization in superconducting settings.


<span id="page-12-0"></span>FIG. 20. Approximate locations of heaters (black rectangles), temperature sensors (red circles) and FGMs (blue cylinders) installed on the prototype cavity inside the HTC.

## VI. RESULTS FROM CONDUCTION-COOLED PROTOTYPE CAVITY

After the last test in LHe at Jefferson Lab, the cavity was shipped under static vacuum to General Atomics where a ∼ 50 m<sup>2</sup> laboratory space was setup to assemble the cavity into the HTC and to perform the cool-down, data acquisition and rf testing. Both the data acquisition programs, written in LabVIEW, and the low-level rf (LLRF) system were adapted from the ones used at Jefferson Lab for SRF cavity cool-down and testing [\[51\]](#page-22-2).

![](0__page_12_Picture_6.jpeg)

**Caption:** This figure shows the installation of the SRF cavity within a testing environment equipped with an extensive wiring setup. The copper structure of the cavity, circled, demonstrates the setup's complexity involved in monitoring and controlling the cavity's thermal and RF characteristics during experimental evaluations.


FIG. 21. Picture of the prototype cavity being assembled in the HTC at General Atomics.

<span id="page-12-1"></span>The room with the HTC was also equipped with 5 x-ray area monitors (Model 375, Ludlum Measurements, Inc., Sweetwater, TX, USA). After the assembly of the HTC was completed, it was evacuated to 1.6 × 10<sup>−</sup><sup>3</sup> Pa prior to cool-down.

#### A. Initial cooldown

The ambient magnetic field at the cavity equator prior to cool-down was Ba,x = 0.09 µT, Ba,y = −0.13 µT and Ba,z = −0.22 µT. Figure [22](#page-13-0) shows the average temperature of the equator ring, Tavg and the ambient magnetic field at the equator during the cool-down. The temperature measured by RTD4 (shown in Fig. [20\)](#page-12-0) and the RTD on stage 2 of CCR 1 was significantly higher than nearby sensors, likely due to improper mounting of those two sensors. Tavg reached ∼ 3.7 K after one day and it took three more days for Tavg to reach 2.76 K. The temperature of the cavity beam tubes was within 0.1 K of Tavg. Based on the CCRs capacity map, it is estimated that the total static heat load to stage 2 of the CCRs is ∼ 1 W, and that the static heat loads on stage 1 of CCR 2 and CCR 3 are ∼ 40 W and ∼ 65 W, respectively. A significant change in the ambient magnetic field at the cavity equator ring occurred during cool-down, likely due to thermoelectric currents for bimetallic surfaces, which can result in spontaneous magnetic flux upon cooling down below critical temperatures of 18 K for Nb3Sn and 9 K for Nb [\[52,](#page-22-3) [53\]](#page-22-4). During the cool-down, the magnitude of B<sup>a</sup> reached a maximum of ∼ 22 µT at Tavg = 29 K.

![](0__page_13_Figure_0.jpeg)

**Caption:** Average temperature and ambient magnetic field development during the initial cavity cool-down phase are displayed here. This figure is pivotal for understanding the thermal behavior and ambient magnetic field interactions during early cooling which significantly influence residual resistance and quality performance.


<span id="page-13-0"></span>FIG. 22. Average temperature and ambient magnetic field at the cavity equator as a function of time, starting at the beginning of the cool-down, during the first cool-down of the cavity in the HTC. The dashed line is at T<sup>c</sup> ∼ 17.7 K.

#### B. Rf test results

Figure [23](#page-14-0) shows a plot of Q<sup>0</sup> measured as a function of the rf field after three different cool-downs in the HTC. The cool-down conditions are summarized in Table [III](#page-15-0) and described in details in what follows. Each data point in Fig. [23](#page-14-0) is taken in steady-state conditions, achieved 2 − 4 min after each step increasing the rf field. Unlike when cooling with a LHe bath, the cavity temperature increases significantly with increasing rf power dissipation, Prf , as the CCRs' cooling power increases with temperature. No x-ray dose was detected by the area monitors above their sensitivity limit of 1 µSv/h. Additional Geiger-M¨uller tubes were placed close to the HTC during one of the rf tests and no radiation was detected.

After the rf test, a temperature cycle through T<sup>c</sup> was performed by turning off the CCRs until Tavg reached about 23 K. Then the CCRs were power-cycled on and off to slowly lower the cavity temperature to 17.5 K, below which the CCRs were kept on, cooling the cavity to a base temperature Tavg = 2.65 K. Figure [24](#page-14-1) shows Tavg and the ambient magnetic field at the cavity equator ring during this second cool-down. The slower cooldown across Tc, compared to the 1st cool-down, resulted in lower Ba. The results from the rf test after the 2nd cool-down are shown in Fig. [23.](#page-14-0) Lower B<sup>a</sup> resulted in a higher Q<sup>0</sup> at low rf field, compared to the 1st test, because of a lower Rres. The maximum B<sup>p</sup> was 50 mT, corresponding to Eacc = 12.4 MV/m. Increasing further the rf input power resulted in a sharp increase in Prf and both Q<sup>0</sup> and B<sup>p</sup> dropped as the cavity becomes undercoupled. Thermal instability was encountered above Prf ∼ 20 W. The thermal instability manifested as a sudden drop in both Eacc and Tavg and jumps in Ba, at which point the rf input power was quickly reduced to prevent further thermal runaway. There was no hysteresis in the

Q0(Bp) curve upon reducing the rf field. The maximum temperature difference around the cavity equator ring was < (70 ± 10) mK. The maximum difference between the temperature of the beam tube on the side with the CCR, TBT2, minus Tavg was < (−50 ± 30) mK and the maximum difference between the temperature measured on the beam tube on the side without CCR, TBT1, minus Tavg was < (170 ± 30) mK. To better determine the maximum stable rf heat load for long-term operation of the cavity, it was attempted to hold the cavity rf heat load for 1 h, starting at Prf ∼ 20 W but thermal instability occurred after 20 min. The test was repeated decreasing Prf in 0.5 W steps until it was successfully completed at Prf ∼ 18.5 W. All temperature readings were at steady state during those conditions, Tavg was between 6.45 − 6.50 K throughout the 1 h long test.

Additional Q0(Bp) curves were measured while applying Pdc = 2−6 W to HTR 6 and the results are shown in Fig. [25.](#page-14-2) Further rf tests were done while applying 2−6 W to HTR 4 and the results were similar to those shown in Fig. [25,](#page-14-2) for the same heater power. Figure [26](#page-14-3) shows a plot of the Tavg as a function of the sum of the cavity rf heat load plus any dc heat from HTR 6, Ptot, showing that adding dc heat to the Cu layer shifts Tavg at the lowest Prf upward and that the total maximum stable heat load remains constant.

Another temperature cycle through T<sup>c</sup> was performed by applying 11.5 W to heaters 5 and 7 and 31 W to heater 6 to reach Tavg ∼ 20 K. The three heaters are mounted on the cavity close to the TLAs as shown in Fig. [20,](#page-12-0) and the heaters' power was chosen to minimize the magnitude of Ba. The heaters' power was then reduced in small steps until Tavg was 16 K, below which the heaters were turned off. Tavg and B<sup>a</sup> during the 3rd cool-down across T<sup>c</sup> are shown in Fig. [27.](#page-15-1) The cavity was cooled to a base temperature Tavg = 2.65 K and the high-power rf test was repeated, the results being shown in Fig. [23.](#page-14-0) The ambient magnetic field during the 3rd cool-down was significantly higher than during the 2nd cool-down, as listed in Table [III,](#page-15-0) resulting in a higher Rres and lower Q0. It was verified that thermo-currents rather than the heaters themselves or their leads were responsible for producing the high ambient magnetic field during cool-down, since the change in B<sup>a</sup> was less than ∼ 0.1 µT when applying up to 20 W of power to each heater with the cavity at room temperature. Table [III](#page-15-0) gives a summary of B<sup>a</sup> at the cavity equator ring and the temperature gradient along the cavity axis between the beam tube on the side without CCR and the equator ring, dT /dy|1, and between the equator ring and the beam tube on the side with CCR, dT /dy|2, when Tavg ∼ 17.7 K, during the cool-down prior to the high-power rf tests with results shown in Fig. [23.](#page-14-0) Overall, the cool-down data show a correlation between high B<sup>a</sup> and a low Q0, because of a higher Rres. The cool-down data also suggest a complex distribution of the magnetic field and temperature gradients at the cavity, preventing simple quantitative estimates of the Rres from such data. For example, it is

known that the trapped flux sensitivity depends on the orientation of B<sup>a</sup> with respect to the cavity axis [\[54,](#page-22-5) [55\]](#page-22-6).

![](0__page_14_Figure_1.jpeg)

**Caption:** The figure depicts intrinsic quality factor Q<sub>0</sub> as a function of accelerating electric field E<sub>acc</sub> for a prototype conduction-cooled SRF cavity. The data spans various cool-down conditions, indicating cavity performance variance due to thermal stability. Insights from this analysis help optimize cooling strategies to maintain RF efficiency.


<span id="page-14-0"></span>FIG. 23. Q<sup>0</sup> as a function of the rf field for the conductioncooled prototype cavity measured at General Atomics for different cool-down conditions summarized in Table [III.](#page-15-0) All the data points were measured in steady-state conditions and the symbols' colors correspond to the average equator ring temperature. The cavity performance was limited by thermal instability beyond the lowest measured Q0-value. The inset shows a photo of the HTC.

![](0__page_14_Figure_3.jpeg)

**Caption:** This figure presents T<sub>avg</sub> and B<sub>a</sub> during the second cool-down, showing how the ambient magnetic field varied with temperature as the cryocoolers were cycled on and off. Insights from this figure are essential to understand the role of temperature dynamics and external magnetic fields in optimizing cavity cool-down processes .


<span id="page-14-1"></span>FIG. 24. Tavg and B<sup>a</sup> along three orthogonal directions at the cavity equator during the second cool-down for which the CCRs were power cycled on and off. The dashed line is at T<sup>c</sup> ∼ 17.7 K.

It was noticed during the rf tests that the magnitude of the ambient field at the cavity equator ring increased nearly linearly with increasing Prf as shown, as an example, in Fig. [28](#page-15-2) for the rf test after the 1st cool-down. For the same value of Prf , the magnitude of B<sup>a</sup> increased linearly with increasing Pdc also. Figure [28](#page-15-2) shows a correlation between the difference TBT<sup>1</sup> − Tavg and Ba.

The rf heat load and temperatures measured during

![](0__page_14_Figure_7.jpeg)

**Caption:** This figure plots Q0, the intrinsic quality factor, as a function of Rf field for the conduction-cooled prototype cavity. Measurements were taken after the second cool-down for varying levels of applied DC heat using HTR 6. This highlights how thermal instability limits cavity performance and demonstrates the impact of thermal and RF management strategies on cavity stability .


<span id="page-14-2"></span>FIG. 25. Q<sup>0</sup> as a function of the rf field for the conductioncooled prototype cavity measured at General Atomics after the second cool-down for different values of dc heat applied by HTR 6. All the data points were measured in steadystate conditions and the symbols' colors correspond to the average equator ring temperature. The cavity performance was limited by thermal instability beyond the lowest measured Q0-value, except for Pdc = 2 W for which the cavity test was stopped before reaching thermal instability.

![](0__page_14_Figure_9.jpeg)

**Caption:** Figure represents average temperature, T<sub>avg</sub>, as a function of total heat load, P<sub>tot</sub>, for distinct DC heat loads. Observations reveal the temperature response to changing RF loads, critical for assessing thermal limits and enhancing thermal management practices in superconductor applications.


<span id="page-14-3"></span>FIG. 26. Tavg as a function of the total heat load, Ptot = Prf + Pdc, for different values of Pdc. For each value of Pdc, Prf was increased up to the thermal stability limit, except for Pdc = 2 W.

the high-power rf test after the 2nd cool-down allowed calculating the total thermal conductance, h<sup>T</sup> , of the system comprising of the interface between stage 2 of the CCR and the TLA, the TLA and the interface between the TLA and the cavity, assuming that each of the 3 CCRs absorbs 1/3 of Prf , which is acceptable given the uniformity of the temperature in the Cu layer. Figure [29](#page-15-3) shows a plot of hT<sup>3</sup> = Prf /3(Tavg − TCCR3) and

![](0__page_15_Figure_1.jpeg)

**Caption:** This figure shows average temperature T<sub>avg</sub> and ambient magnetic field B<sub>a</sub> along three orthogonal directions at the cavity equator during the third cool-down period. The temperature control was dictated by power adjustments to heaters. This critical control data helps elucidate the effects of magnetic fields and temperature gradients on cavity stability and RF performance during cooldown processes .


<span id="page-15-1"></span>FIG. 27. Tavg and B<sup>a</sup> along three orthogonal directions at the cavity equator during the third cool-down for which the cavity temperature was controlled by adjusting the power to heaters 5-7. The heaters location is shown in Fig. [20.](#page-12-0) The dashed line is at T<sup>c</sup> ∼ 17.7 K.

hT<sup>2</sup> = Prf /3(TBT<sup>2</sup> − TCCR2) as a function of Tavg<sup>3</sup> = (Tavg +TCCR<sup>3</sup> +TT LA3)/3 and Tavg<sup>2</sup> = (TBT<sup>2</sup> +TCCR<sup>2</sup> + TT LA2)/3, respectively, where TCCR is the temperature of stage 2 of the CCR and TT LA is the temperature of the TLA.

![](0__page_15_Figure_4.jpeg)

**Caption:** This figure displays variations of the ambient magnetic field, Ba, at different equator ring locations along three orthogonal directions versus the temperature difference, TBT1 − Tavg, as a function of RF power, Prf. It sheds light on the field distribution during cooling cycles, particularly during the high-power RF test after the first cool-down, and underscores the challenge of managing magnetic fields in ensuring efficient cooling and RF performance in SRF cavities .


<span id="page-15-2"></span>FIG. 28. B<sup>a</sup> at different locations of the cavity equator ring, along three orthogonal directions, and temperature difference TBT <sup>1</sup> − Tavg as a function of Prf , measured during the highpower rf test after the 1st cool-down.

#### C. Microphonics

The digital LLRF system used to measure the cavity Q0(Bp) provides analog outputs of the in-phase, I(t), and quadrature, Q(t), amplitudes of the transmitted voltage

![](0__page_15_Figure_8.jpeg)

**Caption:** This figure charts total thermal conductance as a function of temperature between the cavity and Cryocooler CCRs 2 and 3, displaying data from high-power RF tests post-second cool-down. It illustrates the thermal optimisation required for effective RF heat management in advanced RF systems to ensure stable operation under varying thermal conditions .


<span id="page-15-3"></span>FIG. 29. Total thermal conductance between the cavity and CCRs 2 and 3 as a function of temperature obtained from the data measured during the high-power rf test after the 2nd cool-down.

<span id="page-15-0"></span>TABLE III. Ambient magnetic field at the cavity equator ring and temperature gradient at either side of the equator ring when Tavg = 17.7 − 17.8 K for each of the three cool-downs. The x, y and z directions are shown in Fig. [20.](#page-12-0)

| Cool<br>down  | dT /dy 1<br>(K/m) | dT /dy 2<br>(K/m) | Ba,x<br>(µT) | Ba,y<br>(µT) | Ba,z<br>(µT) |
|---------------|-------------------|-------------------|--------------|--------------|--------------|
| No.<br>1<br>2 | -5.1<br>-5.1      | 8.1<br>6.3        | 6.2<br>4.3   | -15<br>-1.7  | -4.6<br>-0.2 |
| 3             | 3.0               | -3.9              | 5.4          | 1.6          | -12.3        |

from the cavity. Because we are using a frequency tracking rf system, the change in frequency relative to the center frequency of the cavity resonance, δf(t), can be calculated as [\[56\]](#page-22-7):

$$
\delta f(t) = \frac{Q(t)\frac{dI}{dt} - I(t)\frac{dQ}{dt}}{2\pi\left(I^2 + Q^2\right)}.\tag{2}
$$

The I(t) and Q(t) signals are digitized by a sound and vibration device (model USB-4431, National Instruments, Austin, TX, USA) connected to a computer, acquiring data through a LabVIEW program. Multiple 3 min long data sets were acquired at B<sup>p</sup> = 14 mT with a rate of 20 kS/s. The data was filtered through a 1 kHz low-pass filter. Figure [30](#page-16-0) shows a snapshot of δf(t) acquired with the 3 CCR and the turbo-pump cart pumping on the vacuum vessel being on. The average peak excursion of |δf(t)| from 3 sets of 3 min-long data was δfpk = (23.0 ± 0.9) Hz. Figure [31](#page-16-1) shows a fast-Fourier transform (FFT) of δf(t). The frequencies of the three highest peaks are 42 Hz, 360 Hz and 1.2 Hz. Additional

data were taken after turning off CCR2 and both CCR2 and CCR3, in which case δfpk decreased to (20.8±0.4) Hz and (18.7 ± 1.3) Hz, respectively. No significant changes were observed when turning off the turbo-pump cart.

![](0__page_16_Figure_1.jpeg)

**Caption:** The graph displays the time-dependent variations in cavity frequency during RF operations, captured over 10 seconds, indicating microphonic effects present in the cavity system. Understanding these shifts assists in reducing detuning instability and optimizing noise reduction mechanisms .


<span id="page-16-0"></span>FIG. 30. Snapshot of the shift in the resonant frequency of the prototype cavity inside the HTC, due to microphonics.

![](0__page_16_Figure_3.jpeg)

**Caption:** Figure illustrates changes in surface resistance normalized to a reference temperature (15 K) across a temperature range of 14.5 K to 19.5 K under differing conditions before and after Cu structure deposition. Such analysis helps identify shifts in superconducting transition and material properties crucial for SRF cavity performance.


<span id="page-16-1"></span>FIG. 31. Magnitude of the FFT of δf(t) shown in Fig. [30.](#page-16-0) The inset shows an expanded view of the peaks below 50 Hz.

A tri-axial accelerometer (model 356B18, PCB Piezotronics, Inc., Depew, NY, USA) was used to measure the acceleration due to mechanical vibrations at different locations on the HTC. The data acquisition system consisted of the same setup used to acquire the transmitted voltage signals from the LLRF. 3 min-long data sets with a rate of 2 kS/s were measured at each location. Figure [32](#page-16-2) shows the magnitude of the FFT of the acceleration in the vertical direction at two representative locations: one on the HTC support cart and one on top of CCR1. The peak acceleration amplitude measured on the support cart was 0.06g, where g is the acceleration of gravity and the spectrum matches that of the acceleration measured on the floor. The top three frequencies with the highest magnitude in Fig. [32\(](#page-16-2)a) are at 361.1 Hz, 118.8 Hz and 39.6 Hz. The peak acceleration amplitude measured on top of CCR1 was 1.7g and the top three frequencies with the highest magnitude in Fig. [32\(](#page-16-2)b) are at 360 Hz, 1.2 Hz and 2.4 Hz.

The peak at 1.2 Hz in the spectra shown in Figs. [31](#page-16-1) and [32](#page-16-2) corresponds to the frequency of the displacer moving inside the CCRs' cold stage. The peak at ∼ 40 Hz and its higher harmonics were caused by the CCRs' He compressors, which were located in the same room as the HTC, ∼ 2 m away.

![](0__page_16_Figure_9.jpeg)

**Caption:** This figure illustrates the FFT magnitude of vertical accelerations measured (a) on the HTC support cart and (b) on top of CCR1, with an accelerometer highlighted in the red circle in inset photos. This data is crucial for understanding vibrational impacts, or microphonics, affecting the SRF cavity in operation, which can induce frequency shifts and affect the cavity's RF stability .


<span id="page-16-2"></span>FIG. 32. Magnitude of the FFT of the vertical acceleration measured on the HTC support cart (a) and on top of CCR1. The accelerometer is circled in red in the inset photos.

### VII. DATA ANALYSIS AND DISCUSSION

#### A. Surface resistance field-dependence

The data show a stronger increase of the surface resistance with increasing B<sup>p</sup> after the Cu layer was deposited onto the outer cavity surface. The data have been analyzed with a one-dimensional thermal feedback model [\[57,](#page-22-8) [58\]](#page-22-9) for three cases: (1) Nb/Nb3Sn cooled by LHe, (2) Cu/Nb/Nb3Sn cooled by LHe and (3) Cu/Nb/Nb3Sn cooled by CCRs. The following equations were solved numerically to find the temperature of the inner (Tm) and outer (Ts) surfaces of the cavity as a function of B<sup>p</sup> for case 1:

<span id="page-17-0"></span>
$$\frac{1}{2}R\_s(T\_m, B\_p) \left(\frac{B\_p}{\mu\_0}\right)^2 = h\_{cav}(T\_m - T\_s) \tag{3}$$

<span id="page-17-2"></span>
$$h\_{cav}(T\_m - T\_s) = h\_1(T\_s - T\_0)^{2.5},\tag{4}$$

where T<sup>0</sup> = 4.3 K is the LHe bath temperature and h<sup>1</sup> = 1 kW/(m<sup>2</sup> K2.<sup>5</sup> ) is the nucleate boiling heat transfer coefficient in He I [\[59\]](#page-22-10). <sup>1</sup> hcav = dNb κNb + dNb3Sn κNb3Sn is the thermal resistance through the cavity wall, dN b = 3.8 mm is the thickness of the Nb substrate, dN b3Sn = 3 µm is the thickness of the Nb3Sn film, κN b = 88 W/(m K) is the thermal conductivity of Nb at 4.3 K and κN b3Sn = 49 mW/(m K) is the thermal conductivity of Nb3Sn at 4.3 K [\[30\]](#page-21-1). For case 2 the term dCu/κCu is added to the cavity thermal resistance, where dCu = 7 mm is the thickness of the Cu layer and κCu = 2365 W/(m K) is the thermal conductivity of the electroplated Cu at 4.3 K. For case 3 the following equation is solved numerically:

<span id="page-17-1"></span>
$$\frac{1}{2}R\_s(T\_m, B\_p)\left(\frac{B\_p}{\mu\_0}\right)^2 = h\_{cav}(T\_{avg})[T\_m - T\_{avg}(B\_p)],\tag{5}$$

reflecting the fact that the temperature of the outer surface is no longer a constant. The thermal conductivities of Cu, Nb and Nb3Sn at Tavg are used to calculate hcav(Tavg). The following phenomenological field dependence of Rs, based on a series expansion of Rres(B<sup>2</sup> p ) up to the second order, was used in Eqs. [\(3-](#page-17-0)[5\)](#page-17-1):

$$\begin{aligned} R\_s(T\_m, B\_p) &= R\_{BCS}(T\_m) + \\ R\_{res} \left[ 1 + \gamma \left( \frac{B\_p}{B\_c} \right)^2 + \delta \left( \frac{B\_p}{B\_c} \right)^4 \right], \quad (6) \end{aligned}$$

where B<sup>c</sup> = 500 mT is the thermodynamic critical field of Nb3Sn at 0 K [\[60\]](#page-22-11). The following simplified expression, valid for T < Tc/2, was considered for RBCS(Tm) [\[58\]](#page-22-9):

<span id="page-17-4"></span>
$$R\_{BCS}(T\_m) = \frac{A}{T\_m} e^{-\Delta/k\_B T\_m},\tag{7}$$

with A = 1.7 × 10−<sup>4</sup> ΩK and ∆/k<sup>B</sup> = 41.6 K for case 1, ∆/k<sup>B</sup> = 39 K for cases 2 and 3. A least-squares fit of the data for the 2nd coating shown in Fig. [10](#page-8-0) for case 1 and of the data shown in Fig. [13](#page-9-2) for case 2 with Eqs. [\(3\)](#page-17-0) and [\(4\)](#page-17-2) was done to determine the value of the fit parameters Rres, γ and δ. For case 1 we obtained Rres = 20.5 nΩ, γ = 24, δ = 6.6 × 10<sup>3</sup> , whereas Rres = 8.9 nΩ, γ = 3×10−<sup>6</sup> , δ = 4.8×10<sup>4</sup> for case 2. For case 3, the data set Rs(Bp) and Tavg(Bp) from Fig. [23](#page-14-0) for the 2nd cool-down was used for a least-squares fit with Eq. [\(5\)](#page-17-1). Since the cavity was kept under vacuum between the rf tests considered for cases 2 and 3, the values of γ and δ were kept constant and Rres = 9.6 nΩ was the only fit parameter for case 3. Figure [33](#page-18-0) shows Rs(Bp) for the three cases considered, along with the results from the least-squares fits, which are in good agreement with the data. The higher Rres for case 1 compared to cases 2 and 3 could be due to the higher B<sup>a</sup> along the cavity axis for case 1.

Besides the field dependence of R<sup>s</sup> described by Eq. [\(6\)](#page-17-3), another possibility was considered multiplying <sup>R</sup>BCS(Tm) by a factor 1 + θ B<sup>p</sup> B<sup>c</sup> 2 , where the B<sup>2</sup> p term represents pair-breaking by the rf current, and neglecting the second-order term in the series of Rres(B<sup>2</sup> p ). However, the quality of the fits was poorer and the field dependence of R<sup>s</sup> could not be reproduced keeping the same values of θ and γ between cases 2 and 3. Adding a fixed thermal boundary resistance term to the cavity thermal resistance, because of the Nb/Cu interface, as an additional fit parameter for cases 2 and 3 did not improve significantly the quality of the fit of the case 3 data. This suggests that the main source of the field dependence of R<sup>s</sup> at B<sup>p</sup> B<sup>c</sup> comes from the residual resistance. The field dependence of Rres at B<sup>p</sup> B<sup>c</sup> may result from weakly-coupled grain boundaries in Nb3Sn [\[61](#page-23-0)[–64\]](#page-23-1) or trapped vortices [\[65–](#page-23-2)[67\]](#page-23-3).

The analysis of Rs(Bp) data with Eqs. [\(3-](#page-17-0)[7\)](#page-17-4) suggests that the quality of the Nb3Sn film degraded after depositing the thick Cu outer layer, given the increase of δ by a factor of ∼ 7 between cases 1 and 2. Similar results were measured in a 1.5 GHz Cu/Nb/Nb3Sn single-cell cavity [\[5\]](#page-21-2). The quadratic term provides a much smaller contribution to the observed increase of R<sup>s</sup> with increasing rf field.

<span id="page-17-3"></span>The superconducting properties of Nb3Sn are very sensitive to strain [\[68](#page-23-4)[–70\]](#page-23-5) and additional stress in the film occurs after depositing the thick Cu outer layer, because of the differential thermal contraction between Cu and the Nb substrate. Since the residual stresses in the film after formation on the Nb substrate are unknown, it is difficult to estimate the absolute strain value in the film after deposition of the Cu layer and cool-down to 4 K. If one neglects any initial residual stress, the maximum strain in the film from Fig. [8\(](#page-6-0)b) can be estimated to be of the order of 0.2%. Such strain value would result in a ∼ 1% decrease of Tc, consistent with the results from the cavity rf measurements at low field, and ∼ 25% decrease
![](1__page_18_Figure_0.jpeg)

**Caption:** The figure presents surface resistance R<sub>s</sub> as a function of peak magnetic field B<sub>p</sub> for Nb/Nb3Sn and Cu/Nb/Nb3Sn cavities cooled with liquid helium (LHe) and conduction-cooled using cryocoolers (CCRs). The graph includes data fits comparing measurements with empirical models to determine the surface resistance contributions from Cu layers during RF operations. This evaluation highlights the increased surface resistance with rising B<sub>p</sub>, pointing out the thermal feedback effects introduced by the Cu outer layer.


FIG. 33. Rs(Bp) data for the Nb prototype cavity cavity with Nb3Sn film on the inner surface cooled by LHe and with Cu outer layer tested both in LHe and with cryocoolers. Solid lines are least-squares fit to each data set.

of Jc. Compositional and morphological changes across grain boundaries in Nb3Sn makes them prime suspects for producing residual rf losses [\[61,](#page-23-0) [71,](#page-23-1) [72\]](#page-23-2). The additional strain from the cool-down stresses could result in weaker coupling among grain boundaries, producing an upturn of the field dependent residual rf losses.

# B. Cu outer layer

The thermal analysis of the prototype cavity discussed in Sec. [III A](#page-5-0) was repeated using Ansys in order to validate the simulation with the test results. The main difference compared to the initial analysis was the use of the overall thermal conductance values from Fig. [29,](#page-15-0) h<sup>T</sup> = 4.3 W/K at 4 K and h<sup>T</sup> = 11.1 W/K at 10 K, instead of h<sup>T</sup> = 16.8 W/K at 4 K and h<sup>T</sup> = 36.5 W/K at 10 K calculated from the specified TLA thermal conductance and the interface heat transfer coefficient assumed for the initial analysis. In addition, the stage 2 temperature versus cooling power curve shown in Fig. [3,](#page-3-0) which is ∼ 10% better than the nominal data from the vendor, was used in the revised analysis. The process to determine the highest rf heat load at which the finite-element analysis converged also had to be modified as follows: an iterative solving process was applied to all of the CCRs' stage 2 and a very small damping value of ∼ 0.05, multiplying the difference in the stage 2 temperatures from the solution of one iteration step to the next, needed to be used when determining the stage 2 temperatures set for the subsequent iteration step. Figure [34](#page-18-0) shows the cavity temperature distribution at B<sup>p</sup> = 47 mT at the highest rf heat load of 16.9 W, which resulted in a converging solution with Ansys. A multiplying factor of 5.2

was applied to the field-dependent part of the empirical Rs(Bp, T) used in the initial analysis, in order to match the measured rf heat load at 47 mT. This factor reflects the steeper increase of Rs(Bp) than that from the empirical formula, which is based on data prior to the Cu coating. The maximum stable cavity temperature and rf heat load calculated with Ansys are consistent with the experimental results, given the uncertainty in the measurement of the overall thermal conductance. The maximum stable cavity temperature Tmax ∼ 6.7 K from the Ansys simulation is consistent with the temperature for which dPrf /dT exceeds dh<sup>T</sup> /dT, meaning that above Tmax the rf heat can no longer be conducted effectively to the CCRs.

![](1__page_18_Figure_6.jpeg)

**Caption:** The temperature distribution for a 952.6 MHz prototype cavity cooled by three CCRs is shown. The Ansys simulation implemented at B<sub>p</sub> = 47 mT and RF load of 16.9 W illustrates optimal cooling capabilities of conduction systems at these settings, with color-coded temperature gradients providing essential thermal management insights.


<span id="page-18-0"></span>FIG. 34. Temperature distribution for the 952.6 MHz prototype cavity cooled by 3 CCRs, calculated with Ansys for B<sup>p</sup> = 47 mT at the maximum Prf = 16.9 W which resulted in a stable solution. The temperature color map scale is in units of Kelvin.

In order to better understand the role of the thick Cu outer layer in the maximum stable rf heat load that can be achieved, the Ansys thermal analysis was repeated with the following modifications: (i) the Cu layer was removed, (ii) the Cu equator ring and beam-tube attachment were replaced by 4 mm thick high-RRR Nb equator ring and beam-tube attachment and (iii) the same empirical Rs(Bp, T) used for the initial analyis (Fig. [4\)](#page-4-0) was used in the simulation. The Nb equator ring and beamtube attachment could be electron-beam welded to the 4 mm thick Nb cavity prior to Nb3Sn coating. Using the same procedure mentioned above, the simulation resulted in a maximum stable cavity temperature Tmax ∼ 5.5 K at B<sup>p</sup> = 44 mT and Prf = 4.8 W.

The results of the Ansys analysis show that the presence of a thick Cu outer layer significantly enhances the maximum stable rf heat load that could be achieved in a conduction cooled SRF cavity. However, the increase of the Nb3Sn surface resistance with increasing B<sup>p</sup> after the deposition of the Cu layer did not allow to fully exploit the higher thermal stability towards achieving higher accelerating gradients. The presence of an outer layer with high thermal conductivity could prove to be particularly beneficial when considering the additional heat loads from the FPCs and from the beam pipes extending between the cavity flanges and the outside of a cryomodule to be used in an accelerator. Another benefit of the Cu outer layer is the increase in the mechanical stability of the cavity, making it less sensitive to microphonics and less susceptive to cracking of the Nb3Sn film. The occurrence of cracks in the film because of high stresses due to, for example, differential pressure, transportation or handling, would severely degrade R<sup>s</sup> [\[73,](#page-23-3) [74\]](#page-23-4).

Whereas the presence of the Cu outer layer helps achieving a uniform cool-down of the cavity across T<sup>c</sup> when cooled with LHe, higher temperature gradients occur when cooling with CCRs. The presence of thermoelectric currents between the Cu layer and the Nb, both during the cool-down and during the high-power rf test, is manifested by changes in the local magnetic field at the cavity surface. This phenomenon makes controlling the uniformity of the cavity temperature and therefore the local residual field by external heaters somewhat more challenging than in absence of the Cu layer. Modeling the thermoelectric currents and residual magnetic field for the prototype cavity in the HTC configuration will be part of future work.

A drawback of having the Cu outer layer on the Nb/Nb3Sn cavity is that if any kind of damage to the Nb3Sn thin film was to occur, a substantial rework of the cavity to remove the Cu layer would be necessary, given the high temperature required to form a new Nb3Sn layer. The Cu ring and attachment plate could be cut by wire electro-discharge machining and the remainder of the Cu could be etched away by immersing the cavity in nitric acid. The Cu shell would need to be reformed on the cavity outer surface after re-coating the inner surface with Nb3Sn. Given the large margin in terms of the maximum rf dissipation that was demonstrated with our setup, it should be possible to limit the amount of Cu to perhaps only the equator region and the beam-tube region where a high-power FPC would be connected to the cavity. Further engineering analysis to optimize the amount of Cu needed should allow finding a good compromise between handling the required heat loads and mitigating the drawbacks of the Cu layer mentioned above.

# C. Type of cryocooler

Pulse-tube (PT) cryocoolers with the same nominal cooling capacity as that of the GM-type we have used for this project are also being considered for conduction cooling of SRF cavities. The main advantage of the PT- type CCR, compared to the GM-type, is the absence of a mechanical displacer, therefore reducing the amplitude of mechanical vibrations of stage 2 [\[75\]](#page-23-5). However, the MW-class beam-power envisioned for conduction-cooled SRF accelerators sets the loaded-Q of the cavity to the order of ∼ 1 × 10<sup>5</sup> , corresponding to a bandwidth of ∼ 9 kHz. A frequency shift of less than ∼ 1/10 of the cavity bandwidth due to microphonics would not require any additional overhead in the high-power rf source. The δfpk we have measured on a cavity with 3 GM-type CCRs is similar to that measured on cavities cooled by superfluid He inside cryomodules and with loaded-Q values of the order of ∼ 1 × 10<sup>7</sup> [\[76\]](#page-23-6).

A major drawback of PT CCRs is that the wall-plug efficiency is significantly lower than that of GM-type, for the same cooling capacity. For example, the PT CCR model PT420 (Cryomech, Syracuse, NY, USA) requires a wall-plug power of 12.5 kW at 60 Hz [\[77\]](#page-23-7), compared to 7.6 kW at 60 Hz for the GM-type model RDE-418D4 [\[78\]](#page-23-8), both providing a cooling power of 2 W at 4.2 K. In addition, the cost of a PT CCR is significantly higher than that of a GM CCR with the same nominal cooling capacity at 4.2 K.

## D. Thermal link and interface materials

High-purity (99.999%) Al is an ideal material for a TLA because its thermal conductivity is higher than that of OFHC Cu at 4 K [\[79\]](#page-23-9). This material is being used to fabricate thermal links for conduction-cooled SRF cavity application at Fermilab [\[4,](#page-21-0) [80,](#page-23-10) [81\]](#page-23-11). We used TLAs made of oxygen-free Cu due to its lower cost, greater availability and to avoid a contact potential between the TLA and the Cu cavity and stage 2, which could produce thermo-currents. In addition, press-welding of Cu foils is a low-cost solution to fabricate a flexible TLA with no thermal resistance between the foils. A more complex and expensive method, such as electron-beam welding, would be needed to achieve a similar outcome if highpurity Al foils were to be used [\[49\]](#page-22-0). Finally, given the proximity between the CCR stage 2 and the cavity, the thermal conductivity of the TLA material plays less of a role compared to the contact thermal resistance of the connections between the cavity and stage 2 of the CCR. This is evident by comparing Figs. [18](#page-11-0) and [17.](#page-10-0)

Regarding the type of material to be used as part of the thermal interface between stage 2 of the CCR, the TLA and the cavity, both thin In foils and Apiezon N grease have been considered. No definitive answer was found in the literature about which of the two provides the highest thermal conductance across the surface of different metals at the temperatures of interest [\[31,](#page-22-1) [82–](#page-23-12) [85\]](#page-23-13). However, the disassembly of a joint in which an In foil is used between the mating surfaces is more complex than that in which grease is used. Using In as part of a joint often requires the use of custom tools to break the joint apart, to remove the In from the surfaces and, in the case of In being removed from Nb, soaking the Nb in nitric acid to remove staining. The chances of leaving scratches onto the surface during In removal is also higher, which would require additional mechanical polishing and/or machining.

## VIII. SUMMARY AND OUTLOOK

We have scaled the design of a 1 MW, 1 MeV conduction-cooled SRF linac to 915 MHz to take advantage of the availability of low-cost and efficient highpower, commercial rf magnetrons. The key component to be developed for such an accelerator is a single-cell SRF cavity cooled by conduction using CCRs and operating at B<sup>p</sup> = 41.5 mT. We have re-purposed a 952.6 MHz single-cell Nb cavity to be the prototype to develop the technologies needed to demonstrate the operation of the cavity, cooled by CCRs, at these fields. The cavity was coated with a Nb3Sn film on the inner surface. A method was developed to produce a thick, high-purity Cu layer with three CCR attachment locations onto the cavity outer surface. The cavity performance was measured at Jefferson Lab in a vertical cryostat filled with LHe before and after the Nb3Sn coating and after the formation of the Cu outer layer. The performance of the Nb3Sncoated cavity was limited by MP at B<sup>p</sup> ∼ 53 mT.

A horizontal test cryostat was designed and procured to be able to measure the cavity performance when cooled by three GM-type CCRs. As part of the cryostat development, a thermal link made of high-purity Cu was successfully designed, procured and tested, meeting the requirements in terms of both mechanical flexibility and thermal conductance. The cavity was assembled inside the HTC at General Atomics and its performance was measured for different cool-down conditions and for different values of additional heat applied by local heaters mounted onto the cavity. The cavity achieved B<sup>p</sup> = 50 mT, which is the highest peak surface magnetic field reached in a conduction-cooled cavity to date and exceeding the performance requirement with a ∼ 20% margin. The cavity was operated stably, in a steady state, for 1 h at a maximum rf heat load of 18.5 W. These results validated all the technical design choices that were made and represent an important stepping-stone towards the development of conduction-cooled SRF cavities for industrial linac applications.

Whereas the addition of the Cu outer layer signifi-

cantly extends the thermal limit and mechanical stability of the cavity, a stronger increase of Rs(Bp) was found after depositing such layer. This could be due to strain in the film resulting from the differential CTE between Cu and Nb. A better understanding of the effect of strain on the field-dependent surface resistance of Nb3Sn, along with the search for alternative materials with high thermal conductivity at 4 K and a CTE close to that of Nb, as well as further engineering design of the cavity outer layer would all be important paths for future R&D. Given the ongoing efforts towards forming high-performance Nb3Sn films directly onto a Cu substrate [\[86–](#page-23-14)[89\]](#page-23-15), the approach described in this article could be a proven solution to utilize Cu/Nb3Sn cavities in a conduction-cooled SRF cryomodule for future accelerators.

Future work aims at the design of a 10 MeV, 1 MW electron linac for environmental remediation using a multi-cell, conduction-cooled SRF cavity and to validate some of the design aspects on a multi-cell prototype cavity to be tested in the HTC at General Atomics.

## ACKNOWLEDGMENTS

The authors would like to acknowledge B. Golesich at CTC for the Cu cold-spray, A. Tuck at AJ Tuck Co. for the Cu electroplating and D. Combs of JLab's machine shop for the machining of the Cu outer layer. We would like to thank our colleagues from the SRF Cavity Production Group at Jefferson Lab for helping with the cavity cleaning, assembly and cool-down in the vertical cryostat. We would like to thank A. Cuffe, R. Bachimanchi, C. Wilson and P. Owen of Jefferson Lab for help preparing the LLRF modules for off-site use and software development. We would also like to thank M. Dale of Sumitomo Cryogenics of America for many useful discussions. Finally, we would like to acknowledge C. Bott and M. Pearce of Hampton Roads Sanitation District for useful discussions on the potential use of SRF accelerators for wastewater treatment. This work is authored by Jefferson Science Associates, LLC and it was supported by the U.S. Department of Energy, Office of Science, Office of Accelerator Research & Development and Production, under contract No. DE-AC05-06OR23177. SB's microscopy work at the National High Magnetic Field Laboratory was partly supported by the U.S. Department of Energy, Office of Science, Office of High Energy Physics under Award No. DE-SC0009960.

[Science and Technology](https://doi.org/10.1142/S1793626812300071), Vol. 5, edited by A. W. Chao and W. Chou (WORLD SCIENTIFIC, Singapore, 2013) pp. 185–203.

<sup>[1]</sup> S. Belomestnykh, Superconducting radio-frequency systems for high-β particle accelerators, in [Reviews of Accel](https://doi.org/10.1142/S1793626812300006X)[erator Science and Technology](https://doi.org/10.1142/S1793626812300006X), Vol. 5, edited by A. W. Chao and W. Chou (WORLD SCIENTIFIC, Singapore, 2013) pp. 147–184.

<sup>[2]</sup> M. Kelly, Superconducting radio-frequency systems for low-beta particle accelerators, in [Reviews of Accelerator](https://doi.org/10.1142/S1793626812300071)

<sup>[3]</sup> S. Posen and D. L. Hall, Nb3Sn superconducting radiofrequency cavities: fabrication, results, properties, and prospects, [Superconductor Science and Technology](https://doi.org/10.1088/1361-6668/30/3/033004) 30[, 033004 \(2017\).](https://doi.org/10.1088/1361-6668/30/3/033004)

- <span id="page-21-0"></span>[4] R. C. Dhuley, S. Posen, M. I. Geelhoed, O. Prokofiev, and J. C. T. Thangaraj, First demonstration of a cryocooler conduction cooled superconducting radiofrequency cavity operating at practical cw accelerating gradients, [Super](https://doi.org/10.1088/1361-6668/ab82f0)[conductor Science and Technology](https://doi.org/10.1088/1361-6668/ab82f0) 33, 06LT01 (2020).
- [5] G. Ciovati, G. Cheng, U. Pudasaini, and R. A. Rimmer, Multi-metallic conduction cooled superconducting radiofrequency cavity with high thermal stability, [Supercon](https://doi.org/10.1088/1361-6668/ab8d98)[ductor Science and Technology](https://doi.org/10.1088/1361-6668/ab8d98) 33, 07LT01 (2020).
- [6] N. A. Stilin, A. T. Holic, M. Liepe, R. D. Porter, J. Sears, and Z. Sun, Conduction Cooling Methods for Nb3Sn SRF Cavities and Cryomodules, in [Proc. IPAC'21](https://doi.org/10.18429/JACoW-IPAC2021-MOPAB391) (JACoW Publishing, Geneva, Switzerland) pp. 1192–1195.
- [7] S. Henderson and T. Waite, Workshop on Energy and Environmental Applications of Accelerators, Argonne, IL, Tech. Rep. (U.S. Department of Energy, 2015).
- [8] G. Ciovati, J. Anderson, B. Coriton, J. Guo, F. Hannon, L. Holland, M. LeSher, F. Marhauser, J. Rathke, R. Rimmer, T. Schultheiss, and V. Vylet, Design of a cw, low-energy, high-power superconducting linac for environmental applications, [Phys. Rev. Accel. Beams](https://doi.org/10.1103/PhysRevAccelBeams.21.091601) 21, [091601 \(2018\).](https://doi.org/10.1103/PhysRevAccelBeams.21.091601)
- [9] R. C. Dhuley, I. Gonin, S. Kazakov, T. Khabiboulline, A. Sukhanov, V. Yakovlev, A. Saini, N. Solyak, A. Sauers, J. C. T. Thangaraj, K. Zeller, B. Coriton, and R. Kostin, Design of a 10 MeV, 1000 kW average power electron-beam accelerator for wastewater treatment applications, [Phys. Rev. Accel. Beams](https://doi.org/10.1103/PhysRevAccelBeams.25.041601) 25, 041601 (2022).
- [10] S. K. Vyas, R. K. Verma, S. Maurya, and V. Singh, Review of magnetron developments, [Frequenz](https://doi.org/doi:10.1515/freq-2015-0196) 70, 455 [\(2016\).](https://doi.org/doi:10.1515/freq-2015-0196)
- [11] A. C. Dexter, G. Burt, R. G. Carter, I. Tahir, H. Wang, K. Davis, and R. Rimmer, First demonstration and performance of an injection locked continuous wave magnetron to phase control a superconducting cavity, [Phys.](https://doi.org/10.1103/PhysRevSTAB.14.032001) [Rev. ST Accel. Beams](https://doi.org/10.1103/PhysRevSTAB.14.032001) 14, 032001 (2011).
- [12] G. Kazakevich, R. Johnson, G. Flanagan, F. Marhauser, V. Yakovlev, B. Chase, V. Lebedev, S. Nagaitsev, R. Pasquinelli, N. Solyak, K. Quinn, D. Wolff, and V. Pavlov, High-power magnetron transmitter as an rf source for superconducting linear accelerators, [Nuclear](https://doi.org/https://doi.org/10.1016/j.nima.2014.05.069) [Instruments and Methods in Physics Research Section](https://doi.org/https://doi.org/10.1016/j.nima.2014.05.069) [A: Accelerators, Spectrometers, Detectors and Associ](https://doi.org/https://doi.org/10.1016/j.nima.2014.05.069)[ated Equipment](https://doi.org/https://doi.org/10.1016/j.nima.2014.05.069) 760, 19 (2014).
- [13] C. Liu, H. Huang, Z. Liu, F. Huo, and K. Huang, Experimental study on microwave power combining based on injection-locked 15-kw s -band continuous-wave magnetrons, [IEEE Transactions on Plasma Science](https://doi.org/10.1109/TPS.2016.2565564) 44, 1291 [\(2016\).](https://doi.org/10.1109/TPS.2016.2565564)
- [14] G. Kazakevich, R. Johnson, V. Lebedev, V. Yakovlev, and V. Pavlov, Resonant interaction of the electron beam with a synchronous wave in controlled magnetrons for high-current superconducting accelerators, [Phys. Rev.](https://doi.org/10.1103/PhysRevAccelBeams.21.062001) Accel. Beams 21[, 062001 \(2018\).](https://doi.org/10.1103/PhysRevAccelBeams.21.062001)
- [15] X. Chen, B. Yang, N. Shinohara, and C. Liu, Modeling and experiments of an injection-locked magnetron with various load reflection levels, [IEEE Transactions on Elec](https://doi.org/10.1109/TED.2020.3009901)tron Devices 67[, 3802 \(2020\).](https://doi.org/10.1109/TED.2020.3009901)
- [16] W. Li, P. Zhang, B. Zhou, H. Zhang, Y. Liu, L. Zhang, and S. An, Development of a 1497 Mhz, 13.5 kW continuous-wave magnetron for superconducting radio frequency accelerator, [IEEE Transactions on Plasma Sci](https://doi.org/10.1109/TPS.2020.3048982)ence 49[, 663 \(2021\).](https://doi.org/10.1109/TPS.2020.3048982)
- [17] C.-S. Ha, T.-H. Kim, S.-R. Jang, J.-S. Kim, and S.-T. Han, Experimental study on the effect of the characteristics of a switching-mode power supply on a phase-locked magnetron, [IEEE Electron Device Letters](https://doi.org/10.1109/LED.2022.3156543) 43, 619 (2022).
- [18] H. Wang, J. Blum, B. Coriton, K. Jordan, C. Moeller, R. Nelson, S. Overstreet, R. Rimmer, K. Thackston, J. Vega, and G. Ziemyte, Magnetron R&D Progress for High Efficiency CW RF Sources of Industrial Accelerators, in [Proc. NAPAC'22](https://doi.org/10.18429/JACoW-NAPAC2022-WEZD3) , International Particle Accelerator Conference No. 5 (JACoW Publishing, Geneva, Switzerland, 2022) pp. 597–600.
- [19] M. J. de Loos and S. B. van der Geer, General Particle Tracer: A New 3D Code for Accelerator and Beamline Design, in Proc. EPAC'96 (JACoW Publishing, Geneva, Switzerland) pp. 1245 – 1247.
- [20] P. Pierini, D. Barni, A. Bosotti, G. Ciovati, and C. Pagani, Cavity Design Tools and Applications to the TRASCO Project, in [Proc. SRF'99](https://jacow.org/SRF99/papers/WEP004.pdf) (JACoW Publishing, Geneva, Switzerland) pp. 380–383.
- [21] J. H. Billen and L. M. Young, Poisson Superfish, Tech. Rep. LA-UR-96-1834 (Los Alamos National Laboratory, 1996) revised February 6, 2003.
- [22] E. Donoghue, G. Wu, J. Mammosser, R. Rimmer, M. Stirbet, L. Phillips, and H. Wang, Studies of Electron Activities in SNS-Type Superconducting RF Cavities, in [Proc. SRF'05](https://jacow.org/SRF2005/papers/TUP67.pdf) (JACoW Publishing, Geneva, Switzerland) pp. 402–405.
- [23] W. Xu, Z. Altinbas, S. Belomestnykh, I. Ben-Zvi, M. Cole, S. Deonarine, M. Falletta, J. Jamilkowski, D. Gassner, P. Kankiya, D. Kayran, N. Laloudakis, L. Masi, G. McIntyre, D. Pate, D. Philips, T. Seda, T. Schultheiss, A. Steszyn, T. Tallerico, R. Todd, D. Weiss, G. Whitbeck, and A. Zaltsman, Design, simulations, and conditioning of 500 kW fundamental power couplers for a superconducting rf gun, [Phys. Rev. ST Ac](https://doi.org/10.1103/PhysRevSTAB.15.072001)cel. Beams 15[, 072001 \(2012\).](https://doi.org/10.1103/PhysRevSTAB.15.072001)
- [24] CST Studio Suite, 3D electromagnetic software package, [https://www.3ds.com/products-services/simulia/](https://www.3ds.com/products-services/simulia/products/cst-studio-suite/) [products/cst-studio-suite/](https://www.3ds.com/products-services/simulia/products/cst-studio-suite/) (2022).
- [25] F. Marhauser, R. A. Rimmer, K. Tian, and H. Wang, Enhanced Method for Cavity Impedance Calculations, in [Proc. PAC'09](https://jacow.org/PAC2009/papers/FR5PFP094.pdf) (JACoW Publishing, Geneva, Switzerland) pp. 4523–4525.
- [26] U. Pudasaini, G. V. Eremeev, J. W. Angle, J. Tuggle, C. E. Reece, and M. J. Kelley, Growth of Nb3Sn coating in tin vapor-diffusion process, [Journal of Vac](https://doi.org/10.1116/1.5113597)[uum Science & Technology A](https://doi.org/10.1116/1.5113597) 37, 051509 (2019), [https://doi.org/10.1116/1.5113597.](https://arxiv.org/abs/https://doi.org/10.1116/1.5113597)
- [27] G. Ciovati, P. Dhakal, I. Parajuli, T. Saeki, and M. W. Pathiranage, Thermal Conductivity of Electroplated Copper Onto Bulk Niobium at Cryogenic Temperatures, in [Proc. SRF'21](https://doi.org/10.18429/JACoW-SRF2021-WEPFDV008) , International Conference on RF Superconductivity No. 20 (JACoW Publishing, Geneva, Switzerland, 2022) pp. 576–580.
- [28] Ansys Mechanical, 3D engineering and designing software, [https://www.ansys.com/products/structures/](https://www.ansys.com/products/structures/ansys-mechanical) [ansys-mechanical](https://www.ansys.com/products/structures/ansys-mechanical) (2022).
- [29] N. J. Simon, E. S. Drexler, and R. P. Reed, Properties of copper and copper alloys at cryogenic temperatures, Tech. Rep. NIST Monograph 177 (National Institute of Standards and Technology, 1992).
- [30] G. D. Cody and R. W. Cohen, Thermal conductivity of Nb3Sn, [Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.36.121) 36, 121 (1964).
- <span id="page-22-1"></span>[31] A. Dillon, K. McCusker, J. V. Dyke, B. Isler, and M. Christiansen, Thermal interface material characterization for cryogenic electronic packaging solutions, [IOP](https://doi.org/10.1088/1757-899X/278/1/012054) [Conference Series: Materials Science and Engineering](https://doi.org/10.1088/1757-899X/278/1/012054) 278[, 012054 \(2017\).](https://doi.org/10.1088/1757-899X/278/1/012054)
- [32] J. Halbritter, FORTRAN-Program for the computation of the surface impedance of superconductors, Tech. Rep. FZK 3/70-6 (Forschungszentrum Karlsruhe, 1970).
- [33] The power into the FPC was slightly higher than the target value because the Qext of the FPC in the 3D model was not exactly at the target value.
- [34] T. Satogata and Y. Zhang (JLEIC Design Study), JLEIC - A Polarized Electron-Ion Collider at Jefferson Lab, ICFA Beam Dyn. Newslett. 74, 92 (2018).
- [35] R. A. Rimmer, W. A. Clemens, F. Fors, J. Guo, F. E. Hannon, J. Henry, F. Marhauser, L. Turlington, H. Wang, and S. Wang, 952.6 MHz SRF Cavity Development for JLEIC, in [Proc. IPAC'18](https://doi.org/10.18429/JACoW-IPAC2018-THPAL144) (JACoW Publishing, Geneva, Switzerland) pp. 3982–3985.
- [36] F. Marhauser, [Recent Results on a Multi-Cell 802 MHz](https://indico.cern.ch/event/656491/contributions/2932251/attachments/1629681/2597650/5_cell_Cavity_Marhauser.pdf) [bulk Nb Cavity,](https://indico.cern.ch/event/656491/contributions/2932251/attachments/1629681/2597650/5_cell_Cavity_Marhauser.pdf) presented at FCC Week 2018 (9-13 April 2018).
- [37] E. Daly, G. K. Davis, and W. R. Hicks, Testing of the New Tuner Design for the CEBAF 12 GeV Upgrade SRF Cavities, in [Proc. PAC'05](https://jacow.org/p05/papers/TPPT073.pdf) (JACoW Publishing, Geneva, Switzerland) pp. 3910–3912.
- [38] Y. M. Pischalnikov, S. Cheban, and J. C. Yun, Design of 650 MHz Tuner for PIP-II Project, in [Proc. IPAC'18](https://doi.org/10.18429/JACoW-IPAC2018-WEPML002) (JACoW Publishing, Geneva, Switzerland) pp. 2671– 2673.
- [39] Y. Pischalnikov, E. Borissov, I. Gonin, J. Holzbauer, T. Khabiboulline, W. Schappert, S. Smith, and J. Yun, Design and Test of the Compact Tuner for Narrow Bandwidth SRF Cavities, in [Proc. 6th International Parti](https://doi.org/https://doi.org/10.18429/JACoW-IPAC2015-WEPTY035)[cle Accelerator Conference \(IPAC'15\), Richmond, VA,](https://doi.org/https://doi.org/10.18429/JACoW-IPAC2015-WEPTY035) [USA, May 3-8, 2015](https://doi.org/https://doi.org/10.18429/JACoW-IPAC2015-WEPTY035) , International Particle Accelerator Conference No. 6 (JACoW, Geneva, Switzerland, 2015) pp. 3352–3354, https://doi.org/10.18429/JACoW-IPAC2015-WEPTY035.
- [40] H. Padamsee, Applications and operations, in [RF Super](https://doi.org/10.1002/9783527627172.ch10)[conductivity](https://doi.org/10.1002/9783527627172.ch10) (John Wiley & Sons, Ltd, 2009) Chap. 10, pp. 313–332.
- [41] R. J. Corruccini and J. J. Gniewek, Thermal expansion of technical solids at low temperatures, Tech. Rep. NBS Monograph 29 (National Bureau of Standards, 1961).
- [42] M. Poirier, R. Plamondon, J. D. N. Cheeke, and J. F. Bussi`ere, Elastic constants of polycrystalline nb3sn between 4.2 and 300 k, [Journal of Applied Physics](https://doi.org/10.1063/1.333370) 55, 3327 [\(1984\),](https://doi.org/10.1063/1.333370) [https://doi.org/10.1063/1.333370.](https://arxiv.org/abs/https://doi.org/10.1063/1.333370)
- [43] G. Eremeev, W. Clemens, K. Macha, C. E. Reece, A. M. Valente-Feliciano, S. Williams, U. Pudasaini, and M. Kelley, Nb3Sn multicell cavity coating system at Jefferson Lab, [Review of Scientific Instruments](https://doi.org/10.1063/1.5144490) 91, 073911 (2020), [https://doi.org/10.1063/1.5144490.](https://arxiv.org/abs/https://doi.org/10.1063/1.5144490)
- [44] Y. Trenikhina, S. Posen, A. Romanenko, M. Sardela, J.- M. Zuo, D. L. Hall, and M. Liepe, Performance-defining properties of Nb3Sn coating in SRF cavities, [Supercon](https://doi.org/10.1088/1361-6668/aa9694)[ductor Science and Technology](https://doi.org/10.1088/1361-6668/aa9694) 31, 015004 (2017).
- [45] U. Pudasaini, C. Reece, and J. Tiskumara, Managing Sn-Supply to Tune Surface Characteristics of Vapor-Diffusion Coating of Nb3Sn, in [Proc. SRF'21](https://doi.org/10.18429/JACoW-SRF2021-TUPTEV013) , International Conference on RF Superconductivity No. 20 (JA-CoW Publishing, Geneva, Switzerland, 2022) pp. 516– 521.
- [46] D. Hall, M. Liepe, D. Liarte, and J. P. Sethna, Impact of trapped magnetic flux and thermal gradients on the performance of Nb3Sn cavities, in [Proc. 8th Int. Parti](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA118)[cle Accelerator Conf. \(IPAC'17\), Copenhagen, Denmark,](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA118) [May 2017](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA118) (JACoW Publishing, Geneva, Switzerland) pp. 1127–1129.
- [47] J. P. Turneaure, J. Halbritter, and H. A. Schwettman, The surface impedance of superconductors and normal conductors: The Mattis-Bardeen theory, [Journal of Su](https://doi.org/10.1007/BF00618215)[perconductivity](https://doi.org/10.1007/BF00618215) 4, 341 (1991).
- [48] M. Hein, Measurements of the surface impedance at linear response, in [High-Temperature-Superconductor Thin](https://doi.org/10.1007/BFb0111182) [Films at Microwave Frequencies](https://doi.org/10.1007/BFb0111182) (Springer Berlin, 1999) Chap. 2, pp. 43–102.
- <span id="page-22-0"></span>[49] T. Trollier, J. Tanchon, J. Lacapere, P. Renaud, J. Rey, and A. Ravex, Flexible thermal link assembly solutions for space applications, in [Proc. Cryocoolers 19](https://cryocooler.org/Cryocoolers-19) (ICC Press, Boulder, CO, 2016) pp. 595–603.
- [50] G. Cheng, G. Ciovati, and M. Morrone, Evaluation of Low Heat Conductivity RF Cables, in [Proc. SRF'19](https://doi.org/10.18429/JACoW-SRF2019-THP068) , International Conference on RF Superconductivity No. 19 (JACoW Publishing, Geneva, Switzerland, 2019) pp. 1045–1049, https://doi.org/10.18429/JACoW-SRF2019- THP068.
- [51] T. Powers, Practical Aspects of SRF Cavity Testing and Operations (2019), presented at SRF2019 in Dresden, Germany, unpublished.
- [52] D. Van Harlingen, Thermoelectric effects in the superconducting state, [Physica B+C](https://doi.org/https://doi.org/10.1016/0378-4363(82)90195-4) 109-110, 1710 (1982), 16th International Conference on Low Temperature Physics, Part 3.
- [53] C. D. Shelly, E. A. Matrozova, and V. T. Petrashov, Resolving thermoelectric "paradox" in superconductors, Science Advances 2[, e1501250 \(2016\),](https://doi.org/10.1126/sciadv.1501250) [https://www.science.org/doi/pdf/10.1126/sciadv.1501250.](https://arxiv.org/abs/https://www.science.org/doi/pdf/10.1126/sciadv.1501250)
- [54] F. Kramer, O. Kugeler, J.-M. K¨oszegi, and J. Knobloch, Impact of geometry on flux trapping and the related surface resistance in a superconducting cavity, [Phys. Rev.](https://doi.org/10.1103/PhysRevAccelBeams.23.123101) Accel. Beams 23[, 123101 \(2020\).](https://doi.org/10.1103/PhysRevAccelBeams.23.123101)
- [55] D. Longuevergne and A. Miyazaki, Impact of geometry on the magnetic flux trapping of superconducting accelerating cavities, [Phys. Rev. Accel. Beams](https://doi.org/10.1103/PhysRevAccelBeams.24.083101) 24, 083101 [\(2021\).](https://doi.org/10.1103/PhysRevAccelBeams.24.083101)
- [56] T. Powers, N. Brock, and G. Davis, Microphonics testing of LCLS II cryomodules at Jefferson Lab, in [Proc.](https://doi.org/doi:10.18429/JACoW-SRF2019-TUP034) [SRF'19](https://doi.org/doi:10.18429/JACoW-SRF2019-TUP034) , International Conference on RF Superconductivity No. 19 (JACoW Publishing, Geneva, Switzerland, 2019) pp. 493–498, https://doi.org/10.18429/JACoW-SRF2019-TUP034.
- [57] P. Bauer, N. Solyak, G. Ciovati, G. Eremeev, A. Gurevich, L. Lilje, and B. Visentin, Evidence for non-linear bcs resistance in srf cavities, [Physica C: Superconductiv](https://doi.org/https://doi.org/10.1016/j.physc.2006.03.056)ity 441[, 51 \(2006\),](https://doi.org/https://doi.org/10.1016/j.physc.2006.03.056) proceedings of the 12th International Workshop on RF Superconductivity.
- [58] A. Gurevich, Superconducting radio-frequency fundamentals for particle accelerators, [Reviews of Ac](https://doi.org/10.1142/S1793626812300058)[celerator Science and Technology](https://doi.org/10.1142/S1793626812300058) 05, 119 (2012), [https://doi.org/10.1142/S1793626812300058.](https://arxiv.org/abs/https://doi.org/10.1142/S1793626812300058)
- [59] S. W. Van Sciver, Pool boiling he i heat transfer, in [He](https://doi.org/10.1007/978-1-4899-0499-7_6)[lium Cryogenics](https://doi.org/10.1007/978-1-4899-0499-7_6) (Springer US, Boston, MA, 1986) pp. 199–238.
- [60] S. Posen and M. Liepe, Advances in development of nb3Sn superconducting radio-frequency cavities, [Phys.](https://doi.org/10.1103/PhysRevSTAB.17.112001) [Rev. ST Accel. Beams](https://doi.org/10.1103/PhysRevSTAB.17.112001) 17, 112001 (2014).
- <span id="page-23-0"></span>[61] J. Lee, Z. Mao, K. He, Z. H. Sung, T. Spina, S.-I. Baik, D. L. Hall, M. Liepe, D. N. Seidman, and S. Posen, Grainboundary structure and segregation in nb3sn coatings on nb for high-performance superconducting radiofrequency cavity applications, [Acta Materialia](https://doi.org/https://doi.org/10.1016/j.actamat.2020.01.055) 188, 155 (2020).
- [62] M. Suenaga and W. Jansen, Chemical compositions at and near the grain boundaries in bronze-processed superconducting nb3sn, [Applied Physics Letters](https://doi.org/10.1063/1.94457) 43, 791 [\(1983\),](https://doi.org/10.1063/1.94457) [https://doi.org/10.1063/1.94457.](https://arxiv.org/abs/https://doi.org/10.1063/1.94457)
- [63] M. J. R. Sandim, D. Tytko, A. Kostka, P. Choi, S. Awaji, K. Watanabe, and D. Raabe, Grain boundary segregation in a bronze-route nb3sn superconducting wire studied by atom probe tomography, [Superconductor Science](https://doi.org/10.1088/0953-2048/26/5/055008) [and Technology](https://doi.org/10.1088/0953-2048/26/5/055008) 26, 055008 (2013).
- [64] J. Lee, S. Posen, Z. Mao, Y. Trenikhina, K. He, D. L. Hall, M. Liepe, and D. N. Seidman, Atomic-scale analyses of Nb3Sn on Nb prepared by vapor diffusion for superconducting radiofrequency cavity applications: a correlative study, [Superconductor Science and Technology](https://doi.org/10.1088/1361-6668/aaf268) 32[, 024001 \(2018\).](https://doi.org/10.1088/1361-6668/aaf268)
- [65] T. L. Hylton and M. R. Beasley, Effect of grain boundaries on magnetic field penetration in polycrystalline superconductors, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.39.9042) 39, 9042 (1989).
- [66] J. McDonald and J. R. Clem, Microwave response and surface impedance of weak links, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.56.14723) 56, 14723 [\(1997\).](https://doi.org/10.1103/PhysRevB.56.14723)
- [67] A. Sheikhzada and A. Gurevich, Dynamic transition of vortices into phase slips and generation of vortexantivortex pairs in thin film josephson junctions under dc and ac currents, Phys. Rev. B 95[, 214507 \(2017\).](https://doi.org/10.1103/PhysRevB.95.214507)
- [68] A. Godeke, A review of the properties of Nb3Sn and their variation with A15 composition, morphology and strain state, [Superconductor Science and Technology](https://doi.org/10.1088/0953-2048/19/8/R02) 19, R68 [\(2006\).](https://doi.org/10.1088/0953-2048/19/8/R02)
- [69] G. D. Marzi, L. Morici, L. Muzzi, A. della Corte, and M. B. Nardelli, Strain sensitivity and superconducting properties of Nb3Sn from first principles calculations, [Journal of Physics: Condensed Matter](https://doi.org/10.1088/0953-8984/25/13/135702) 25, 135702 (2013).
- [70] H. Ding and Y. Gao, Analysis of the strain dependence of the superconducting critical properties of single-crystal and polycrystalline Nb3Sn, [Superconductor Science and](https://doi.org/10.1088/1361-6668/abfaef) Technology 34[, 075006 \(2021\).](https://doi.org/10.1088/1361-6668/abfaef)
- <span id="page-23-1"></span>[71] A. Gurevich, Theory of rf superconductivity for resonant cavities, [Superconductor Science and Technology](https://doi.org/10.1088/1361-6668/30/3/034004) 30[, 034004 \(2017\).](https://doi.org/10.1088/1361-6668/30/3/034004)
- <span id="page-23-2"></span>[72] J. Makita, C. Sundahl, G. Ciovati, C. B. Eom, and A. Gurevich, Nonlinear Meissner effect in Nb3Sn coplanar resonators, [Phys. Rev. Res.](https://doi.org/10.1103/PhysRevResearch.4.013156) 4, 013156 (2022).
- <span id="page-23-3"></span>[73] U. Pudasaini, G. V. Eremeev, C. E. Reece, J. Tuggle, and M. J. Kelley, Analysis of RF losses and material characterization of samples removed from a Nb3Sn-coated superconducting RF cavity, [Superconductor Science and](https://doi.org/10.1088/1361-6668/ab75a8) Technology 33[, 045012 \(2020\).](https://doi.org/10.1088/1361-6668/ab75a8)
- <span id="page-23-4"></span>[74] G. Eremeev, W. Crahen, J. Henry, F. Marhauser, U. Pudasaini, and C. Reece, RF Performance Sensitivity to Tuning of Nb3Sn Coated CEBAF Cavities, in [Proc.](https://doi.org/10.18429/JACoW-SRF2019-MOP015) [SRF'19](https://doi.org/10.18429/JACoW-SRF2019-MOP015) , International Conference on RF Superconductivity No. 19 (JACoW Publishing, Geneva, Switzerland, 2019) pp. 55–59, https://doi.org/10.18429/JACoW-SRF2019-MOP015.
- <span id="page-23-5"></span>[75] T. Tomaru, T. Suzuki, T. Haruyama, T. Shintomi, A. Yamamoto, T. Koyama, and R. Li, Vibration analysis of cryocoolers, Cryogenics 44[, 309 \(2004\).](https://doi.org/https://doi.org/10.1016/j.cryogenics.2004.02.003)
- <span id="page-23-6"></span>[76] R. Bachimanchi, T. Allison, E. Daly, M. Drury, C. Hovater, G. Lahti, C. Mounts, R. Nelson, and T. E. Plawski, CEBAF SRF Performance during Initial 12 GeV Commissioning, in [Proc. 6th International Parti](https://doi.org/https://doi.org/10.18429/JACoW-IPAC2015-THXB1)[cle Accelerator Conference \(IPAC'15\), Richmond, VA,](https://doi.org/https://doi.org/10.18429/JACoW-IPAC2015-THXB1) [USA, May 3-8, 2015](https://doi.org/https://doi.org/10.18429/JACoW-IPAC2015-THXB1) , International Particle Accelerator Conference No. 6 (JACoW, Geneva, Switzerland, 2015) pp. 3638–3642, https://doi.org/10.18429/JACoW-IPAC2015-THXB1.
- <span id="page-23-7"></span>[77] Model PT420, Cryomech, Syracuse, NY, USA, [https:](https://www.cryomech.com/products/pt420/) [//www.cryomech.com/products/pt420/](https://www.cryomech.com/products/pt420/), accessed: 2023- 01-10.
- <span id="page-23-8"></span>[78] Model RDE-418D4, Sumitomo (SHI) Cryogenics of America, Inc, Allentown, PA, USA, [https://www.shicryogenics.com/product/](https://www.shicryogenics.com/product/rde-418d4-4k-cryocooler-series/) [rde-418d4-4k-cryocooler-series/](https://www.shicryogenics.com/product/rde-418d4-4k-cryocooler-series/), accessed: 2023- 01-10.
- <span id="page-23-9"></span>[79] A. L. Woodcraft, Recommended values for the thermal conductivity of aluminium of different purities in the cryogenic to room temperature range, and a comparison with copper, Cryogenics 45[, 626 \(2005\).](https://doi.org/https://doi.org/10.1016/j.cryogenics.2005.06.008)
- <span id="page-23-10"></span>[80] R. C. Dhuley, R. Kostin, O. Prokofiev, M. I. Geelhoed, T. H. Nicol, S. Posen, J. C. T. Thangaraj, T. K. Kroc, and R. D. Kephart, Thermal link design for conduction cooling of srf cavities using cryocoolers, [IEEE Transac](https://doi.org/10.1109/TASC.2019.2901252)[tions on Applied Superconductivity](https://doi.org/10.1109/TASC.2019.2901252) 29, 1 (2019).
- <span id="page-23-11"></span>[81] R. Dhuley, M. Geelhoed, and J. Thangaraj, Thermal resistance of pressed contacts of aluminum and niobium at liquid helium temperatures, Cryogenics 93[, 86 \(2018\).](https://doi.org/https://doi.org/10.1016/j.cryogenics.2018.06.003)
- <span id="page-23-12"></span>[82] A. R. Kerr and N. Horne, The low temperature thermal resistance of high purity copper and bolted copper joints., Tech. Rep. EDTN 163 (National Radio Astronomy Observatory - Electronic Division, 1991).
- [83] A. R. Kerr and R. Groves, Measurements of Copper Heat Straps Near 4 K With and Without Apiezon-N Grease, Tech. Rep. EDTN 204 (National Radio Astronomy Observatory - Electronic Division, 2006).
- [84] L. Salerno, P. Kittel, and A. Spivak, Thermal conductance of pressed metallic contacts augmented with indium foil or Apiezon grease at liquid helium temperatures, Cryogenics 34[, 649 \(1994\).](https://doi.org/https://doi.org/10.1016/0011-2275(94)90142-2)
- <span id="page-23-13"></span>[85] R. Dhuley, Pressed copper and gold-plated copper contacts at low temperatures – a review of thermal contact resistance, Cryogenics 101[, 111 \(2019\).](https://doi.org/https://doi.org/10.1016/j.cryogenics.2019.06.008)
- <span id="page-23-14"></span>[86] W. K. Withanage, A. Juliao, and L. D. Cooley, Rapid Nb3Sn film growth by sputtering Nb on hot bronze, [Su](https://doi.org/10.1088/1361-6668/abf66f)[perconductor Science and Technology](https://doi.org/10.1088/1361-6668/abf66f) 34, 06LT01 (2021).
- [87] E. A. Ilyina, G. Rosaz, J. B. Descarrega, W. Vollenberg, A. J. G. Lunt, F. Leaux, S. Calatroni, W. Venturini-Delsolaro, and M. Taborelli, Development of sputtered nb3sn films on copper substrates for superconducting radiofrequency applications, [Superconductor Science and](https://doi.org/10.1088/1361-6668/aaf61f) Technology 32[, 035002 \(2019\).](https://doi.org/10.1088/1361-6668/aaf61f)
- [88] Z. Sun, T. Arias, Z. Baraissov, K. Dobson, G. Gaitan, M. Ge, K. Howard, M. Kelley, M. Liepe, D. Muller, T. Oseroff, R. Porter, J. Sethna, and N. Sitaraman, Toward Stoichiometric and Low-Surface-Roughness Nb3Sn Thin Films via Direct Electrochemical Deposition, in [Proc. SRF'21](https://doi.org/10.18429/JACoW-SRF2021-WEOTEV03) , International Conference on RF Superconductivity No. 20 (JACoW Publishing, Geneva, Switzerland, 2022) pp. 710–713.
- <span id="page-23-15"></span>[89] M. Ge, V. Arrieta, T. Gruber, J. Kaufman, M. Liepe, J. Maniscalco, S. McNeal, T. Oseroff, R. Porter, and Z. Sun, CVD Coated Copper Substrate SRF Cav-

ity Research at Cornell University, in [Proc. SRF'19](https://doi.org/10.18429/JACoW-SRF2019-TUFUB8) , International Conference on RF Superconductivity No. 19 (JACoW Publishing, Geneva, Switzerland,

2019) pp. 381–386, https://doi.org/10.18429/JACoW-SRF2019-TUFUB8.