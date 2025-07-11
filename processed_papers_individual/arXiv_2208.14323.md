# arXiv:2208.14323

**Paper ID:** 4c28fd042a3477f06d819b0192a5d12c

**PDF Path:** apl/Superconductors/arXiv:2208.14323.pdf

**Processing Status:** complete

**Captions Added:** 6

**Generated:** 2025-06-24T13:44:27.673223

---

P. Ferracin, G. Ambrosio, D. Arbelaez, L. Brouwer, E. Barzi, L. Cooley, L. Garcia Fajardo, R. Gupta, M. Juchno, V. Kashikhin, V. Marinozzi, I. Novitski, E. Rochepault, J. Stern, A. Zlobin, N. Zucchi

*Abstract***—The most effective way to achieve very high collision energies in a circular particle accelerator is to maximize the field strength of the main bending dipoles. In dipole magnets using Nb-Ti superconductor the practical field limit is considered to be 8-9 T. When Nb3Sn superconductor material is utilized, a field level of 15- 16 T can be achieved. To further push the magnetic field beyond the Nb3Sn limits, High Temperature Superconductors (HTS) need to be considered in the magnet design. The most promising HTS materials for particle accelerator magnets are Bi2212 and REBCO. However, their outstanding performance comes with a significantly higher cost. Therefore, an economically viable option towards 20 T dipole magnets could consist in an "hybrid" solution, where both HTS and Nb3Sn materials are used. We discuss in this paper preliminary conceptual designs of various 20 T hybrid magnet concepts. After the definition of the overall design criteria, the coil dimensions and parameters are investigated with finite element models based on simple sector coils. Preliminary 2D cross-section computation results are then presented and three main layouts compared: cos-theta, block, and common-coil. Both traditional designs and more advanced stress-management options are considered.**

*Index Terms***— Superconducting magnets, dipole magnets, Nb3Sn magnets, HTS, hybrid magnets.**

#### I. INTRODUCTION

IGH-ENERGY proton-proton circular colliders are among the most powerful tool for direct discovery of new particles and interactions [1]. In a circular accelerator, the most effective way to achieve very high collision energies is to maximize the field strength of the main bending dipoles. So far, particle accelerators like the Tevatron [2], HERA [3], RHIC [4], and LHC [5] have used Nb-Ti, a low temperature superconducting (LTS) material, in their main dipole magnets to achieve fields up to the 9 T level. In the High-Luminosity LHC, for the first time, superconducting magnets based on Nb3Sn, also an LTS material, and operating at a 11-12 T field level will be installed in the LHC interaction regions to increase the collision rate [6]. In parallel, R&D programs in Europe, as part of the FCC collaboration [7], and in the US, as part of the national Magnet Development Program (MDP) [8], are developing superconducting magnets aiming at bore fields of 15 to 16 T, currently considered as the practical limit for Nb3Sn accelerator magnets [9]. **H**

This work was supported by the U.S. Department of Energy, Office of Science, Office of High Energy Physics, through the US Magnet Development Program (*Corresponding author: Paolo Ferracin).*

P. Ferracin. D. Arbelaez, L. Brouwer, L. Garcia Fajardo, and M. Juchno are with Lawrence Berkeley National Lab, Berkeley, CA 94720, USA (e-mail: [pfer](mailto:pferracin@lbl.gov)[racin@lbl.gov\).](mailto:pferracin@lbl.gov)

G. Ambrosio, E. Barzi, V.V. Kashikhin, V. Marinozzi, I. Novitski, and A.V. Zlobin are with Fermi National Accelerator Laboratory, Batavia, IL 80510 USA.

To further push the magnetic field of the dipole magnets beyond the Nb3Sn limits, High Temp. Superconductors (HTS) need to be considered in the magnet design. For accelerator magnets, the most promising HTS materials currently under consideration are Bi2212 [10] and REBCO [11]. However, their outstanding performance still comes with a significantly higher cost than Nb3Sn. Therefore, an economically viable option of 20 T dipole magnets has to involve a "hybrid" approach, where HTS materials are used in the high field part of the coil with socalled "insert coils", and Nb3Sn and Nb-Ti superconductors are adopted in the lower field region with so-called "outsert coils". Preliminary design studies of 20 T hybrid dipole magnets were carried out in 2005 [12] and in 2014-2016 [13]-[15], whereas a full HTS option was analyzed in 2018 [16]. In 2015, a 20 T hybrid block-type design was presented by G. Sabbi, *et al.* in [17]. A hybrid magnet was recently attempted by inserting a REBCO coil inside the FRESCA2 dipole magnet [18]. Finally, REBCO inserts based on Roebel cables were fabricated and tested as part of the EUCARD2 Collaboration [19]-[20].

In this paper a preliminary conceptual magnetic design analysis of a 20 T hybrid dipole magnet for particle accelerators implementing Nb3Sn and HTS coils (Nb-Ti is not considered in this study) is presented. First, a description of the superconducting material properties and of the criteria used in the different designs is provided. Then, an analysis of coil size using sector coils, followed by the magnetic analysis of 3 types of coil layouts, 1) cos-theta, with and without stress management, 2) block-type, with and without stress management, and 3) common-coil, are presented. The work focuses on the field level and margins, and only preliminary considerations related to coil mechanics will be provided; a more detailed analysis will follow.

![](_page_0_Picture_12.jpeg)

**Caption:** Depicted are cross-sections of three conductor arrangements: on the left, a Nb3Sn strand with a Rod Restack Process (RRP) architecture; in the center, a Bi2212 strand showing bundled filaments in a silver matrix; on the right, a schematic of CORC/STAR wires in grooves, illustrating the structure for REBCO cables. These images are not to scale but are critical for understanding the physical construction of the superconductors analyzed. The variations in architecture emphasize the different design strategies and fabrication techniques being employed to enhance superconducting properties, crucial for advancing high-performance magnet technology.


Fig. 1. Cross-sections, not in scale, of Nb3Sn (left, 0.85 mm ) and Bi2212 (center, 0.7 mm ) composite wires produced by Bruker-OST, and a REBCO CORC wire (right, 3.4 mm ) by ACT LLC.

R. Gupta is with BNL, Upton, NY 11973-5000, USA.

L. D. Cooley is with the Applied Superconductivity Center, National High Magnetic Laboratory, Tallahassee, FL 32310, USA

E. Rochepault is with IRFU, CEA, Univers Paris-Saclay, Paris F-91191, France.

J. Stern, and N. Zucchi are with TUFTS University, 419 Boston Ave, Medford, MA 02155, USA.

1

#### II. CONDUCTOR PARAMETERS

An example of the superconducting materials considered for this analysis is shown in Fig. 1, with a Nb3Sn Rod restack Process (RRP) and a Bi2212 strands produced by Bruker-OST, and a REBCO CORC wire produced by APC LLC. For each material, the engineering current density *j<sup>e</sup>* used in the computations, where *j<sup>e</sup>* is the critical strand current divided by the strand crosssection area, is plotted in Fig. 2. A summary of the superconductor properties is provided in Table I.

![](_page_1_Figure_2.jpeg)

**Caption:** The figure illustrates the critical engineering current density (Je in A/mm²) plotted against the magnetic field (Field in T) for three superconducting materials: Nb3Sn at 1.9 K, Bi2212 at 4.2 K, and REBCO (CORC/STAR) at 4.2 K. The data is presented with different line styles for each material: dashed blue line for Nb3Sn, solid black line for Bi2212, and dash-dotted black line for REBCO. The decreasing trend of Nb3Sn's Je with increasing field highlights its superior high-field performance compared to Bi2212 and REBCO at higher field strengths. This information is significant in selecting superconducting materials for applications in high-field environments, providing insight into the operational limitations and effectiveness of these superconductors.


Fig. 2. Critical engineering current density (*j<sup>e</sup>* = Istrand/Astrand ) assumed for the computations for Nb3Sn, Bi2212 and REBCO CORC/STAR round wires.

### *A. Nb3Sn*

Nb3Sn strands have been used in both R&D magnets and accelerator magnets in diameters ranging typically from 0.7 to 1.1 mm [9], [21]. The cross-section of a 0.85 mm strand with 132/169 RRP architecture is shown in Fig. 1, left. Nb3Sn Rutherford cables have been fabricated with up to 60 strands, with widths ranging from 7.8 to 26.2 mm and thicknesses within 1.2- 2.0 mm. As an example, the cross-section of the Nb3Sn Rutherford cable fabricated for the MQXF project is shown in Fig. 3, top [22]. For the numerical computations we assumed a critical current density in the Nb3Sn (virgin strand) of 3000 A/mm<sup>2</sup> at 12 T and 4.2 K. Considering a 1.1 Cu/Non-Cu ratio, this corresponds to a *j<sup>e</sup>* of 870 A/mm<sup>2</sup> at 16 T, 1.9 K, including 5% of cabling degradation. Assuming a 0.150 mm thick insulation one obtains a ratio between *j<sup>o</sup>* and *j<sup>e</sup>* of 0.67 (using the MQXF insulated cable parameters [22]), where *j<sup>o</sup>* is the ratio of the cable current to the insulated cable area.

#### *B. Bi2212*

The cross-section of a Bruker-OST Bi-2212 strand, produced with the powder-in-tube method is shown in Fig. 1, center. The Bi2212 filaments are combined in 18 bundles (of 55 filaments each) and embedded in a silver matrix. The reaction process is performed following an overpressure heat treatment (OPHT), which allowed a significant improvement of the *j<sup>e</sup>* in recent years. For the analysis in this paper, we assumed a *j<sup>e</sup>* of 740 A/mm<sup>2</sup> at 1.9 K and 20 T, a value obtained in short samples for the racetrack sub-scale coils [23]. As Nb3Sn, the Bi2212 strands will be used in form of Rutherford cable. In terms of strand and cable dimensions, we assumed that the same ranges as for the Nb3Sn can be achieved.

|                                            |       | TABLE I |          |         |  |  |  |
|--------------------------------------------|-------|---------|----------|---------|--|--|--|
| SUPERCONDUCTOR STRAND AND CABLE PROPERTIES |       |         |          |         |  |  |  |
| Parameter                                  | Unit  | Nb3Sn   | Bi2212   | REBCO   |  |  |  |
| Strand diameter                            | mm    | 0.7-1.1 | 0.7-1.1  | 1.2-4.0 |  |  |  |
| Cable width                                | mm    |         | 7.8-26.2 | NA      |  |  |  |
| Cable thickness                            | mm    | 1.2-2.0 | 1.2-2.0  | NA      |  |  |  |
| je (at 1.9 K, 16 T)                        | A/mm2 | 870     | 800      | 700     |  |  |  |
| je (at 1.9 K, 20 T)                        | A/mm2 | 360     | 740      | 590     |  |  |  |
| jo / je                                    |       | 0.67    | 0.67     | 0.54    |  |  |  |

#### *C. REBCO*

The CORC (Conductor on Round Core) wire from ACT, shown in Fig. 1, right and described in details in [24], is fabricated by winding several REBCO tapes around a Cu core, with a total wire diameter of about 3-4 mm. These wires were successfully tested in Canted Cos-theta (CCT) coils at LBNL in 2019 [25]. Smaller diameters, of 1.3-2.0 mm, have been achieved with STAR (Symmetric Tape Round) wires described in [26]. In both cases, we assumed for the computations a *j<sup>e</sup>* of 590 A/mm<sup>2</sup> at 1.9 K and 20 T. In order to increase the total current carrying capabilities, two paths are being explored: an increase of the number of tapes and of the diameter of the wire, or a combination of smaller wires in a multi-strand cable. For this analysis we considered the option of a large CORC wire with a 6.5 mm diameter, individually powered and inserted in grooves, as shown in Fig. 3, bottom. In this configuration, the *j<sup>o</sup> / j<sup>e</sup>* ratio reduces, from 0.67 typical for Rutherford cable to 0.54.

![](_page_1_Figure_12.jpeg)

**Caption:** This schematic illustrates a typical cross-section of an Nb3Sn Rutherford cable alongside a conceptual view of CORC/STAR wires set in grooves. The top image shows a wide Nb3Sn cable, critical for high current applications in accelerators. The bottom diagram represents the structural integration of REBCO-based wires, highlighting the geometric adaptations necessary for managing electromagnetic loads in complex cable arrays. Such designs are pivotal for advancing the thermal and mechanical robustness necessary for future accelerator magnets.


Fig. 3. Figures not in scale. Top: cross-section of the insulated Nb3Sn Rutherford cable for MQXF (18.7 mm wide); bottom: schematic view of CORC/STAR wires in grooves (total width 55 mm).

#### III. DESIGN CRITERIA

The criteria defined in the conceptual design are summarized in Table II. The magnet shall be able to produce 20 T in a 50 mm clear aperture with at least 15% of load-line margin. This means that the "short-sample" bore field, i.e. the bore field achieved when the magnet reaches the current limits established by the conductor critical surface, is 23.5 T. The design shall have all the geometrical harmonics field below 5 units at the nominal field and at 2/3 of the aperture radius (magnetization effects are not included at this stage). All the coils shall be powered in series (a condition which impacts both the magnetic design and the quench protection system), and the hot spot temperature at quench shall be limited to 350 K, both in the Nb3Sn [27] and in the HTS coils [28]-[30]. In terms of stress limits, for the Nb3Sn we chose 150 and 180 MPa at 293 K and 1.9 K respectively, consistently with results published in [31], whereas for the HTS we defined a preliminary and more conservative value of 120 MPa [32], [33]. Both quench protection and coil stress levels will not be considered in the preliminary design studies presented in this paper (Section V).

TABLE II DESIGN CRITERIA ON MAGNET PARAMETERS

| Parameter                                 | Unit |     |
|-------------------------------------------|------|-----|
| Aperture                                  | mm   | 50  |
| Operational temperature                   | K    | 1.9 |
| Operational bore field B0_op              | T    | 20  |
| Load-line margin                          | %    | 15 |
| Geometrical harmonics (20 T, Rref=17 mm)  | unit | <5  |
| Maximum Nb3Sn coil eq. stress (293 K)     | MPa  | 150 |
| Maximum Nb3Sn coil eq. stress (1.9 K)     | MPa  | 180 |
| Maximum HTS coil eq. stress (293K, 1.9 K) | MPa  | 120 |
| Maximum hot spot temperature              | K    | 350 |

#### IV. ANALYSIS WITH SECTOR COILS

For a preliminary investigation of the overall coil size, loadline margins and ratio between LTS and HTS coil area, we performed a magnetic analysis using sector coils. The study follows the same approach as that described in [34], [35], where the superconducting coil is simulated with 60 sector, and with a uniform *jo*. The cross-sections of the analyzed sector coils analyzed are shown in Fig. 4, where the outserts (in red) are assigned the properties of the Nb3Sn and the inserts (in grey) the properties of Bi2212. The analysis includes an iron yoke (not shown in the figure), which starts at 25 mm from the coil outer radius of the coil and has a thickness of 250 mm. The Bi2212 coil has an aperture of 50 mm, and the two coils are dimensioned so that with a 20 T bore field they both operate at 85% of their limit on the load-line.

![](_page_2_Figure_5.jpeg)

**Caption:** This figure displays the configuration and load-line analysis of sector coils composed of Bi2212 as an insert and Nb3Sn as an outsert. Case I (top) applies the same engineering current density (Je) to both the insert and outsert coils, whereas Case II (bottom) reduces the Je in the outsert by 33% compared to the insert. Both configurations aim to achieve a 20 T bore field. This analysis demonstrates the potential to balance coil dimensions and material utilization in achieving high magnetic fields, an important consideration for efficient and cost-effective magnet design.


Fig. 4. Sector coils with Bi2212 coil as insert and Nb3Sn coil as outsert. Case I (top): 20 T dipole generated with equal *jo* in both insert and outsert (left). Case II (bottom): 20 T dipole generated with *jo* in the outsert 33% lower than in the insert (left). For both cases, the stand-alone coils are also shown (center and right).

The main parameters of the different cross-sections are listed in Table III, and the corresponding load-lines are plotted in Fig. 5. Two cases were studied: in the first one (Case I, Fig. 4 top) we applied the same uniform *j<sup>o</sup>* to both insert and outsert coils. This can be seen in Fig. 5, where the two markers in the Case I load-line pointed to a *j<sup>o</sup>* = 570 A/mm<sup>2</sup> in both Nb3Sn and Bi2212 for a bore field of 20 T, and to a *j<sup>o</sup>* = 570 A/mm<sup>2</sup> for the short sample condition. For this case, the overall coil width is 69 mm, and the majority of it (47 mm) is given by the Bi2212 coil. If the insert and outsert coils are tested in stand-alone with iron (Fig. 4, center and right), at 85% of their load-line limits they generate a bore field of 16.0 T in 50 mm aperture and 10.7 T in 144 mm aperture respectively.

| TABLE III                                   |
|---------------------------------------------|
| SECTOR COILS PARAMETERS FOR 20 T BORE FIELD |

| Parameter                             | Unit  | Case I    | Case II   |  |  |  |  |
|---------------------------------------|-------|-----------|-----------|--|--|--|--|
| jo<br>in insert (Bi2212)              | A/mm2 | 570       | 570       |  |  |  |  |
| jo<br>in outsert (Nb3Sn)              | A/mm2 | 570       | 380       |  |  |  |  |
| Coil width insert/outsert             | mm    | 47/22     | 33/55     |  |  |  |  |
| Area quadrant coil insert/outsert     | mm2   | 2387/1912 | 1434/4924 |  |  |  |  |
| Bbore stand-alone insert/outsert      | T     | 16.0/10.7 | 12.5/14.1 |  |  |  |  |
|  hybrid                             | MPa   | -191      | -177      |  |  |  |  |
| r hybrid                             | MPa   | -212      | -196      |  |  |  |  |
|  stand-alone insert/outsert         | MPa   | -144/-249 | -108/-175 |  |  |  |  |
| rstand-alone insert/outsert          | MPa   | -142/-85  | -93/-126  |  |  |  |  |
| E hybrid (4 quad)                     | MJ/m  | 2.3       | 2.6       |  |  |  |  |
| E stand-alone insert/outsert (4 quad) | MJ/m  | 1.0/1.4   | 0.5/2.47  |  |  |  |  |

In order to reduce the amount of HTS material, in Case II we imposed a lower *j<sup>o</sup>* to the Nb3Sn with respect to the Bi2212 (380 vs 570 A/mm<sup>2</sup> ) coils and we re-optimized the dimension to reobtain 20 T bore field with a 15% margin in both HTS and LTS coils. The result is a significant increase in Nb3Sn coil, but with a substantial reduction in Bi2212 coil. This is an interesting result if the goal is to minimize the amount of HTS. This means that if with an increase of the *j<sup>o</sup>* in the outer part of the coil (*grading*) one obtains an overall coil size reduction, in a hybrid situation one can follow an opposite approach (a sort of *antigrading*) to minimize the amount of HTS material.

![](_page_2_Figure_12.jpeg)

**Caption:** Illustrated here are the critical engineering current densities (Je) of Nb3Sn and Bi2212, together with the load-lines representing peak field scenarios for sector coils configured as hybrids. The graph contrasts different engineering approaches by plotting current density against coil peak field, showing performance limits of hybrid magnet designs operating at 20 T. The data suggest effective field management via modification of superconducting materials and their spatial distribution within the coil, important for maximizing magnet efficiency and stability in high energy physics applications.


Fig. 5. Critical engineering current density (*j<sup>e</sup>* = Istrand/Astrand) for Nb3Sn and Bi2212, and sector coil load-lines (engineering current density vs. coil peak field) for the two cases analyzed in Fig. 4.

If now the accumulated stress due to the azimuthal and radial electro-magnetic (e.m.) forces are considered we can see in Table III that in in both hybrid magnets the stresses exceed the 180 MPa limit. Interestingly, with such wide coil, the radial stress appears to be the most critical, with generally smaller values for

![](_page_3_Figure_0.jpeg)

**Caption:** This figure shows the cross-sections of various 20 T hybrid dipole coil designs. The Cos-theta (CT) design features 4 (top) and 2 (bottom) layer configurations of Bi2212. The Stress Management Cos-theta (SMCT) design also includes 4 (top) and 2 (bottom) Bi2212 layers. The Canted Cos-theta (CCT) configuration utilizes 4 layer Bi2212 coils. The Block (BL) design is presented with and without stress management features, and the Common Coil (CC) design incorporates Bi2212 in combination with 3 (top) and 5 (center) external Nb3Sn layers, as well as REBCO CORC with 4 external Nb3Sn layers (bottom). These designs reflect innovative strategies in superconducting magnet structures, essential for optimally distributing the fields and managing mechanical stresses.


Fig. 6. Cross-sections of 20 T hybrid dipole coils. From left two right: Cos-theta (CT) design, with 4 (top) and 2 (bottom) layer Bi2212 coils; Stress management Cos-theta (SMCT) design, with 4 (top) and 2 (bottom) layers Bi2212 coils; Canted Cos-theta (CCT) design, with 4-layer Bi2212 coil; Block (BL) design, with and without stress management; Common Coil (CC) design, with Bi2212 (top, with 3 external Nb3Sn layers, and center, with 5 external Nb3Sn layers) and REBCO CORC coils (bottom, with 4 external Nb3Sn layers). For all the CC designs, only one aperture is shown.

|                             |      |           |                                                                                                                         |            | TABLE IV                      |          |          |                                                                                                             |            |            |            |
|-----------------------------|------|-----------|-------------------------------------------------------------------------------------------------------------------------|------------|-------------------------------|----------|----------|-------------------------------------------------------------------------------------------------------------|------------|------------|------------|
|                             |      |           |                                                                                                                         |            | 20 T HYBRID MAGNET PARAMETERS |          |          |                                                                                                             |            |            |            |
| Parameter                   | Unit | CT I      | CT II                                                                                                                   | SMCT I     | SMCT II                       | CCT      | BL I     | BL II                                                                                                       | CC I       | CC II      | CC III     |
| Ins. cable I width/thick.   | mm   | 10.7/1.5  | 12.0 / 1.7                                                                                                              | 12.3 / 1.5 | 12.3 / 1.5                    | 18.7/1.9 | 17.1/2.1 | 17.1/2.1                                                                                                    | 18.7 / 1.8 | 18.7 / 1.8 | 7.5 / 7.5  |
| Ins. cable II width/thick.  | mm   | 9.4/1.5   | 14.2 / 2.1                                                                                                              | 10.7 / 1.5 | 13.9 / 1.5                    | -        | 17.1/2.1 | 17.1/2.1                                                                                                    | 13.6 / 1.9 | 13.6 / 1.9 | 21.6 / 1.9 |
| Ins. cable III width/thick. | mm   | 9.3/1.5   | 7.9 / 1.5                                                                                                               | 9.1 / 1.5  | 9.1 / 1.5                     | -        | -        | -                                                                                                           | -          | -          | -          |
| Current_op                  | kA   | 10.7      | 13.0                                                                                                                    | 11.4       | 11.8                          | 12.8     | 12.6     | 12.2                                                                                                        | 14         | 13.9       | 17.8       |
| B_bore_op                   | T    | 20.0      | 20.0                                                                                                                    | 20.1       | 20.0                          | 20.0     | 20.0     | 20.0                                                                                                        | 20.0       | 20.0       | 20.0       |
| B_peak_op HTS/LTS           | T    |           | 20.5 / 12.7 20.3 / 16.1 20.6 / 13.6 20.6 / 16.0 20.2 / 13.2 20.6 / 15.1 20.9 / 15.2 20.4 / 13.8 20.2 / 13.7 21.0 / 17.0 |            |                               |          |          |                                                                                                             |            |            |            |
| B_bore_ss                   | T    | 24.4      | 23.5                                                                                                                    | 24.4       | 23.2                          | 23.4     | 23.6     | 23.6                                                                                                        | 22.9       | 23         | 21.7       |
| B_peak_ss HTS/LTS           | T    | 24.9/15.4 |                                                                                                                         |            |                               |          |          | 23.8 / 17.7 24.9 / 16.4 23.8 / 18.4 23.6 / 12.9 24.3 / 17.7 24.7 / 18.0 23.3 / 15.7 23.3 / 15.7 24.7 / 18.2 |            |            |            |
| Load-line margin            | %    | 18 / 25   | 21 / 15                                                                                                                 | 22 / 18    | 20 / 15                       | 14 / 14  | 21/17    | 22 / 17                                                                                                     | 13 / 13    | 13 / 13    | 15 / 7     |
| Area quad. ins. cable HTS   | mm2  | 3241      | 1494                                                                                                                    | 2091       | 1527                          | 4490     | 1360     | 1500                                                                                                        | 1290       | 1154       | 1012       |
| Area quad. ins. cable LTS   | mm2  | 2150      | 6106                                                                                                                    | 3780       | 5148                          | 4915     | 4740     | 4640                                                                                                        | 2326       | 2558       | 4191       |
| Coil width*                 | mm   | 105       | 129                                                                                                                     | 144        | 149                           | 135      | 80       | 112                                                                                                         | 70         | 104        | 106        |
| Coil inner radius*          | mm   | 25        | 25                                                                                                                      | 30         | 30                            | 30       | 35       | 35                                                                                                          | 25         | 25         | 25         |

\* (Inner radius innermost cable on the mid-plane) - (Outer radius of outermost cable on the mid-plane)

\*\* Inner radius innermost cable on the mid-plane

Case II. Also, when tested in stand-alone, one can notice the extremely high stress in the large aperture Nb3Sn coils.

Finally, the stored energy for the 20 T hybrids and the standalone cases are provided in Table III: also for these parameters, the most critical condition also appearsto be the one of the large aperture Nb3Sn coil, with a stored energy similar to the 20 T hybrid case.

#### V. MAGNETIC DESIGN

For a preliminary magnetic analysis of the 20 T hybrid magnet, we investigated different coil design options, all shown in Fig. 6. In particular, the following lay-outs were considered: traditional Cos-theta (CT) design, Stress Management Cos (SMCT) design, Canted Cos-theta (CCT) design, Block (BL) design, and Common-Coil (CC) design. The main parameters are given in Table IV. The cables considered range from 9.1 to 21.6 mm width and from 1.5 to 2.1 mm thickness (including insulation). All the designs implement HTS Bi2212 Rutherford cables; REBCO CORC wire, with an assumed 7.5 x 7.5 mm<sup>2</sup> of dimension with insulation, was considered only for the common-coil, given the large bending radius of the latter. With a 20 T bore field, the operational current varies from 10.7 to 17.8 kA, and the peak field in the HTS and LTS coils is respectively 20.2-21.0 T and 12.7-17.0 T. The target load-line margin is achieved in the CT, SMCT, and BL designs, while in the others the margin is a few percentage points below target. In terms of field quality, design criteria are met by all the designs except the BL, which features geometric harmonics up to the 10 units level.

Finally, it is important to point out that only a preliminary investigation of the accumulated electro-magnetic (e.m.) forces in some of the designs was carried out, and a complete mechanical analysis aimed at bringing the stress in the HTS and LTS below the limits fixed in Table II has not been performed yet. Therefore, the designs depicted in Fig. 6 represent only a first iteration and a starting point of the design, and, since they meet only part of the criteria, they are not yet comparable. In the next sub-sections, we formulate some initial considerations for each design.

# *A. Cos-theta (CT), Stress Management Cos-theta (SMCT), and Canted Cos-theta (CCT) Designs*

For the CT options we considered a design with double layers coils, each wound with the same cable. This design choice avoids interlayer splices and has been implemented in most of the Nb3Sn CT coils fabricated so far (the only exception being the CERN-ELIN and UT-CERN dipole magnets [9]). The first design (top CT in Fig. 6) has 4 Bi2212 layers and 2 Nb3Sn layers, with a peak field in the Nb3Sn of 12.7 T and a total coil width of 105 mm. As we did for the sector coils, in a second design (bottom CT in Fig. 6) we reduced the Bi2212 layers from 4 to 2 by increasing the width of the Nb3Sn coils. As a result, the peak field in the Nb3Sn rose to 16.1 T and the total coil width rose to 129 mm. For a detailed description of these options we refer to [36], whereas the description of 4-layer CT option for high-field Nb3Sn magnet can be found in [37]. As shown by the sector coil analysis, a traditional CT design magnet aiming at the 20 T level is characterized by high coil stress both in the azimuthal and in the radial direction. A possible alternative solution is represented by the SMCT, where each layer is separated by 5 mm thick spars (or mandrels) and each cable block is separated by ribs, connected to the mandrel [38]- [40]. The implementation of stress intercepting elements results in an overall increase of coil width from 102-129 mm in the CT to 144-149 mm, and in the conductor area. A further step towards the reduction of the stress is done with the CCT design, where each turn is separated by spars and ribs [41]-[43]. The field quality is naturally achieved by superimposing the two tilted solenoids (see Fig. 6 center). For the 20 T hybrid we chose a simple design with 4 Bi2212 layers and 2 Nb3Sn layers, all wound with a MQXF cable. The total area of the insulated cable (taken from a simple cross-section of the 3D design) is, as expected, larger than in the previous CT and SMCT designs. However, since the layer-to-layer splices are located in the coil ends, a full grading coil, with cables progressively smaller from the inserts to the outsert, can reduce significantly the coil size, and it will be the goal of the next step in the optimization.

## *B. Block (BL) Design*

The Block design [44]-[46] allows for a very efficient subdivision between the HTS and LTS coils, since the cables are aligned with the flux lines. Therefore, the area of Bi2212 in the block design shown in Fig 6 (BL top design), is smaller than for the CT, SMCT and CCT options (see Table IV). Also, in terms of total conductor area the design is very compacted, despite the inclusion of a 10 mm thick internal support in the inner coil that increases the coil aperture to 70 mm (similarly to the coil design of FRESCA2 [47] and TDF [48] magnets). However, as shown in [49], the peak stress in the coil, in particular because of the horizontal e.m. forces, can be as high as 280 MPa in the Nb3Sn and 160 MPa in the Bi2212 at 20 T. Therefore, an alternative has been considered where vertical and horizontal plates are included to intercept part of the e.m. forces. In the bottom BL design in Fig. 6, intercepting plates separates Bi2212 and Nb3Sn coils: consequently, the coil increases in size, but the stress in the Nb3Sn and in the Bi2212 coils decreases to about 160 MPa and 140 MPa respectively at 20 T. For a complete description of the two designs, and a discussion about fabrication issues and stress management options, we refer to [49]. Further design work on the BL design will be aimed at reducing the coil stress and improve the field quality.

#### *C. Common-coil design*

On the right side of Fig. 6, 3 different common coil designs for the 20 T hybrid are shown. The common-coil design [50]- [52] is based on racetrack coils that, with a very large bending radius in the ends, cover both magnet apertures (in Fig. 6 only one aperture is shown). The large bending radius opens the possibility not only of implementing the *react-and-wind* technique, but also to utilize REBCO CORC cable, whose rigidity makes small bending radius a possible source of conductor degradation. Similar to the block design described above, the commoncoil allows aligning the block with the flux lines, thus minimizing the HTS conductor use. Also, by having the layer-to-layer splice inside the winding pole at the center of the coil, one can wind and react individual layers, and "grade" each layer to maximize efficiency. The two top designs in Fig. 6 uses Bi2212 cables in the small blocks around the aperture and in the first layer, followed by either 3 or 5 layers of Nb3Sn cables. In both designs, the coil area and width are small compared to the previous designs; however, it is important to point out that the loadline margins are below the 15% criteria. In the third design, we implement a large CORC wire, in series with 4 layers of HTS. Also in this case further magnetic analysis will be carried out to bring the load-line margin to the design criteria. Regarding the coil stress, the common coil design allows the insertion of vertical plates to intercept the horizontal e.m. forces, and similarly to the BL design, horizontal bars can be considered to intercept the vertical force. As a next step, a mechanical analysis will be performed to verify the stress, and the magnetic design will be updated accordingly.

#### VI. CONCLUSIONS

We presented in this paper a preliminary investigation of a hybrid 20 T dipole, which we consider a promising option for a dipole magnet operating beyond the limits of Nb3Sn and aimed at minimizing the HTS volume. Two HTS conductors are considered: Bi2212, in the form of a Rutherford cable with *j<sup>e</sup>* of 740 A/mm<sup>2</sup> at 20 T, and REBCO tape in a CORC/STAR wire with *j<sup>e</sup>* of 590 A/mm<sup>2</sup> at 20 T. As part of the design criteria, we target a bore field of 20 T with a load-line margin of at least 15% for both LTS and HTS coils. Also, all the coils shall be powered in series, and stress must be kept below 150-180 MPa in the Nb3Sn and below 120 MPa in the HTS. A preliminary analysis done with sector coils indicated that 1) with identical *jo* in both HTS and LTS, we have a coil width of ~70 mm, 2) radial stresses of about 200 MPa are generated by the radial/horizontal e.m. forces, and 3) a significant reduction of HTS area can be obtained by "anti-grading", i.e. by increasing the size of the Nb3Sn outsert. Finally, we performed a preliminary analysis of a 20 T hybrid with different coil design options, all shown in Fig. 6. The designs are not comparable yet since they do not meet all the specifications, but they provide a first idea of the overall coil features, and they constitute a starting point for further analysis. The next step will include a mechanical analysis, and the continuation of the magnetic analysis, with the goal of meeting margin, field quality, and stress criteria in all the designs.

#### REFERENCES

- [1] [https://www.usparticlephysics.org.](https://www.usparticlephysics.org/)
- [2] K. Koepke, *et al*., "Fermilab Doubler Magnet Design and Fabrication Techniques", *IEEE Trans. Magnetics*, Vol MAG-15, No. 1, January 1979.
- [3] S. Wolff, "Superconducting HERA Magnets, *IEEE Trans. Magnetics*, Vol 24, No. 2, March 1988.
- [4] A. Greene, et al., "The Magnet System of the Relativistic Heavy Ion Collider (RHIC)", *IEEE Trans. Magnetics*, Vol 32, No. 4, July 1996.
- [5] L. Rossi, "Superconducting magnets for the LHC main lattice", *IEEE Trans. Appl. Supercond*., vol. 14, no. 2, June 2004, pp. 153.
- [6] "The High Luminosity Large Hadron Collider", edited by O. Brüning and L. Rossi, World Scientific, October 2015.
- [7] D. Schoerling, *et al.,* "The 16 T Dipole Development Program for FCC and HE-LHC", *IEEE Trans. Appl. Supercond*., vol. 29, no. 5, August 2019, Art. no. 4003109.
- [8] [https://usmdp.lbl.gov.](https://usmdp.lbl.gov/)
- [9] "Nb3Sn Accelerator Magnets", edited by D. Schoerling and A. Zlobin, Springer Open, 2019.
- [10] Larbalestier, D. C., et al., "Isotropic Round-Wire Multifilament Cuprate Superconductor for Generation of Magnetic Fields above 30 T", *Nature Materials* 13 (4): 375–81.
- [11] Selvamanickam, V., "High temperature superconductor (HTS) wires and tapes in High Temperature Superconductors (HTS) for Energy Applications", Melhem, Z., Ed.; Woodhead Publishing Series in Energy; Woodhead Publishing: Cambridge, UK, 2012; pp. 34–68.
- [12] P. McIntyre and A. Sattarov, "On the Feasibility of a Tripler Upgrade for the LHC", PAC (2005) 634.
- [13] L. Rossi and E. Todesco, "Conceptual design of 20 T dipoles for highenergy LHC," CERN, Geneva, Switzerland, CERN Yellow Rep. 2011-3, pp. 13–19, 2011.
- [14] E. Todesco, *et al*., "Dipoles for High-Energy LHC", *IEEE Trans. Appl. Supercond*., vol. 24, no. 3, June 2014, Art. no. 4004306.
- [15] Qingjin Xu, *et al.*, "20-T Dipole Magnet with Common-Coil Configuration: Main Characteristics and Challenges", *IEEE Trans. Appl. Supercond*., vol. 26, no. 4, June 2016, Art. no. 4000404.
- [16] J. van Nugteren, et al., "Toward REBCO 20 T+ Dipoles for Accelerators", *IEEE Trans. Appl. Supercond*., vol. 28, no. 4, June 2018, Art. no. 4008509.
- [17] [https://indico.ihep.ac.cn/event/4900.](https://indico.ihep.ac.cn/event/4900)
- [18] D. Martins Araujo, *et al.* "Preliminary Integration for Testing HTS Feather-M2 in the FRESCA2 Dipole Magnet", *IEEE Trans. Appl. Supercond*., vol. 30, no. 4, June 2020, Art. no. 4003605.
- [19] M. Durante, *et al.* "Manufacturing of the EuCARD2 Roebel-Based Cos-Theta Coils at CEA Saclay", *IEEE Trans. Appl. Supercond*., vol. 30, no. 4, June 2020, Art. no. 4602505.
- [20] L. Rossi, *et al.* "The EuCARD2 Future Magnets Program for Particle Accelerator High-Field Dipoles: Review of Results and Next Steps", *IEEE Trans. Appl. Supercond*., vol. 28, no. 3, April 2018, Art. no. 4001810.
- [21] P. Ferracin, "LARP Nb3Sn Quadrupole Magnets for the LHC Luminosity Upgrade", in *AIP Conf. Proc*. 1218, 1291-1300 (2010).
- [22] P. Ferracin, et al., "The HL-LHC low-b quadrupole magnet MQXF: from short model to long prototype", *IEEE Trans. Appl. Supercond.* vol. 29, no. 5, August 2019, Art. no. 4001309.
- [23] Tengming Shen, Laura Garcia Fajardo, "Superconducting accelerator magnets based on high temperature superconducting Bi-2212 round wires", *Instruments,* 2020, 4(2), 17.
- [24] J. D. Weiss, *et at.,* "Introduction of the next generation of CORC® wires with engineering current density exceeding 650 A mm−2 at 12 T based on SuperPower's ReBCO tapes containing substrates of 25 μm thickness", 2020 *Supercond. Sci. Technol.* 33 044001.
- [25] X. Wang, et al., "A 1.2 T canted cos θ dipole magnet using high-temperature superconducting CORC® wires", *Supercond. Sci. Technol*. 32 (2019) 075002.
- [26] S. Kar, *et al.,* "Next-generation highly flexible round REBCO STAR wires with over 580 A mm−2 at 4.2 K, 20 T for future compact magnets", 2019 *Supercond. Sci. Technol*. 32 10LT01.
- [27] G. Ambrosio, "Maximum allowable temperature during quench in Nb3Sn accelerator magnets", CERN Yellow Report, CERN-2013-006, p.43.
- [28] L. Ye, P. Li, T. Shen and J. Schwartz, "Quench degradation limit of multifilamentary Ag/Bi2Sr2CaCu2O x round wires", 2016 *Supercond. Sci. Technol*. 29 035010.
- [29] A. Ishiyama, *et al*., "Degradation of YBCO Coated Conductors Due to Over-Current Pulse.", *IEEE Trans. Appl. Supercond* , vol 17, no 2, 2007, pp. 3509–3512.
- [30] X. Wang, *et al*., "Critical Current Degradation of Short YBa2Cu3O7- \delta Coated Conductor due to an Unprotected Quench." 2010 *Supercond. Sci. Technol.* 24 (3) 035006.
- [31] P. Ebermann, *et al.,* "Irreversible degradation of Nb3Sn Rutherford cables due to transverse compressive stress at room temperature", 2018 *Supercond. Sci. Technol*. 31 065009
- [32] D. R. Dietderich, *et al.,* "Critical current variation of Rutherford cable of Bi-2212 in high magnetic fields with transverse stress", *Physica C,* Vol. 341–348, Part 4, November 2000, Pages 2599-2600.
- [33] D.C. van der Laan, *et al.,* "Effect of Transverse Compressive Monotonic and Cyclic Loading on the Performance of Superconducting CORC® Cables and Wires." 2018 2018 *Supercond. Sci. Technol* 32 (1): 015002.
- [34] L. Rossi, E. Todesco, "Electromagnetic design of superconducting quadrupoles", *Phys. Rev. ST Accel. Beams* 10 (2007) 112401.
- [35] L. Rossi and Ezio Todesco, "Electromagnetic design of superconducting dipoles based on sector coils", *Phys. Rev. ST Accel. Beams* 9 (2006) 102401.
- [36] V. Marinozzi, *et al*., "Conceptual design optimization of a 20 T hybrid cos-theta dipole superconducting magnet for future High-Energy particle accelerators", *IEEE Trans. Appl. Supercond*., submitted for publication.
- [37] R. Valente, *et al.,* "Baseline Design of a 16 T cos θ Bending Dipole for the Future Circular Collider", *IEEE Trans. Appl. Supercond*., Vol. 29, No. 5, August 2019, Art. no. 4003005.
- [38] A. Zlobin, et al., "Large-Aperture High-Field Nb3Sn Dipole magnets", Proceedings of IPAC2018, Vancouver, BC, Canada, WEPML026.
- [39] I. Novitski, *et al*., "Development of a 120-mm aperture Nb3Sn dipole coil with stress management", *IEEE Trans. Appl. Supercond*., submitted for publication.
- [40] A. Patoux, J. Perot, J. M. Rifflet, "Test of New Accelerator Superconducting Dipoles Suitable for High Precision Field", *IEEE Trans. on Nuclear Science*, Vol. 30, No. 4, Aug. 1983.
- [41] S. Caspi, *et al.,* "Design of a Canted-Cosine-Theta Superconducting Dipole Magnet for Future Colliders", *IEEE Trans. Appl. Supercond*., Vol. 27, No. 4, June 2017, Art. no. 4001505.
- [42] B. Auchmann, *et al., "*Electromechanical Design of a 16-T CCT Twin-Aperture Dipole for FCC", *IEEE Trans. Appl. Supercond*., Vol. 28, No. 3, April 2018, Art. no. 4000705.
- [43] L. Brouwer, *et al.*, "Design of CCT6: a large-aperture, 12 T, Nb3Sn Dipole Magnet", *IEEE Trans. Appl. Supercond*., submitted for publication.
- [44] G. Sabbi, et al., "Design Study of a 16-T Block Dipole for FCC", IEEE Trans. Appl. Supercond., Vol. 26, No. 3, April 2016, 4004705.
- [45] M. Segreti et al., "2-D and 3-D design of the bock-coil dipole option for the future circular collider," *IEEE Trans. Appl. Supercond*., vol. 29, no. 5, Aug. 2019, Art. no. 4000404.
- [46] E. Rochepault, *et al*., "The Use of Grading in Nb3Sn High-Field Block-Coil Dipoles", *IEEE Trans. Appl. Supercond*., Vol. 31, No. 4, June 2021, Art. no. 4001510.
- [47] P. Ferracin, *et al.*, "Development of the EuCARD Nb3Sn Dipole Magnet FRESCA2", *IEEE Trans. Appl. Supercond*., Vol. 23, No. 3, June 2013, Art. no. 4002005.
- [48] G. Vallone, *et al.*, "Magnetic and Mechanical Analysis of a Large Aperture 15 T Cable Test Facility Dipole Magnet", *IEEE Trans. Appl. Supercond*., Vol. 31, No. 5, August 2021, Art. no. 9500406
- [49] E. Rochepault, *et al*., "20 T Hybrid Nb3Sn-HTS Block-coil Designs for a Future Particle Collider", *IEEE Trans. Appl. Supercond*., submitted for publication.
- [50] R. Gupta, *et al.,* "Common Coil Dipoles for Future High Energy Colliders", *IEEE Trans. Appl. Supercond*., Vol. 27, No. 4, June 2017, Art. no. 4000605.
- [51] F. Toral, *et al.*, "Magnetic and Mechanical Design of a 16 T Common Coil Dipole for an FCC", *IEEE Trans. Appl. Supercond*., Vol. 28, No. 3, April 2018, Art. no. 4004305.
- [52] E. Ravaioli and G. Sabbi, "Design of a Compact 16 T Common-Coil Dipole for Future Colliders", *IEEE Trans. Appl. Supercond*., Vol. 28, No. 4, June 2018, Art. no. 4008005.