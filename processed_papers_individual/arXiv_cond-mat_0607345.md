# arXiv:cond-mat_0607345

**Paper ID:** bb19f50aca110e187b8e2627bc945625

**PDF Path:** apl/Superconductors/arXiv:cond-mat_0607345.pdf

**Processing Status:** complete

**Captions Added:** 32

**Generated:** 2025-06-24T13:44:28.217089

---

## **Title**

**Transverse Load Optimisation in Nb3Sn CICC Design; Influence of Cabling, Void Fraction and Strand Stiffness**

## **Authors**

A. Nijhuis and Y. Ilyin

## **Authors address**

University of Twente, Faculty of Science and Technology, Low Temperature Division, P.O. Box 217, 7500 AE Enschede, The Netherlands

## **Short title**

Transverse load optimisation in Nb3Sn CICCs

# **Abstract**

We have developed a model that describes the transverse load degradation in Nb3Sn CICCs, based on strand and cable properties, and that is capable to predict how such degradation can be prevented.

The Nb3Sn Cable In Conduit Conductors (CICC) for the International Thermonuclear Experimental Reactor (ITER) show a significant degradation in their performance with increasing electromagnetic load. Not only the differences in the thermal contraction of the composite materials affect the critical current and temperature margin, but mostly electromagnetic forces, cause significant transverse strand contact and bending strain in the Nb3Sn layers.

Here, we present the model for Transverse Electro-Magnetic Load Optimisation (TEMLOP) and report the first results of computations for ITER type of conductors, based on the measured properties of the internal tin strand used for the Toroidal Field Model Coil (TFMC). As input, the model uses data describing the behaviour of single strands under periodic bending and contact loads, measured with the TARSIS setup, enabling a discrimination in performance reduction per specific load and strand type.

The most important conclusion of the model computations is that the problem of the severe degradation of large CICCs can be drastically and straightforwardly improved by increasing the pitch length of subsequent cabling stages. It is for the first time that an increase of the pitches is proposed and no experimental data are available yet to confirm this beneficial outcome of the TEMLOP model. Larger pitch lengths will result in a more homogeneous distribution of the stresses and strains in the cable by significantly moderating the local peak stresses associated with the intermediate-length twist pitches. The twist pitch scheme of the present conductor layout turns out to be unfortunately close to a worst-case scenario.

The model also makes clear that strand bending is the dominant mechanism causing degradation. The transverse load on strand crossings and line contacts, abbreviated as contact load, can reach locally 90 MPa but this occurs in the low field area of the conductor and does not play a significant role in the observed critical current degradation. The model gives an accurate description for the mechanical response of the strands to a transverse load, from layer to layer in the cable, in agreement with mechanical experiments performed on cables.

It is possible to improve the ITER conductor design or the operation margin, by mainly a change in the cabling scheme. We also find that a lower cable void fraction and larger strand stiffness add to a further improvement of the conductor performance.

## **1. Introduction**

The envisaged CICCs for the International Thermonuclear Experimental Reactor consist out of more than 1000 strands with a strand diameter of about 0.8 mm. The conductors have a void fraction (*vf*) of 33 % and are cabled by twisting in several stages, thus creating a wavy pattern of the strands throughout the cable. When cooling from reaction heat treatment to cryogenic operation temperature, the different thermal contraction between conduit material and strand bundle causes besides a contraction of the strands in axial direction, likely also bending of the strands. This bending strain is again moderated by the coil hoop stress during magnet operation. The conductors are carrying more than 50 kA in a magnetic field locally exceeding 13 T, hence subjecting them to severe transverse loading due to the Lorentz forces. This imposes distributed magnetic loads along each strand, but also cumulated loads from other strands transferred by the strand-tostrand contacts. These bending and contact loads on the Nb3Sn strands affect the critical current and creates a

periodic strain variation along the filaments. The magnitude and periodicity of the periodic strain pattern in combination with the ability of a strand to redistribute the current between the filaments determines the impact on the critical current (*I*c) and *n*-value[1,2].

Since the first tests of the Central Solenoid Model Coils (CSMC) in Japan, lots of effort has been spent to the understanding of the unexpected severe degradation of the conductors compared to that of the single strand performance [3-8]. Although many papers were published on this subject, they mostly jus describe the degradation giving possible explanations varying from sometimes-severe current non-uniformity to oftensevere strand bending. Most of the explanations are supported by models but unfortunately, none of these models is able to provide an adequate solution to the problem although it was shown that a lower void fraction and shorter cabling pitches partly confines the degradation [3]. In the meantime, it proven experimentally that mechanical support of the strands improved the performance significantly [9] and the ITER conductor design has already been modified compared to the CSMC layout [10]. It is assumed that the resistance to the degradation is increased by using a steel jacket, providing thermal pre-compression in reducing the tensile strain levels being associated with strand bending [3,11]. The void fraction was reduced from 36 % to 33 % and the non-copper material in the cross section was increased by 25%. In addition, newly developed, so-called high *J*<sup>c</sup> advanced Nb3Sn strands are being pursued in attempt to compensate for the performance loss.

Nevertheless, today it seemed accepted that the transverse load inherently leads to a severe degradation in the transport properties due to strand deformation. Although not conclusively proven, some of the analysts in the fusion community suggest that about 30 % to 40 % of the Nb3Sn layer in the strand material of the cable is breaking [3,10]. This would implicitly suggest that the cable in conduit concept for the large ITER type of conductors is beyond the limits of efficient application. In view of this thought, the development of advanced high *J*<sup>c</sup> strands would not immidately lead to an appropriate solution either, if a significant fraction of the Nb3Sn material then breaks in the cable. It is evident that a method to solve this severe degradation would lead to significant cost savings when requiring less superconducting strand or a significant improvement of the ITER magnets performance in terms of operation margin.

We believe that we have revealed a straightforward and economical method to overcome a significant part of this excessive reduction in performance. A newly developed model describing the mechanical response of the individual strands within a cable bundle, clearly shows that applying longer cabling pitches provides a convenient solution against transverse load degradation. In all probability, insufficient practical experience with varying cabling pitches, explains why this factor remained hidden until now. Moreover, it was assumed that only shorter cabling pitches could lead to a better performance obviously by shortening the bending beam [3], which is indeed confirmed by our model but for various reasons can be considered an un-practical solution. The model calculations on the present cable design are in good agreement with the deformation experiments that were performed on full-size ITER conductors in the Twente Cryogenic Press, as well as with the *I*<sup>c</sup> degradation observed in the CSMC tests [5].

An first essential parameter (also deduced from the press measurements) turns out to be the maximum possible compression of the cable, i.e. the available space for bending and contact deformation of strands. In addition, the strand mechanical properties determined from tests with the "Test ARangement for Strain Influence on Strands" (TARSIS) form the second key input for the TEMLOP-Model and are briefly reviewed in section 2 of this paper. In section 3, we give a full description of the mechanical model and identify the required parameters to describe the cable layout adequately. Then, in section 5, we present the results of calculations with parametric variations in cabling pattern, void fraction and strand stiffness. These results are discussed in section 6, leading to a number of detailed recommendations. The symbols used in the equations are listed i[n Table I.](#page-1-0)

Table I. List of symbols.

<span id="page-1-0"></span>

| A [m2<br>]         | strand cross section area                                      |
|--------------------|----------------------------------------------------------------|
| Ac [m2<br>]        | projected strand to strand contact area                        |
| Ac⊥ [m2<br>]       | minimum projected contact area for a strand crossing           |
| B [T]              | magnetic field                                                 |
| cos-1<br>θ         | cos<br>factor representing the actual cable cross section<br>θ |
| D [m]              | cable width                                                    |
| df [m]             | diameter filamentary region                                    |
| ds [m]             | diameter wire                                                  |
| E⊥<br>or Etr [GPa] | Young's modulus of the strand, E (transverse)                  |

| E  <br>[GPa]            | Young's modulus of the strand, E (axial)                             |  |
|-------------------------|----------------------------------------------------------------------|--|
| Fc [N]                  | transverse contact load                                              |  |
| fcbm [m]                | maximum possible average deflection of the cable                     |  |
| fcbm [m]                | maximum possible deflection available for bending                    |  |
| Fmax [N]                | transverse peak load                                                 |  |
| fmc [m]                 | maximum cable-compression at Fmax                                    |  |
| Fn [N]                  | transverse load with n indicating the order (n=1 for Lw1)            |  |
| fsb [m]                 | strand deflection from bending, bending amplitude                    |  |
| fsb,n [m]               | strand deflection from bending, n indicating the order (n=1 for Lw1) |  |
| fsbm [m]                | maximum bending amplitude                                            |  |
| fsc [m]                 | strand deformation per strand crossing and line contact              |  |
| fsm [m]                 | maximum possible average deflection per strand                       |  |
| Ia [m4<br>]             | moment of inertia                                                    |  |
| Ic [A]                  | critical current                                                     |  |
| Ic0 [A]                 | critical current before loading, virgin state                        |  |
| Is [A]                  | strand current                                                       |  |
| Jc [A/m2<br>]           | critical current density                                             |  |
| kθ                      | cos-1<br>(<br>) correction factor<br>θ                               |  |
| Lw [m]                  | characteristic bending wavelength                                    |  |
| Lw,n [m]                | characteristic wavelength with n indicating the order (n=1 for Lw1)  |  |
| Lφ [m]                  | is the characteristic length, 4.3⋅10-3 m                             |  |
| Nl                      | number of strand layers                                              |  |
| Ns                      | number of strands in the cable                                       |  |
| n-value                 | value characterizing the steepness of the V-I curve [2]              |  |
| V [V]                   | voltage                                                              |  |
| rho [Ωm]                | matrix interfilament resistivity                                     |  |
| vf                      | cable void fraction                                                  |  |
| Wb [m3<br>]             | 3<br>·π/32<br>section factor written as ds                           |  |
| max [Pa]<br>Β           | maximum magnetic field in the cable                                  |  |
| min [Pa]<br>Β           | minimum magnetic field in the cable                                  |  |
| ε                       | axial strain in the Nb3Sn filaments                                  |  |
| ε<br>b                  | bending strain in the Nb3Sn filaments                                |  |
| bo, e peak-fil reg<br>ε | peak bending strain in the Nb3Sn filaments                           |  |
| εth                     | thermal pre-compression of the Nb3Sn filaments                       |  |
| ϕ                       | angle between crossing strands                                       |  |
| θ                       | angle between strand and conductor longitudinal axis                 |  |
| [Pa]<br>σ               | contact stress                                                       |  |
| max [Pa]<br>σ           | maximum contact stress in the cable                                  |  |
| min [Pa]<br>σ           | minimum contact stress in the cable                                  |  |

#### **2. Strand data from TARSIS**

In the TARSIS setup, the influence of various loads and deformations (axially tensile, bending strain, contact load from crossing strands or homogeneous transverse load) that frequently occur in a CICC, are studied with different probes [2,12-21]. Over the last few years, there is an increasing interest to study the impact of mainly bending strain on the transport properties of superconducting wires [7,22,23]. The crucial advantage of the TARSIS setup is its ability to measure both the amplitude of deflection or deformation and the applied force with high precision, enabling a full axial and transverse stress-strain analysis.

Lately we presented the results of the probe for periodical strand bending with cyclic loading using different bending wavelengths [21]. In addition, we completed probes for the characterisation of axial tensile stressstrain behaviour of strands [18] and sub-size cables [19]. Since it is the stress distribution, - originating from differential thermal contraction and from the electromagnetic load - that drives the final strain distribution, the stiffness of the strand and cable are definitely key parameters in this study.

Recently, we completed a probe to study also the influence of transverse loading on perpendicularly crossing and pinching strands [17]. While bending is broadly studied, the impact of local transverse loads at strand

crossings has hardly received any attention, even though these stresses can be quite severe in the part of the cable cross-section where the load accumulates.

The design of the TARSIS setup is presented in [12] while various experimental results obtained on powder in tube-, internal tin- and bronze-processed Nb3Sn wires are reported in [13-21].

### **3. Mechanical model for transverse load**

## **3.1. General model assumptions**

The TEMLOP-Model describes mechanical strand interactions within the cable, including strand bending, strand crossing and line contacts under the influence of an electromagnetic load. The plastic deformation of strands is not separately distinguished although the overall quantitative analysis is improved by accounting for yielding in both, transverse and axial direction, at higher stress levels. This is incorporated in the description for the axial Young's modulus [18]. Thus, a way is found to optimise the quantitative accuracy for the outcome of the model. We do not take into account axial strain variations, other then already included in the bending strain model, so a possible influence of temperature variations and conduit material choice with different thermal expansion is not studied here. The main input for the equations in the TEMLOP-Model is listed in [Table II.](#page-3-0) The input parameters are based on the design layout of the CS1 type of conductor from the CSMC [8], while the strand properties are used from the TFMC strand manufactured by EM (Italy), presently OKCI [20,24].

In the model, the shape of the cable cross section is simply taken square instead of round and no central channel is present. Although it is not complicated to account for the circular cross-section of the actual cable in the numerical model, preliminary estimates showed that the influence on the outcome is not significant. Omitting the central channel from the model corresponds to considering it as a fully stiff medium that transfers the reaction force and displacement without any intrusion.

<span id="page-3-0"></span>

| Table II. Input parameters for the TEMLOP-Model.         |                   |         |  |  |
|----------------------------------------------------------|-------------------|---------|--|--|
| Input parameters<br>strand                               |                   |         |  |  |
| Diameter wire, ds                                        | 810               | µm      |  |  |
| Diameter filamentary region, df                          | 660               | µm      |  |  |
| Bending wavelength, Lw                                   | 6                 | mm      |  |  |
| E modulus strand (axial), E                              | 29                | GPa     |  |  |
| E modulus strand (transverse), E⊥<br>or Etr              | 3.3               | GPa     |  |  |
| Input parameters<br>cable                                |                   |         |  |  |
| Strands, Ns                                              | 1152              | strands |  |  |
| Layers, Nl                                               | 34                | layers  |  |  |
| cable void fraction, vf                                  | 0.36              |         |  |  |
| factor (cos-1<br>Cos<br>)<br>θ<br>θ                      | 1.05              |         |  |  |
| Cabling scheme CS1 Model Coil conductor                  | 3x4x4x4x6         |         |  |  |
| Cabling pitches CS1 Model Coil conductor                 | 45x74x123x160x380 | mm      |  |  |
| Cable-compression @ Fmax<br>measured in Cable Press, fmc | 1.134             | mm      |  |  |
| magnetic field, B                                        | 12                | T       |  |  |

### **3.2. Principle of the mechanical model for load distribution**

The local transverse deformation of the strands in a CICC is composed of a distribution of bending, contact and homogeneously distributed loads. The local spatial periodic bending, causing a wave-pattern [2,12,13] can actually be described by a periodic repetition of a point load with clamped ends, schematically shown in [Figure 1.](#page-3-1)

![](_page_3_Figure_10.jpeg)

**Caption:** Schematic representation illustrating the application of forces (F) on a strand bent over multiple supports. The drawing demonstrates the conceptual setup for studying bending impacts, showing forces applied at intervals along a strand resting on cylindrical supports spaced by distance L, with deflection f.


<span id="page-3-1"></span>*Figure 1. Schematic view of a point load with clamped ends.*

The corresponding bending beam equations are:

$$f\_{sb} = \frac{F \cdot L\_{\text{w}}^3}{192 \cdot E\_{\text{ll}} \cdot I\_a} \qquad \text{[m]} \tag{1}$$

$$
\mathcal{E}\_b = \frac{F \cdot L\_w}{8 \cdot E\_\vee \cdot W\_b} \qquad \qquad \text{[-]} \tag{2}
$$

in which *fsb* is the strand deflection, *L*<sup>w</sup> is the bending wavelength [\(Figure 1\)](#page-3-1), *E//* is the Young's modulus in axial direction, *I*<sup>a</sup> is the moment of inertia while *W*<sup>b</sup> is the section factor written as *ds* 3 ·π/32 with *ds* is the strand diameter. The load *F* represents the local accumulated magnetic load on a strand in a given layer of the CICC.

The basic model for the strand mechanical interaction between the strands is schematically represented in [Figure 2](#page-4-0) where three layers of strands (A,B,C) are depicted to illustrate the deformation, for the load in three steps for increasing load from [Figure 2a](#page-4-0) to c.

![](_page_4_Figure_6.jpeg)

**Caption:** This diagram provides a schematic representation of the layered bending configurations in a superconducting cable with different wavelength parameters. It outlines the layer interactions and conceptualizes the bending wave mechanics underpinning the cable's structural integrity and response to loads, which is crucial for understanding deformation mechanisms .


<span id="page-4-0"></span>*Figure 2. Schematic model for strand mechanical interaction in the cable bundle from layer to layer with increasing electromagnetic force pointing from top to bottom.*

The first case [\(Figure 2a](#page-4-0)) corresponds to the virgin situation before electromagnetic loading of the cable bundle. The second scheme in [Figure 2b](#page-4-0) shows the situation when the deflection of strand 1 in layer B, characterised by the bending amplitude *f*sb1, has virtually reached its maximum value and strand 1 is about to touch the surface of strand 2 in the layer C below. This situation constitutes a limiting case for bending with the wavelength characterised by *L*w1. The strand (1) with the wavy pattern is still only supported by the crossing strands associated with the wavelength indicated with *L*w1, i.e. the transverse contact load on the strand crossings remains concentrated on the same number of strand crossings that were already present from zero load, on a distance *L*w1. In this situation, the bending is fully described by the point load *F*1, the wavelength *L*w1 and the maximum bending amplitude (*f*sbm1). The transverse stress is transmitted through the contact points between strand 1 and the two supporting strands 3 and 4.

With further increasing the load [\(Figure 2b](#page-4-0)), strand 1 with the wavy pattern will make contact with strand 2. Arriving at this stage the periodicity in loading contacts increases by a factor two, while the maximum possible deflection (*f*sb2) becomes half the initial value and the load *F*<sup>2</sup> becomes half *F*1. Hence, we can distinguish for a decrease by a factor two of the characteristic bending wavelength from *L*w1 to *L*w2 and a doubling of the number of contact loads. The transverse load, initially a reaction force only supported by the contacts spaced by the original periodicity in crossing strands represented by *L*w1, becomes also partly distributed now by strand 2 from layer C below. The decrease in wavelength from *L*w1 to *L*w2 has two important effects: it restricts further deflection (see relations (1) and (2) and it causes a more homogeneous distribution of the contact stress and point loads.

When the load is further increased by other strands in layer A, (not indicated in the schedule) strand 1 will make contact with other strands from the same layer (C) as to which strand 2 belongs [\(Figure 2c](#page-4-0)). The result is a further successive decrease of the bending wavelength introducing now the more general description *L*w,n, the point load *F*<sup>n</sup> and the maximum possible deflection *f*sb,n. At the same time a progressive increase of the contact area occurs, which furthermore becomes more homogeneously distributed. Both factors reduce the local peak contact stress.

Now that the basic model for strand mechanical interaction has been introduced, we can derive the maximum bending amplitude and strand deformation at the crossovers.

The peak strain occurs at the strand crossings with the point loads. With reaching the maximum deflection and going from *L*w,n to *L*w,n+1, the peak strain at the initial point load *F*<sup>1</sup> will not increase further (may actually decrease) but new maxima are created at other locations corresponding with the position of the *F*<sup>2</sup> point loads. To obtain the strain value that limits the critical current in a strand, we take the peak strain along the strand.

### **3.3. Strand and cable deformation**

The accumulation of Lorentz load *Fn* on the strands in a CICC is further explained in [Figure 3](#page-5-0) where the electromagnetic *IxB* load increases progressively from the right to the left. At the same time, the local magnetic field decreases also in this direction, due to the combination of the applied field generated in the coil (or background field) and the cable self-field due to the transport current.

![](_page_5_Figure_6.jpeg)

**Caption:** Schematic diagram demonstrating the mechanical model for cable compression, showing potential for bending and deformation due to contact stress. The representation highlights structural arrangement considerations in cable design, critical for optimizing load distribution within strand arrangements .


Cross section Full-Size ITER CICC

<span id="page-5-0"></span>*Figure 3. The magnetic field profile in the cable cross section in relation to the accumulated load distribution.*

The locally occurring accumulated peak load in an ITER conductor with 1152 strands is about 20 kN/m, assuming that the number of strand layers can be approximated by the square root of the total number of strands (*Ns*) in a cable [2,6]:

$$F = I\_s \cdot B \cdot N\_s^{0.5} \tag{\text{[N/m]}},\tag{3}$$

As already indicated in [25], the most favourable transverse load distribution is obviously a homogeneous one, corresponding to a situation with infinite line contacts between parallel strands and avoiding local bending effects. In that case, the transverse peak stress, σmax, is the locally occurring accumulated peak load in an ITER conductor of 20 kN/m on the projected area of a strand cross section (*ds*).

$$
\sigma\_{\text{hom}} = \frac{I\_s \cdot B \cdot N\_s^{0.5}}{d\_s} \tag{\text{Pa}} \tag{4}
$$

The peak transverse stress in this homogeneous case, σhom, amounts to ∼20 MPa. However, as soon as there is locally sufficient space available for strand bending, the load at the crossovers rises and will exceed the value determined for a homogeneous distribution, since the projected contact area *A*<sup>c</sup> becomes smaller than in the case of an infinite line contact. The projected contact area between two crossing strands depends on the angle between the wires, ϕ, and on the strand diameter:

$$A\_c = \frac{d\_s^2}{\sin \varphi} \tag{5}$$

Correspondingly, the local transverse stress is multiplied with a contact area geometrical factor *k*:

$$k = \frac{d\_s \cdot L\_w}{A\_c} = \frac{L\_w \cdot \sin \phi \cdot}{d\_s} \qquad \qquad \text{[-]} \tag{6}$$

Combining (4), (5) and (6) we can write for the local peak stress:

$$
\sigma\_{\text{max}} = \frac{\sigma\_{\text{hom}} \cdot L\_w \cdot \sin \varphi}{d\_s} \tag{7}
$$

which can also be written as:

$$
\sigma\_{\text{max}} = \frac{I\_s \cdot \mathcal{B} \cdot N\_s^{0.5} \cdot L\_w \cdot \sin \varphi}{d\_s^2} \tag{Pa} \tag{8}
$$

Apart from the electromagnetic load and strand mechanical properties, the associated bending deflection is also determined by the distance between the supporting strands (relation 1). This wavelength is in turn determined by the cabling pattern of the strand bundle and is connected to the number of elements per subcable and the twist pitches and finally by the load determining *L*w(n).

The actual average wavelength (*Lw*) for crossing strands (and bending) is determined from a disassembled full-size ITER CS1 conductor and is 6 mm with a standard deviation of 2 mm, while the crossing angle amounts to 45 degrees. For a so-called SeCRETS-A sub-size conductor with a practically identical cabling scheme, also a wavelength of 6 mm was found [26]. A photograph of the CS1 cable bundle in [Figure 4](#page-6-0) shows clearly the deformation at strand crossings and varying periodicity.

A conductor with longer pitches, a CS2 type of conductor used for the outer module of the CSMC, is shown in [Figure 5.](#page-6-1) For the CS2 type of conductor, the average *L*<sup>w</sup> is difficult to determine as the positions of the crossings are less clear but it is defivitely longer than 6 mm while the crossing angle ϕ amounts to about 10 degrees.

![](_page_6_Picture_9.jpeg)

**Caption:** Photograph displaying a detailed view of a cable's structure, illustrating the arrangement of strands typical of an ITER CS1.1 Central Solenoid Model Coil conductor. The image highlights the strands' periodic crossing and interaction crucial for understanding structural integrity and electromagnetic performance.


*Figure 4. The cable structure of an ITER CS1.1 Central Solenoid Model Coil conductor.*

<span id="page-6-1"></span><span id="page-6-0"></span>![](_page_6_Picture_11.jpeg)

**Caption:** Image showing the cable structure of an ITER CS2 Central Solenoid Model Coil conductor, with emphasis on longer twist pitches and less frequent periodic strand crossings. This visual highlights the cable arrangement's structural factors essential for understanding mechanical and electromagnetic interactions in cable design.


*Figure 5. The cable structure of an ITER CS2 Central Solenoid Model Coil conductor with longer twist pitches.*

This implies that for a wavelength of 6 mm the accumulated load per strand crossing amounts to about 100 N. For a periodicity of *L*w=6 mm of the CS1 conductor and a strand diameter of 0.81 mm, the local peak stress can then reach a maximum level of 110 MPa.

Due to the cabling scheme in the ITER CICCs [\(Table II\)](#page-3-0) in several cabling stages with subsequent twist pitches it seems unavoidable to have local stress/strain concentrations at the crossover contacts and bending in between them. The bending deflection itself is restricted in absolute sense by the average free space available in the cable bundle [\(Figure 6\)](#page-7-0). When the cable bundle is represented by a square bundle of strands with *N*<sup>l</sup> layers of strands and each layer containing *N*<sup>l</sup> strands (in this case *N*l=34) we can derive the maximum possible free distance for a strand to move before encountering a neighbouring strand. The numbering of the strand layers *N*<sup>y</sup> goes from 1 to *N*l, starting at the high field (low *IxB*) side of the conductor.

![](_page_7_Figure_3.jpeg)

**Caption:** Schematic showing the cable cross-section configuration for strand packing used in simulation models. This is vital for understanding the spatial relationship within strands under mechanical and electromagnetic loads .


<span id="page-7-0"></span>*Figure 6. Schematic representation of the mechanical model for cable compression to derive the initial free deflection for bending per strand.*

For a given void fraction *vf*, number of strands *Ns* and strand diameter *ds*, the cable width *D* is calculated with:

$$D = \left(\frac{N\_s \cdot \pi \cdot d\_s^2}{4 \cdot \left(1 - \nu f\right) \cdot \cos\theta}\right)^{0.5} \qquad \text{[m]}\tag{9}$$

The maximum possible deflection available for bending for *N*l=(*N*s) 0.5=34 (*f*cbm) is derived from the cable width *D*, the strand diameter *ds* and the cos*(*θ*)* factor, correcting the strand cross-section area for the average angle between the twist pitches in the cabling with respect to the cable axis:

$$f\_{cbm} = D - N\_s^{0.5} \cdot d\_s \cdot \cos^{-0.5}(\theta) \tag{10}$$

The average cos*(*θ*)* factor from the CS1 type of conductor in [Figure 4,](#page-6-0) with angles for θ between roughly 15 and 20 degrees, is determined on 0.95 [27]. The average angle θ decreases with an increasing cabling pitch length, and is correlated to the cable void fraction. We assume that the cos*(*θ*)* correction factor rises by approximation linearly with the increase of the cable twist pitches and at the same time that the characteristic bending wavelength *L*<sup>w</sup> also increases linearly with the twist pitch. As we want to use the *L*<sup>w</sup> as a variable input parameter for the model we define the cos-1 *(*θ*)* correction factor *k*<sup>θ</sup> as:

$$\cos^{-1}(\theta) = k\_{\theta} = 1 + \frac{\lambda}{L\_{\text{w}}} \qquad \qquad \text{[-]} \tag{11}$$

with λ=3⋅10-3 m, empirically derived from the CS1 type of conductor, for *k*<sup>θ</sup> = 1.05 and *L*w=0.006 m. When we fix the void fraction as a conductor parameter, the space between the strands for maximum possible deflection (for bending) will change with varying wavelength, following relation 11.

The maximum possible average deflection per strand (*f*sm) is then calculated from the entire cable (*f*cbm) as:

$$f\_{sbm} = \frac{f\_{cbm}}{N\_s^{0.5}} \quad \text{ [m]} \tag{12}$$

For the calculation of the maximum free deflection per strand we take the two dimensional free space direction. If we allow for a 2D free space displacement, multiplication of the one-dimensional *f*sbm by a factor (2)0.5 is required. The real maximum possible deflection of the strands however, is expected less than for a

regular grid of parallel strands with cos*(*θ*)=1,* as schematically presented in [Figure 6.](#page-7-0) In a real cable, the spatial distribution of strands is not that regular at all. Most of the free space in a cable, determining the void fraction, seems concentrated at larger gaps likely between the higher cabling stages (not the final-one) and is not available for the majority of the individual strands. This is illustrated by the CS1 cable cross section in [Figure 7,](#page-8-0) where larger voids are surrounded by regions with rather highly compacted strands. Although this conductor was mechanically loaded in the press, virgin conductors show the same pattern of larger void concentrations.

![](_page_8_Figure_2.jpeg)

**Caption:** Cross-sectional image of an ITER CS1.1 Central Solenoid Model Coil conductor post-transverse cyclic loading, highlighting the sublayer interactions and distortions within the conductor. This visual aims to provide insights into mechanical deformations typical within superconducting applications .


<span id="page-8-0"></span>*Figure 7. The cable cross section of an ITER CS1.1 Central Solenoid Model Coil conductor after transverse cyclic loading in the Twente Cable Press.*

Together with the variation in the wavelengths, the angle of the strands with the surrounding strands and the friction between the strands (causing hysteresis) the theoretically maximum deflection becomes further restricted. Based on the photographs of conductor cross sections we estimate the average available space between strands and we correct the theoretically maximum deflection per strand (*f*sbm) by a factor of 3 and determine the effective *f*sbm by:

$$f\_{sbm} = \frac{f\_{cbm}}{\Im \cdot N\_s^{0.5}} \text{ [m]} \tag{13}$$

For the CS1 type of cable with a void fraction of 0.36 and 6 mm wavelength, we obtain a maximum (average) possible bending deflection per strand (*f*sbm) of 41 µm. This is still without accounting for contact deformation at the strand crossings but this mechanism is built-in further on.

Transverse stress-strain measurements are performed with the TARSIS crossing-strands probe on the LMI-EM strand [20] and are presented in [Figure 8.](#page-9-0) It appears that the loading curve is practically without hysteresis when unloading. The initial deformation requires only a very small load and we assume that the initial strand deformation up to a diameter compression of almost 40 µm (5 % strain) already occurs partly during cabling of the conductor, but mostly throughout the final compacting step, after inserting the cable in the conduit. This allows us to use a constant value of 3.3 GPa for the strand transverse modulus of elasticity, *E*<sup>⊥</sup> in the model.

![](_page_9_Figure_1.jpeg)

**Caption:** Stress-strain graph illustrating the mechanical response of LMI-EM strands under transverse load using the TARSIS crossing strands probe. The figure presents data on deformation and strain profiles crucial for understanding plastic and elastic deformation characteristics in strand interactions .


<span id="page-9-0"></span>*Figure 8. The stress-strain characteristic of the LMI-EM strand measured in the TARSIS crossing strands probe [20].*

The actual available space for bending in the virgin state is determined by the calculated *f*sbm (relation 13). However, when the cable is compressed by the electromagnetic load (or mechanically in the press), the strands will deform not only by bending but also at the crossings and line contacts. As a result, the available space for bending will be further restricted. This is formulated in the model as:

$$f\_{ab} = f\_{abm} - f\_{sc} \tag{10}$$

with *f*sc the strand deformation per strand crossing and line contact and *f*sb is the space for bending, decreasing with accumulating electromagnetic load from layer to layer in the cable bundle.

The bending mechanism throughout the layers of the cable, is now described with relations 1, 2 and 9 and the successive limitation of the possible deflection *f*sb-n with increasing *n* for higher orders of *Lw*, limiting the possible peak strain in the strands with relation 14.

The basic principle of contact stress evolution in the TEMLOP-Model is depicted in the scheme of [Figure 2,](#page-4-0) where strand 1 from layer B is crossing under an angle ϕ with the strands belonging to layers A and C. With further increasing the load [\(Figure 2b](#page-4-0)), strand 1 with the wavy pattern will make contact with strand 2. The transverse load, initially a reaction force only supported by the contacts spaced by the original periodicity in crossing strands represented by *L*w1, becomes also partly distributed now by strand 2 from layer C below. Important is also the angle ϕ in determining the contact stress and this angle is directly connected to the cabling pitches [\(Figure 4](#page-6-0) and [Figure 5\)](#page-6-1). We assume that the relation between *L*<sup>w</sup> and the crossing angle ϕ is linear and inverse proportional for larger wavelengths. Based on the crossing angle ϕ and *L*<sup>w</sup> determined for the mentioned CS1 and CS2 type of conductors we obtain empirically:

$$L\_w \cdot \sin(\phi) = 4.3 \cdot 10^{-3} \qquad\qquad\qquad\text{[m]}\tag{15}$$

in which 4.3⋅10-3 m is the characteristic length *L*φ. So, instead of relation 5, we can write now for the contact area, *A*c:

$$A\_c = \frac{A\_{c\_\perp} \cdot L\_w}{L\_\wp} \tag{1m^2}$$

with *A*c<sup>⊥</sup> as the minimum projected contact area for a strand crossing. Towards shorter wavelengths, the absolute minimum of this contact area is restricted by the square of the strand diameter, *d*s. Thus, we use a hyperbolic description for the computation of the effective contact area in relation to *L*w:

$$A\_c = d\_s^2 \left(\frac{L^2}{L\_\wp^2} + 1\right)^{0.5} \tag{17}$$

This description of the contact area is used for the computation of the results presented in this paper. The numerical method uses discrete steps in the increase of the contact area from *L*w(n) to *L*w(n+1), which is the explanation for the discontinuities in some of the curves shown later on. The *A*<sup>c</sup> increases stepwise from *L*w(n) to *L*w(n+1), proportional to *n* instead of doubling each time the contact area in order to make the strand layer
interactions more gradual. This can be considered as a conservative approach with respect to the influence of contact stress.

Instead of using the discrete steps from *L*w(n) to *L*w(n+1) in the numerical process for the increase of the contact area with rising load as explained above, we can also define a more gradual evolution of this process. In the basic principle of the TEMLOP-Model depicted in the scheme of [Figure 2,](#page-4-0) strand 1 from layer B is crossing under a relatively large angle ϕ with the strands belonging to layers A and C. It is not difficult to imagine that for a much smaller crossing angle, the crossing strands from layer C act nearly as a line contact for strand 1 from layer B. This is well illustrated by the cabling pattern in [Figure 5.](#page-6-0)

The effect of a limited bending deflection, further restricted by the deformation at the strand contacts at increasing load, combined with a progressively homogenisation of the contact load can be formulated as follows:

$$A\_c = \left(L\_{w\_s} - 2 \cdot L\_c\right) \cdot \frac{d\_s}{\sin \varphi} \qquad \qquad \text{[m}^2\text{]} \tag{18}$$

in which *L*<sup>c</sup> is the (decreasing) "characteristic bending length" as indicated in [Figure 2c](#page-4-0) and *L*<sup>l</sup> is the (increasing) length along strand 1 with (homogenously) distributed load: *L*l=*L*w-2.*L*c. The minimum value of *A*<sup>c</sup> is given by relation 5 (with ϕ=45° for *L*w=6 mm) starting in the virgin condition of the cable, represented by [Figure 2a](#page-4-0). As soon as strand 1 meets strand 2, not only the bending wavelength becomes shorter but also at the same time, *A*<sup>c</sup> becomes larger. The characteristic bending length *L*<sup>c</sup> is derived from the standard bending beam formula for a point load with one clamped end and a supported end at the other side but obviously also other descriptions can be used.

$$L\_c = \left(\frac{48\sqrt{5} \cdot f\_{sb} \cdot E\_{\parallel} \cdot I}{F}\right)^{\frac{1}{3}}, \quad \text{[m]}\tag{19}$$

It appeared that utilising relation 19 leads to similar conclusions as when using relation 17.

With mentioned relations 1-17 and the relevant strand parameters like the experimentally determined axial and transversal *E*-modulus (and diameter), the overall mechanical response of the cable to an electromagnetic force can be described now. For the cable properties, we need the void fraction to determine the bending deflection limit. The *L*<sup>w</sup> is a parameter that is directly corresponding to the cabling pitches and will be varied to demonstrate the influence on the conductor performance. The conductor performance in terms of critical current reduction due to bending and contact stress is included by means of experimentally determined relations that are presented hereafter.

#### **4. Critical current reduction in strands**

### **4.1. Bending strain**

It was recognised by Ekin [1], that the transport properties of an Nb3Sn strand is not only affected by the applied bending strain but also depends on the inter-filament electrical resistivity. The electrical resistance between the twisted filaments determines whether the distance between the periodically distributed peak strains in the filaments is short or long compared to the current transfer length. One extreme is that current transfer between the filaments is not allowed due to the high resistance, and then the minimum *I*<sup>c</sup> for each filament specifies the filament critical current. In this case, the strand *I*<sup>c</sup> corresponds to the sum of the filament critical currents, which is limited by the maximum strain along filaments at any point. In this extreme case for short pitch or high matrix resistivity (rho), each *I*<sup>c</sup> is written as [1]:

$$I\_c = \frac{2A}{\mathcal{E}\_{b0}^2} \int\_0 J\_c(B, T, \varepsilon\_{th} - \varepsilon) \varepsilon d\varepsilon \tag{20}$$

*A* is the strand cross section area, ε is the strain in the filaments over a cross section of the strand, εbo is the peak bending strain in the outer ring of filaments, εth is the thermal pre-compression of the Nb3Sn filaments and *J*<sup>c</sup> is the critical current density. Note that for a strand the position of the filaments is fixed on a certain radius when travelling along the axis of the wire while in a CICC the strands are fully transposed.

If on the other hand current transfer is allowed at a low voltage level, the overall *I*<sup>c</sup> of a strand is the sum of the filament currents at any section considering the local strain variation over the section. In this other limiting case for long pitch or low matrix resistivity, *I*<sup>c</sup> can be expressed as:

$$I\_c = \frac{2A}{\pi \varepsilon\_{b0}^2} \int\_{-\varepsilon\_{b0}}^{\varepsilon\_{b0}} J\_c(B, T, \varepsilon\_{th} + \varepsilon) \sqrt{\varepsilon\_{th}^2 - \varepsilon^2} \, d\varepsilon \,, \tag{A} \tag{21}$$

In order to solve the equations 20 and 21 one needs to know the axial *I*c(ε) relation, which can be obtained experimentally [28]. The shape of the computed curves from relation 20 and 21 is primarily driven by the *I*cstrain variation of the particular Nb3Sn strand and are both plotted in [Figure 9.](#page-11-0) It was clearly identified recently, at least for several of the ITER Model Coil strands used in the CSMC and TFMC, that the reduced critical current can be accurately described by only the low resistivity regime representing full interfilament current transfer. [21] (see [Figure 9\)](#page-11-0). The data are measured with an *I*<sup>c</sup> criterion of 10 µV/m. However, experiments were performed on different strands that were swaged into a steel tube [14], to simulate the influence of the axial thermal pre-compression of the conduit on the cable bundle due to the higher coefficient of thermal expansion. It appeared for these strands that the reduced *I*<sup>c</sup> does not correspond anymore to the low transverse resistivity regime [16] although the internal resistivity remains the same. An example is given in [Figure 10,](#page-12-0) where a bronze strand indicated with VAC (from CS1-type) which was used for the inner layer CSMC production, showed such a typical performance. The reduced *I*<sup>c</sup> versus bending strain is plainly lower then predicted by relation 21 and it is not clearly understood why, although strand FEM simulations performed by Mitchell [29,30] suggest an influence of the radial strain component. We confirmed by means of AC loss measurements, that the steel compression did not affect the interfilament resistivity, possibly causing a shift downwards the high resistivity regime [16].

For the TEMLOP-Model, we assume that the real *I*<sup>c</sup> reduction with increasing bending strain under axial precompression applies to the uncompressed strand condition (only strand cool-down strain). This is less severe than the performance of the strand in a steel tube (with imposed axial and radial thermal compression) but conservative again with respect to the case from relation 21, representing axial compressed strand. Therefore, we take the polynomial fit through the experimental data in [Figure 9](#page-11-0) to account for the bending strain in the TEMLOP-Model. In fact, this assumption and the factor 3 from relation (13) mainly determine the quantitative accuracy of the model, but the qualitative prediction in performance remains unaffected.

For the layers of strands travelling from lowest load to the accumulated peak load, the magnetic field decreases from 12 T to 11 T (see [Figure 3\)](#page-5-0). As the critical current varies substantially through the cable cross section due to this field variation, we account for this effect by calculating the dependency of the critical current at 11 T with relation 21 and the strand *I*c(ε) data. The local critical current reduction is then determined by the local field inside the cable by a weighted interpolation of the reduction between 11 T and 12 T, assuming a linear field profile in the cable cross-section from layer to layer.

![](0__page_11_Figure_6.jpeg)

**Caption:** Graph showing reduced critical current (Ic) versus bending strain for LMI-EM strands at 12 T and 4.2 K. The curve is fitted using a polynomial model for both high and low rho conditions, showing significant critical current decrease with increasing bending strain. This emphasizes the importance of understanding mechanical strain impacts on critical properties for optimizing superconducting cable design.


<span id="page-11-0"></span>*Figure 9. Measured Ic versus applied bending strain (with polynomial fit) at 12 T and 4.2 K for the LM-EM strand used for the TFMC (taken from [20])with the computation of the curves for full and no inter-filament current transfer under bending.*

![](0__page_12_Figure_1.jpeg)

**Caption:** Plot illustrating reduced critical current (Ic) as a function of peak bending strain for different strand configurations (bare strand and SS tube) at 12 T and 4.2 K, showing fitted polynomial curves for low and high resistivity conditions. This figure underscores the interplay between mechanical strain and electromagnetic properties, highlighting the effects of mechanical constraints on strand performance .


<span id="page-12-0"></span>*Figure 10. Measured reduced Ic versus applied bending strain (with polynomial fit) at 12 T and 4.2 K for the VAC bronze strand used for the CSMC (taken from [16])with the computation of the curves for full and no inter-filament current transfer under bending (bare strand and with SS tube).*

### **4.2. Contact stress**

A similar strategy is followed for the interaction of the transverse load on strand crossings. Although several experiments are known [31-34], there is no analytical expression available that describes the reduction of the critical current under increasing contact load. On top of that, it is obvious that a polynomial fit through the experimentally obtained dependency has the best accuracy. Therefore, we measured the reduction of the critical current for the LMI-EM strand with the TARSIS probe for crossing strands [20] and the results are depicted in [Figure 11.](#page-12-1)

In addition, also here we account for the magnetic field decrease along the cable cross section from 12 T to 11 T (see [Figure 3\)](#page-5-0) and measured the dependency at both fields. Similar to the method applied for strand bending, the local critical current reduction is determined by the local field through weighted interpolation between the measured reduction at 11 T and 12 T through the cable from layer to layer. Note that for the strands that are in 11 T magnetic field in the accumulated load region, the critical current is not affected up to 75 MPa stress.

![](0__page_12_Figure_6.jpeg)

**Caption:** The plot illustrates the reduction of the critical current (Ic) as a function of transverse stress for LMI-TFMC strands at 12 T and 4.2 K. Measurements are shown for specific strand configurations, indicating polynomial fits to data that highlight stress-dependent Ic degradation. This figure is crucial for understanding how transverse mechanical stress impacts superconducting efficiency and for validating stress-strain response models used in superconductor design .


<span id="page-12-1"></span>*Figure 11. The reduction of the critical current versus the applied stress for the LMI-EM strand at 11 T and 12 T measured in the TARSIS crossing strands probe [17].*

# **4.3. Integrated model**

The overall critical current reduction in the cable is calculated layer by layer from minimum to maximum accumulated load and in a magnetic field changing from 12 T to 11 T. Just as for interfilament current

transfer in a strand described by relations 20 and 21, we can distinguish between no and full current transfer between strands in the cable. When we assume full current transfer between strands then the current is allowed to redistribute and we can add all critical currents from layer to layer to get the average cable critical current reduction. This is reflected by the following relation:

$$\left| \left( \frac{I\_c}{I\_{c0}} \right)\_{cable} = \frac{\sum\_{l}^{N\_l} \left| \sum\_{L\_{v\_l}}^{L\_{v\_l}} (\mathcal{E}\_b, \mathcal{B}) \cdot \frac{I\_c}{I\_{c\_0}} (\mathcal{F}\_c, \mathcal{B}) \right|}{N\_l} \right| \tag{22}$$

in which the peak bending strain in the filamentary region (εb) is derived from the load in the layer of strands *N*l. The reduced critical current, *I*c/*I*c0, depends on the field and if the interpolated value exceeds 1 when the combination of bending strain and magnetic field is sufficiently low not to reduce the real critical current, a value of 1 is just taken instead. The reduced critical current from the transverse contact load (*F*c), represented by the right term in relation 22, is multiplied with the reduction obtained from the bending strain (left term) assuming that both reductions add at the strand crossovers. The *I*<sup>c</sup> reduction of all layers is averaged to come to the reduced *I*<sup>c</sup> of the entire cable, because we assume full current distribution between strands.

When the current in the strands is not able to redistribute due to high interstrand contact resistance, the peak strain and the corresponding minimum *I*<sup>c</sup> limits the current for all strands as all strands have to pass the peak strain region within one pitch length of the final cabling stage. Depending on the ratio of the consecutive cabling pitches, it may be possible that the peak strain region is passed with a periodicity larger than the last stage cabling pitch. A high interstrand contact resistance, or in other words, a long current transfer length, further limits the *I*c.

In [35] the influence of the current transfer length is pointed out for a full size NbTi conductor. We found that the current transfer length between the wrapped final-minus-one cabling elements (petals) amounts to several tens of meters. Without the resistive wraps around the last minus one cabling stages the current transfer length amounts to several meters. This means that for strands inside the last-minus-one cabling stage some redistribution may be allowed when the periodicity is larger than one last stage pitch length. A conservative approach would be to choose for no redistribution. For our analysis we examine both limts, full and no current redistribution.

The model can be further refined by changing the stepwise change in the bending wavelengths from *L*w(n) to *L*w(n+1) into a more gradual numerical approach but this has likely only minor effect on the overall outcome.

As already mentioned above, the exact influence of the conduit axial thermal pre-compression on the critical reduction of the cable *I*c/*I*c0(εb,*B*) is not known and only for this part of the model, we rely on a hypothesis with some higher degree of uncertainty. At this point some additional effort is required, asking for a more fundamental approach in investigation of the 3D strain behaviour of Nb3Sn layers in relation to strand geometries.

### **5 Results of model computations**

## **5.1. Mechanical response**

The cable deformation (transverse compression) versus the applied electromagnetic *IxB* load is calculated for different wavelengths *L*<sup>w</sup> in a conductor with 1152 strands and a void fraction of 0.36. The strand properties are from the LMI-EM strand with E// = 29 GPa and E<sup>⊥</sup> = 3.3 GPa [20]. The results are plotted in [Figure 12.](#page-14-0) The cable becomes stiffer with shorter wavelength, which is also illustrated by [Figure 13](#page-15-0) where the overall transverse cable modulus of elasticity versus the applied electromagnetic *IxB* load is presented for different wavelengths.

The deflection for bending and contact load is depicted in [Figure 14,](#page-15-1) for the total deflection we assume that both components can simply be added although this is not trivial. The TEMLOP-Model gives a total cable compression of about 1.3 mm, with 1.1 mm for bending and 0.2 mm for direct strand contact stress on a cable with a void fraction of 0.36 and *L*w=0.006 m. The total transverse cable compression saturates for contact and bending stress with increasing *IxB* load. After about 400 kN/m, bending reaches its maximum deflection limit while the contact deformation progressively continues.

The deflection from bending and contact load differs from layer to layer through the cable. In [Figure 15](#page-15-2) we can see how the deflection starts to increase linearly in the layers from high to low magnetic field until the maximum limit is reached around layer 4, coming in contact with strands in the next layer (*L*w=0.007 m).

Here the contact stress changes rapidly due to the numerical structure of the model, imposing a sudden increase of the contact surface when changing from *L*w1 to *L*w2. Behind this layer the bending deflection becomes further restricted because the increasing contact stress is compressing further the strand crossovers, at the same time increasing the contact deformation (deforming the strands). At layer 20, the contact stress changes rapidly again due to the sudden increase of the contact surface when changing from *L*w2 to *L*w3.

In [Figure 16](#page-16-0) we find the typical initial increase of the bending strain from the section characterised by *L*w1=0.007 mm together with the rise of the contact stress until reaching contact with the first strand of the layer below. Then the bending strain linked to *L*w1 (almost 0.7 %) relaxes and the contact stress increases with only half the initial slope. However, at the same layer the bending strain connected with *L*w2 starts to increase, reaching a peak strain of about 1.1 %, exceeding that of the maximum strain connected to *L*w1, but deeper in the conductor and in significantly lower field. This maximum for *L*w2 bending occurs at layer 20, where the maximum deflection is reached for this wavelength. The peak strain related to *L*w3 remains restricted to below 0.5 %. The contact stress reaches its maximum accumulated load at the low field side of the conductor at 40 MPa. We also find that *L*w4 has not yet become involved in the loading process.

[Figure 17](#page-16-1) shows what happens with the peak strain and stress when we double the *L*w1 to 0.014 mm. We find that *L*w4 has also become involved here but the peak strain occurs now in layer 11 instead of layer 23 and is associated to *L*w3. However, the peak strain in the cable is now less than 0.6 % and the peak stress is reduced to 35 MPa.

The sudden increase of the contact surface between strands from layer to layer is shown in [Figure 18](#page-16-2) for a wavelength of 0.007 m. For this wavelength, the contact surface remains constant in the first two layers at the high field region while the contact stress increases linearly towards higher load. In the third layer additional strand-to-strand contacts are formed connected with *L*w2, consequently increasing the contact area. From that layer of, the stress increases with halve the initial slope until further increasing contact area at layer 20, associated with *L*w3 where the process repeats.

All formulations and steps applied in the model, describing the local bending strain and contact stress in each layer are essential for an accurate computation of the local critical current.

![](0__page_14_Figure_6.jpeg)

**Caption:** Graph depicting cable deflection (fc) in meters versus IxB load in kN/m for different wavelengths (Lw) ranging from 0.003 m to 0.012 m, with void fraction (vf) of 0.36 and strand modulus (E//) of 29 GPa. The graph illustrates that deflection increases with higher loads and is more pronounced at smaller wavelengths, indicating significant impact of mechanical load and strand geometry on cable performance.


<span id="page-14-0"></span>*Figure 12. The cable deformation (compression) versus the applied electromagnetic IxB load for different wavelengths Lw (vf=0.36). The strand properties are from the LMI-EM strand with E|| = 29 GPa and E*⊥ *= 3.3 GPa.*

![](0__page_15_Figure_1.jpeg)

**Caption:** Figure showing overall transverse cable modulus of elasticity versus applied IxB load for different wavelengths, demonstrating how modulus changes reflect on cable stiffness under electromagnetic loads (vf=0.36). The data is critical for understanding how cable design and wavelength affect performance metrics like stiffness and deformation .


<span id="page-15-0"></span>*Figure 13. The overall transverse cable modulus of elasticity versus the applied electromagnetic IxB load for different wavelengths Lw (vf=0.36).*

![](0__page_15_Figure_3.jpeg)

**Caption:** Graph detailing the transverse cable compression versus IxB load for different components like bending and contact stress with measurements of 0.36 void fraction. Highlights the progressive cable compression as loads increase, indicating component contributions to overall structural deformation.


<span id="page-15-1"></span>*Figure 14. The transverse cable compression versus the applied electromagnetic (IxB) load for bending, contact stress and both components added (vf=0.36).*

![](0__page_15_Figure_5.jpeg)

**Caption:** Graph showing deflection per strand layer versus strand layer number (Nl) for bending deflection, contact compression, and total deflection under a load of 1152 strands with Lw of 0.007 m and vf of 0.36. It indicates a significant deflection starting at low number layers, leveling off at higher layers due to increased load bearing from strand accumulation.


<span id="page-15-2"></span>*Figure 15. The bending deflection and strand contact deformation (per strand layer) versus the strand layer number, Nl, from high to low field at 750 kN/m load for Lw=0.007 m (vf=0.36).*

![](0__page_16_Figure_1.jpeg)

**Caption:** Figure showing bending strain and contact stress per strand layer against strand layer number (Nl) from high to low field at a load of 750 kN/m with a wavelength of 0.007 m and void fraction of 0.36. The figure delineates variations in mechanical stress distribution as a function of strand layer structure, essential for understanding deformation and structural performance .


<span id="page-16-0"></span>*Figure 16. The bending strain and contact stress (per strand layer) versus the strand layer number, Nl, from high to low field at 750 kN/m load for Lw=0.007 m (vf=0.36). In the legend the 'e peak-fil reg' is the peak bending strain in the filamentary region or* ε*b0.*

![](0__page_16_Figure_3.jpeg)

**Caption:** This figure depicts the bending strain and contact stress per strand layer versus the strand layer number at a load of 750 kN/m for a wavelength of 0.014 m and a void fraction of 0.36. The results highlight the variation in strain and stress across layers, particularly how these parameters interact during high-load conditions .


<span id="page-16-1"></span>*Figure 17. The bending strain and contact stress (per strand layer) versus the strand layer number, Nl, from high to low field at 750 kN/m load for Lw=0.014 m (vf=0.36).*

![](0__page_16_Figure_5.jpeg)

**Caption:** Image showing a close-up of strands bundled together, emphasizing the arrangement and texture. This visual provides insight into strand connectivity and microstructural coherence essential for understanding mechanical and electromagnetic properties in applications.


<span id="page-16-2"></span>*Figure 18. The interstrand contact stress and the contact area versus the strand layer number, Nl, from high to low field at 750 kN/m load for Lw=0.007 m (vf=0.36).*

### **5.2. Critical current and transverse** *IxB* **load**

The reduction of the critical current depends on the applied bending and contact load, which varies from layer to layer, and the local magnetic field. Thus, we calculate the *I*<sup>c</sup> reduction for each strand layer number, *N*l, from high to low field from zero to 750 kN/m load, in a conductor with a void fraction of 0.36 and a wavelength of 0.006 m. The strand properties are from the LMI-EM strand with *E*//=29 GPa and *E*⊥=3.3 GPa. The results for bending and contact load are plotted in [Figure 19.](#page-18-0) The curves represent the reduction from purely interstrand contact load distinguished from the reduction from pure bending throughout the layers of strands in the cable. The reduction from strand contact load is practically nil so almost all reduction is caused by bending. The evolution of the critical current reduction from layer to layer is illustrated in [Figure 20](#page-18-1) for strand currents (*I*s) from 5 A to 50 A. A strand current of 50 A corresponds to an *IxB* load of 750 kN/m. In [Figure 21](#page-19-0) the same is carried out for a wavelength of 0.009 m. The *I*<sup>c</sup> reduction is restricted at the low field (accumulated load) zone of the cable cross-section because the maximum deflection is reached.

In the high field region of the cable cross section, bending is responsible for a rapid decrease of the *I*<sup>c</sup> towards the low field region, until the maximum bending deflection is reached for *L*w1 [\(Figure 2\)](#page-4-0). The reduction in this high field region is amost completely attributed to bending with the larger wavelength *L*w1 while the reduction related to *L*w2 becomes significant somewhat deeper in the conductor. It appears that when the strand has reached the first bending limit (see [Figure 2b](#page-4-0)) and just before the strand becomes supported by the layer of strands below, (*N*y+1), dividing the wavelength *L*<sup>w</sup> by two (*L*w1 to *L*w2), the bending strain reaches up to 0.8 % for an *L*w1 of 6 mm in layer 5. When the strand becomes actually supported by the strand below, changing *L*w1 to *L*w2, the bending strain reach a maximum of 1.3 % and although this level of strain is extreme, the cable *I*<sup>c</sup> is only moderately affected because this strain occurs in layers *N*<sup>y</sup> with largest accumulated load that are in lowest magnetic field.

When we increase the wavelength *L*w1 to 0.009 m (see [Figure 21\)](#page-19-0), we find that the reduction in *I*<sup>c</sup> becomes less excessive, in particular for higher currents. This is due to the lower peak bending strains and contact stresses. The maximum available space for bending leads to lower bending strain for longer *L*w, while at the same time the strands become supported along longer lengths and on more locations, reducing the local peak contact stress.

When averaging the *I*<sup>c</sup> reduction from all layers for successive *IxB* loads, we obtain the overall cable *I*<sup>c</sup> reduction versus the *IxB* load (relation 22). This is summarised in [Figure 22](#page-19-1) for different wavelengths. Here we find an impressive improvement in the performance when the wavelength *L*<sup>w</sup> is increased from 6 mm to 20 mm.

The average wavelength (*Lw*) determined from the disassembled full-size ITER CS1 conductor in [Figure 4](#page-6-1) is 6 mm with a standard deviation of 2 mm. Most wavelengths are thus within the range *L*w=4 mm and *L*w=8 mm. When we take the average overall reduction of the *I*<sup>c</sup> versus the *IxB* load calculated for *L*w= 4, 5, 6, 7, and 8 mm we obtain the upper curve presented in [Figure 23.](#page-19-2) The computed *IxB* behaviour is in good agreement with analyses that has been reported from experiments on coils and short sample tests [3,5]. However, this curve represents the situation for full current transfer. For comparison we show in the same Figure the curve accounting for no current transfer with much stronger degradation. All other results presented here are reflecting full interstrand current transfer.

The remarkable improvement in the performance, when the wavelength *L*<sup>w</sup> is increased from 6 mm to 20 mm, is clearly illustrated in [Figure 24](#page-20-0) with the overall reduction of the critical current versus the characteristic bending wavelength. It appears that also with a wavelength shorter than 5 mm an improved performance is possible, confirming that the present ITER cable design is unfortunately nearly at the worst possible conductor performance. However, application of shorter twist pitches is not favourable as they lead to an inefficient use of strand and an increase of the cable size. In the same Figure we plotted the peak strain in the strands showing a matching correlation with the *I*<sup>c</sup> reduction. The peak strain and the peak contact stress in the conductor versus the wavelength are presented in [Figure 25.](#page-20-1) Also for the contact stress the same dependency is obtained although there is no effect on the critical current. The influence of the interstrand resistivity, allowing for full current redistribution or non at all, is depicted in [Figure 26.](#page-20-2) It appears that for higher *L*<sup>w</sup> the difference between full and no current transfer becomes less pronounced due to a different balance between the stress and strain concentrations across the cable in relation to the self-field profile. This is well illustrated by [Figure 20,](#page-18-1) showing a small number of layers with severe *I*<sup>c</sup> degradation and [Figure 21](#page-19-0) with a more balanced distribution of the degradation over the cable cross section.

A decrease of the void fraction also leads to a better performance. This is demonstrated in [Figure 27](#page-21-0) for void fractions of 0.36 and 0.33, the previous and present ITER conductor design values respectively. The influence of the void fraction is significant and can be explained as limiting the available space for bending deflection.

The effect of the stiffness of the strand is simulated with the properties of the internal tin strand, used for all simulations, and a bronze strand with a lower axial elastic modulus. The result is shown in [Figure 28,](#page-21-1) illustrating that a strand with lower E// is more degraded as it gets stronger deformed by the bending loads.

![](0__page_18_Figure_3.jpeg)

**Caption:** Figure showing the reduction of critical current (Ic) versus strand layer number (Nl) from high to low field at a 750 kN/m load with a wavelength (Lw) of 0.006 m and void fraction (vf) of 0.36. The graph distinguishes between reductions due to pure interstrand contact load and pure bending. The data reveals that pure bending significantly contributes to the overall reduction in critical current as strand layer number increases.


<span id="page-18-0"></span>*Figure 19. The reduction of the critical current versus the strand layer number, Nl, from high to low field at 750 kN/m load for Lw=0.006 m (vf=0.36). The curves represent the reduction from purely interstrand contact load distinguished from the reduction from pure bending.*

![](0__page_18_Figure_5.jpeg)

**Caption:** Graph presenting reduction in critical current (Ic) versus strand layer number (Nl) at different strand currents (Is) up to 750 kN/m load for Lw=0.006 m (vf=0.36). The graph indicates that higher currents exacerbate reduction from bending and contact stress.


<span id="page-18-1"></span>*Figure 20. The reduction of the critical current versus the strand layer number, Nl, from high to low field at different strand currents (Is) up to 750 kN/m load for Lw=0.006 m (vf=0.36). The curves represent the overall reduction from bending and contact stress.*

![](0__page_19_Figure_1.jpeg)

**Caption:** Graph representing reduction of critical current (Ic) versus strand layer number (Nl) for different strand currents (Is), at a 750 kN/m load with Lw of 0.009 m and vf of 0.36, depicting overall reduction from bending and contact stress. Different symbols represent varying currents, demonstrating their impact on critical current reduction across strand layers.


<span id="page-19-0"></span>*Figure 21. The reduction of the critical current versus the strand layer number, Nl, from high to low field at different strand currents (Is) up to 750 kN/m load for Lw=0.009 m (vf=0.36). The curves represent the overall reduction from bending and contact stress.*

![](0__page_19_Figure_3.jpeg)

**Caption:** Graph illustrating the overall reduction of critical current (Ic) against IxB load across different wavelengths from 2 mm to 20 mm for superconducting strands with a void fraction of 0.36, showing the sensitivity of Ic to mechanical load variation. This data is pivotal for predicting performance under variable electromagnetic load conditions.


<span id="page-19-1"></span>*Figure 22. The overall reduction of the critical current versus the IxB load for different wavelengths from 2 mm to 20 mm (vf=0.36).*

![](0__page_19_Figure_5.jpeg)

**Caption:** Simulation of overall reduction in critical current (Ic) across varying bending wavelengths for CS1 conductor, indicating differences between full and no interstrand current transfer modes. This highlights the significance of cable cross-section configuration on current conduction efficiency .


<span id="page-19-2"></span>*Figure 23. The computed average overall reduction of the Ic versus the IxB load from Lw= 4, 5, 6, 7, and 8 mm representing the spread in the measured Lw from the CS1 conductor with an average Lw of 6 mm and a standard deviation of 2 mm (vf=0.36). The upper curve accounts for full interstrand current transfer while the lower curve represents no current transfer.*
![](1__page_20_Figure_1.jpeg)

**Caption:** The graph portrays the overall reduction of critical current (Ic) and peak bending strain versus the characteristic bending wavelength (Lw) with a void fraction (vf) of 0.36. It emphasizes how different wavelengths impact the stress-strain relationship in superconducting cables, specifically under electromagnetic load conditions .


<span id="page-20-0"></span>*Figure 24. The overall reduction of the critical current and the peak bending strain in the cable cross – section versus the characteristic bending wavelength (vf=0.36).*

![](1__page_20_Figure_3.jpeg)

**Caption:** Figure showing the peak contact stress and peak bending strain within a cable cross-section as a function of the characteristic bending wavelength (Lw) with a void fraction (vf) of 0.36. The data demonstrates the relationship between wavelength and mechanical stress-strain behavior, crucial for optimizing cable performance under applied mechanical loading conditions.


<span id="page-20-1"></span>*Figure 25. The peak contact stress and peak bending strain in the cable cross section versus the characteristic bending wavelength (vf=0.36).*

![](1__page_20_Figure_5.jpeg)

**Caption:** The graph outlines the reduction of critical current (Ic) under different bending wavelengths, demonstrating effects of interstrand resistance. It models predictions across a spectrum of ISR levels to optimize conductor configurations.


*Figure 26. The overall reduction of the critical current for full interstrand current transfer (low interstrand resistance, ISR) and no current transfer (high ISR, polynomial fit) in the cable cross –section versus the characteristic bending wavelength (vf=0.36).*

![](1__page_21_Figure_1.jpeg)

**Caption:** Graph displaying the reduction of critical current (Ic) under varying IxB loads for strands with void fractions 0.33 and 0.36, demonstrating how different voids affect performance. This figure provides a vital insight into the mechanical resilience and loading capacity of superconducting cables under specified loading conditions .


*Figure 27. The overall reduction of the critical current versus the IxB load for conductors with a void fraction of 0.36 and 0.33.*

![](1__page_21_Figure_3.jpeg)

**Caption:** Graph showing the relationship between reduced critical current (Ic) and different IxB loads for two types of strands with varying elastic modulus properties (E// = 29 GPa and E// = 21 GPa). The illustration elucidates how differing material properties affect Ic under load conditions, emphasizing the mechanical and electromagnetic trade-offs in superconducting strand design .


*Figure 28. The overall reduction of the critical current versus the IxB load (vf=0.36). The strand properties are from the LMI-EM internal tin strand with E|| = 29 GPa and E*⊥ *= 3.3 GPa and the VAC bronze strand wit. E|| = 21 GPa and E*⊥*= 3.0 GPa*

### **5.3 Evolution of peak strain with** *L***w(n)**

The relation between the local peak strain and stress is depicted in [Figure 24](#page-20-0) and [Figure 25.](#page-20-1) However, the local peak strain and the number of higher orders *n*, required to describe the entire cable (*N*l) strongly depend on the wavelength *L*w1. Knowing now that for longer wavelength the peak strain and stress relaxes, we explore how the peak strain evolves for the higher order wavelengths. The extension to higher order bending wavelengths in the direction of load accumulation from layer to layer, successively going from *L*w(n) to *L*w(n+1), is shown in [Figure 29.](#page-22-0) The peak strain throughout the cable decreases, when higher order wavelengths become involved. For extremely short wavelengths, the higher orders for *n* do not occur in the cable bundle and *n*=1. With increasing *L*w1, the peak strain increases until the higher orders of *n* become involved. For longer *L*w, *n* becomes higher together with the homogenisation of the contact stress and bending strain peaks.

![](1__page_22_Figure_1.jpeg)

**Caption:** Graph depicting peak bending strain against higher order bending wavelength number n, examining different bending wavelengths' effects on mechanical strain within superconducting cables. It illustrates how longer wavelengths promote strain homogenization across cable assemblies .


<span id="page-22-0"></span>*Figure 29. Computed peak bending strain versus the higher order number n of the relevant wavelength Lw1 for different characteristic bending wavelength, Lw1 of a cable.*

#### **6 Discussion**

#### **6.1. Mechanical response of the cable**

The TEMLOP-Model predicts a total cable compression of 1330 µm, with 1070 µm for bending and 250 µm for direct strand contact stress, in a cable with a void fraction of 0.36 and a characteristic wavelength of 6 mm. This is in line with the experimental values obtained from the Cryogenic Press [25] and is obviously connected to the derived limitation in bending deflection *f*sb from relation 9 and the strand contact deformation adopted from [Figure 8.](#page-9-0) The experimental values found for the cable compression and correlated displacement per strand, for three conductors with different void fraction, are plotted in [Figure 30](#page-23-0) [36]. For the CS1 type of cable with a void fraction of 0.36 we found a maximum deflection (*f*cm) of 1040 µm [25]. The measured compression per strand then reaches a maximum amplitude of 34 µm. At this point we can remark that also from thermo hydraulic analysis, it was deduced that a so-called third channel between one side of the cable and the inner conduit wall, due to cable bundle deformation after cyclic loading of the CS Insert Model Coil, would entail a gap of about 1 mm [37].

For a cable with a void fraction of 0.33, the model predicts a total cable compression of 1040 µm, with 810 µm for bending and 225 µm for direct strand contact stress (*L*<sup>w</sup> of 6 mm). This is in agreement with what is expected by interpolation for the total deflection in [Figure 30.](#page-23-0) However, the influence of the void fraction on the cable compression as computed with the model is stronger than obtained from the press measurements. This may be due to an underestimation of the total contact deformation as derived from [Figure 8](#page-9-0) and the assumption that the sum of the bending and contact deformation reflects the total compression, which is not necessarily conform the reality.

![](1__page_22_Figure_7.jpeg)

**Caption:** Graph showcasing total compression of different model coil conductors against initial void fraction, highlighting displacement per strand. Data points to performance metrics tied to structural void considerations .


### <span id="page-23-0"></span>*Figure 30. The total compression of several Model Coil conductors with different void fraction and the deduced possible free displacement for bending per strand.*

Transverse stress-strain measurements performed with the TARSIS crossing-strands probe on the LMI-EM strand showed that the loading curve is practically without hysteresis when unloading (see [Figure 8\)](#page-9-0). The low transverse modulus of elasticity of the strands, compared to the modulus in axial direction being one order of magnitude higher, implicitly suggests that the transverse cable bundle compression through contact stress is connected to mostly plastic deformation of strand crossings and line contacts. The plastic deformation from transverse interstrand contact deformation is primarily determined by the copper stabiliser, practically immediately yielding at 4.2 K, while the axial strand stiffness, linked to bending, is significantly enhanced by the Nb3Sn layer [18]. The conductor cross section depicted in [Figure 7](#page-8-0) illustrates the compaction of a cable after test in the Twente Cable Press with peak transverse load for 40,000 cycles. At the upper perimeter of the cable bundle, a permanent plastic deformation of 0.6 mm is clearly visible leaving a gap between cable and inner conduit wall. This 0.6 mm plastic deformation consists of 250 µm representing direct strand contact stress deformation according to the model and the remaining part can be attributed to plastic deformation from bending. This leaves about 0.5 mm scope for compression with elastic bending after cycling of the conductor. The so-called *IxB* dependency, as it is often presented in a εextra(*IxB*) plot, is then directly linked to the remaining span for elastic bending. The εextra represents the additional axial strain that is required to match the measured strand performance to the degraded cable performance [5].

As the model is not adapted to comply for plastic deformation and creep, the performance reduction with cycling of the load is not reflected in the results.

The overall modulus of elasticity measured on a CS1 conductor in the Twente Cryogenic Press is plot in [Figure 31](#page-23-1) together with the computed curves for *L*w=6 mm, showing good agreement.

![](1__page_23_Figure_5.jpeg)

**Caption:** Graph comparing measured and computed overall transverse cable modulus of elasticity versus applied IxB load for a CS1 conductor configuration. Highlights demonstrate the model's capability in predicting mechanical stiffness variations due to electromagnetic loading.


<span id="page-23-1"></span>*Figure 31. The measured and computed overall transverse cable modulus of elasticity versus the applied electromagnetic IxB load for the CS1 conductor (vf=0.36). The strand properties are from the LMI-EM strand with E|| = 29 GPa and E*⊥*= 3.3 GPa.*

If we compare the computed deflection of the cable versus the applied load for different wavelengths to the measured cable compression in the Twente Press (see [Figure 32\)](#page-24-0), we find that the behaviour of the cable for increasing number of cycles in the experiment, could be described by the model as for increasing *L*w. The model obviously does not incorporate the effect of plastic deformation and work hardening with cycling, but there is a strong analogy in the characteristics. The computed mechanical response with increasing *L*<sup>w</sup> is linked to a decreasing influence of bending at the same time accompanied with a growing contact surface. In that sense we can understand the measured cable deformation, during already the first cycle, as largely plastic deformation. With further cycling of the load on the cable the component of the plastic deformation decays finally leaving only the component of mainly elastic bending.

![](1__page_24_Figure_1.jpeg)

**Caption:** This figure shows the measured and computed overall transverse cable compression versus the applied electromagnetic IxB load for a CS1 conductor with a void fraction of 0.36. The experimental results are from the CS1 conductor under cyclic loading, while the computed curves are for different wavelengths (Lw). This figure illustrates the correlation between experimental data and model predictions, emphasizing the role of wavelength on cable deformation under electromagnetic load .


<span id="page-24-0"></span>*Figure 32. The measured and computed overall transverse cable compression versus the applied electromagnetic IxB load (vf=0.36). The experimental results are from the CS1 conductor along cyclic loading while the computed curves are for different Lw. The strand properties for the computation are from the LMI-EM strand with E|| = 29 GPa and E*⊥*= 3.3 GPa.*

### **6.2. Critical current reduction**

One of the main aims of the TEMLOP-Model is to verify the possible influence of the cabling pitches on the performance degradation originated by transverse load. As already put forward in [25], the most favourable transverse load distribution is a homogeneous one, representing an almost infinite line contact of parallel strands. The most important prediction of the model, in terms of optimisation the conductor performance by increasing the twist pitches, is understood as a homogenisation of the load distribution or in other words, a significant lowering of the local peak strains. This is clearly illustrated in [Figure 24](#page-20-0) and [Figure 25.](#page-20-1)

The outcome of the model can be explained by exploring the limiting cases for *L*w. When *L*<sup>w</sup> →∞, the angle between the crossing strands goes to zero and the stress reaches to the homogeneously distributed case represented by relation 4. At the same time the bending diminishes as the strands become in fact parallel resulting into minimisation of the bending strain. For *L*w→0, the bending strain also goes to zero as reflected by relation 2 and the contact stress reduces to the homogeneity limit from relation 4 with contact area going to the maximum. With *L*<sup>w</sup> in between the limiting cases, periodic bending strain and local concentration of contact stresses lead to reduction of *I*c.

Moreover, it is not difficult to imagine that an increase of the twist pitches advances the flexibility in the self-regulating mechanism of strands being forced to find the mechanically most favourable position in the cable bundle. This way the strands find better distributed support against bending from subsequent layers of strands towards the accumulated load zone in the cable cross section.

In [Figure 4](#page-6-0) it is shown that for the CS1 type of conductor, the periodicity in the number of strand crossings is relatively high with a wavelength of 6 mm. The first cabling stage triplet has a twist pitch of 45 mm and the second stage quadruplet has a pitch of 74 mm. In the CS2 type of conductor from [Figure 5,](#page-6-1) with much longer twist pitches (about 70 mm and 90 mm respectively) we find a less frequent periodicity with an average wavelength larger than 6 mm. At the same time, we observe a much smaller crossing angle than the average 45 degrees from the CS1 type of conductor.

In view of the model predictions, concerning the dependency between the twist pitch and transverse load distribution along the strands in the conductor, the cabling scheme of the CS2 seems much more advantagous for large conductors with high transverse load [25]. Unfortunately, this type of conductor was only in the low field region of the CSMC and was not examined on its temperature margin. A test of the CSMC outer module (without inner module) under appropriate conditions to explore the CS2 performance could serve as a reference test for the prediction of the TEMLOP-Model.

Unfortunately the cabling scheme for CICCs and not only for ITER conductors, from which experimental data are available, is kept practically the same along the years. As said, this is likely the main reason that the influence of cabling remained unknown.

In the Introduction we mention that part of the analysts in the fusion community postulate that a significant fraction of the Nb3Sn layer in the strand material of the cable is breaking [3,10] but that there is no conclusive evidence. The possibility of filament fracture is supported by experimental work [23] even though no results of an internal inspection of strands from a tested conductor have been presented yet. However,

although it is not relevant for the analysis with TEMLOP presented here, whether the permanent degradation is due to filament breakage or plastic deformation of strands, the outcome of the model gives an indication. At the experiments on the TARSIS bending probe with stepwise increase of the load with intermediate unloading, we find a clear irreversible degradation of the *I*<sup>c</sup> and the *n*-value [1]. In general we observe a permanent degradation in *I*<sup>c</sup> when exceeding an applied bending strain of 0.3 ÷ 0.4 % with the TARSIS probe. At the same time we find a permanent deformation of the strand as for zero load the strand deflection never returns to zero after having applied a certain load level. This means that from only this experiment we are not able to distinguish between plastic deformation and filament breakage. The irreversible strain limit of about 0.4 % corresponds more or less to the applied tensile strain that is required for irreversible degradation of the *I*<sup>c</sup> in experiments for strand axial strain testing [28]. With TEMLOP we find an applied peak bending strain in the Nb3Sn filaments of about 1.3 % for the presently used cabling scheme [\(Figure 24,](#page-20-0) [Figure 25](#page-20-1) and [Figure 29\)](#page-22-0). Assuming that a steel conduit would impose an additional axial pre-compression leading to roughly -0.7 % intrinsic axial strain in the filaments after cool-down, the bending strain would lead to a net tensile intrinsic peak strain of + 0.6 %. If in addition, we also take into account that at the location with peak bending strain the contact stress reaches maxima as well, local filament breakage seems evident.

# **6.4. Related effects**

The most favourable transverse load distribution is obviously a homogeneous one, representing an almost infinite line contact of parallel strands [25]. That pleads for long cabling pitch lengths but this is in conflict with the restrictions given by the generation of coupling loss. The AC loss in terms of interstrand coupling loss is the driving factor for minimization of the cabling pitch lengths together with compelling adequate stiffness to the cable bundle. In principle the generation of coupling loss increases with the square of the pitch length as long as the contact resistance remains constant. In that sense the increase of the coupling loss is a potential disadvantage when choosing longer pitches to enhance transverse load optimisation. However, for the CS2 type of cabling structure we find a coupling loss time constant in the virgin state of about 100 ms, while for the CS1 type a value twice as high is obtained [38]. We take note that a third part of the strands in the CS2 is copper strand instead of all superconducting wire as in the CS1-type. In general, we observe that the coupling loss is roughly proportional to the amount of superconducting strands for comparable cable geometries. From this we can expect that an extrapolation of the coupling loss by only using the ratio of the twist pitch is by far too conservative. It is easy to understand that with increasing the cabling pitches, the contact surface between strands expands while simultaneously the contact stress reduces. An increase of the contact resistance may be a likely explanation for the relatively low level of coupling loss from the CS2-type of conductor, counterbalancing the effect of the increased cable pitches [39]. Consequently, a serious drawback from a possible increase of the coupling loss is not expected.

The transparency of the cable bundle for the cooling medium (liquid helium under pressure) might be slightly affected but also here we do not expect a noticeable impact with a nominal void fraction of 0.33.

An advantage of longer cabling pitches besides the lower transverse load degradation, is an increase of the *cos(*θ*)* correction factor leading to higher critical current density, although the effect is modest. When the wavelength is increased from 6 mm to 20 mm, the *cos(*θ*)* increases from 0.950 to 0.985, increasing the *J*<sup>c</sup> by 3.5 %.

Although we find improvement of the performance for lower void fraction, a warning should be given in relation to this aspect. With further reducing the void fraction of the conductor it appears that during the compaction also the length of the conductor increases [36]. This may reduce the strand diameter and at the same time decrease the *I*<sup>c</sup> of the conductor accordingly.

## **6.5. Recommendations**

As a first test, the prediction can be verified experimentally by for example testing a short sample (in SULTAN). Two legs, identical but with a significant difference in the cabling scheme, in particular for the first four stages, can be compared directly on their performance. A cabling scheme with pitches similar, or better even some longer, than the CS2 conductor seems appropriate although eventually a more extensive parametric variation is required to examine the influence of the cabling parameters in detail. Further work is needed to analyse the optimal cabling scheme.

A test of the CSMC outer module under appropriate conditions, to explore the performance of the module with CS2 conductor having longer cabling pitches, could also serve as an indicative test.

The exact influence of the conduit axial thermal pre-compression on the critical reduction of the cable *I*c/*I*c0(εb,*B*) is still unknown although experiments on strand, swaged in a steel tube may give a direction. For this part of the model we rely on a hypothesis containing some uncertainty. On this issue additional effort is required asking for a more fundamental approach looking at the 3D strain behaviour of Nb3Sn layers in relation to strand geometries.

Strands with high stiffness (Young's modulus) should be selected for conductors subjected to the highest *IxB* load if the magnet design concept allows for this choice. Strands with lower modulo can be applied in sections of the magnets with lower transverse load. In the (experimental) parametric study on cabling parameters, the influence of a lower void fraction can be further investigated, in particular in view of the strand elongation, potentially decreasing the cable cross section.

When it is confirmed that the optimisation leads to a significant minimisation of the degradation, as predicted by the model, the ITER conductor design, can be adjusted to a layout with significant economical benefit.

The use of these results can lead to a considerable reduction in the cost of superconductors for ITER. In particular when the large conductor degradation can be largely balanced. In case the time schedule does not allow for a drastic change of the conductor design, with only a change in the cabling scheme, the margin is significantly improved.

# **7. Conclusions**

We developed a model that describes the design optimisation of cabled superconductors for transverse load (TEMLOP-Model) and report the first results of computations for an ITER type conductor leading to new insights, based on the measured mechanical and electromagnetic properties of an internal tin strand.

The most important conclusion of this paper is that the problem of the severe degradation of large CICCs, as designed for ITER, can be solved by increasing the cabling pitches. The principle is based on a more homogeneous distribution of the stresses and strains in the cable and significantly moderate the local peak stresses associated with short twist pitches.

The model gives a good description for the mechanical response of the strands to a transverse load, from layer to layer in the cable, based on the strand stiffness and cable void fraction. The computation is in good agreement with transverse stress-strain experiments on cables.

The simulations point out that the degradation is fully attributed to bending, although the transverse contact load on strand crossings and line contacts reaches up to 90 MPa.

We also find that a lower cable bundle void fraction and higher strand stiffness add to a further improvement of the conductor performance.

## **Acknowledgements**

"This work, supported by the European Communities under the contract of Association between EURATOM/FOM, was carried out within the framework of the European Fusion Development Agreement. The views and opinions expressed herein do not necessarily reflect those of the European Commission."

The proofreading of the manuscript by my colleagues M.J. Dhalle and A. den Ouden at the University of Twente is gratefully acknowledged.

## **References**

- [1] Ekin J W 1980 Proc. of the Topical Conf. on-A15-Supercond., Plenum, New York, NY, USA, **IB: 0306-40622**-**5** 187
- [2] Nijhuis A, van den Eijnden N, Ilyin Y, van Putten EG, Veening GJT, Wessel WAJ, den Ouden A, and ten Kate HHJ, 2005 *Supercond Sci Technol* **18** S273–S283
- [3] Mitchell N, 2005 *Supercond Sci Technol* **18** S396–S404
- [4] Mitchell N, 2004 *Physica C: Superconductivity* **401** (1-4) 28
- [5] Zanino R, Ciazynski D, Mitchell N, Richard LS. 2005 *Supercond Sc Technol* **18** S376-82
- [6] Schultz JH, Chiesa L, Harris DL, Lee PJ, Minervini JV; Senkowicz BJ, Takayasu M, Titus P, 2005 *IEEE Trans Appl Supercond* **15** 1371
- [7] Nunoya Y, Isono T, Okuno K, 2004 *IEEE Trans Appl Supercond* **14** 1468
- [8] Mitchell N, 2003 *Fusion Eng and Design* **66-68** 971
- [9] Pasztor G et al, 2004 *IEEE Trans Appl Supercond* **14** 1527
- [10] Mitchell N, 2006 'Assessment of Conductor Degradation in the ITER CS Insert Coil and Implications for the ITER Conductors', presented at the MEM06, July 3, 2006 Durham UK

- [11] Martovetsky NN, Bruzzone P, Stepanov B, Wesche R, Chen-yu Gung, Minervini JV, Takayasu M, Goodrich LF, Ekin JW, Nijhuis A, 2005 *IEEE Trans Appl Supercond* **15** 1367
- [12] Wessel WAJ Nijhuis A Ilyin Y Abbas W ten Haken B and ten Kate HHJ 2004 *Adv Cryog Eng (Materials)* **50** 466
- [13] Nijhuis A Ilyin Y Wessel WAJ Abbas W ten Kate HHJ, 2005 *IEEE Trans Appl Supercond* **14** 1464
- [14] Nijhuis A Wessel WAJ Knoopers HG Ilyin Y della Corte A Kate HHJ, 2005 *IEEE Trans Appl Supercond* **15** 3466
- [15] Nijhuis A Wessel WAJ Ilyin Y den Ouden A ten Kate HHJ, 2006 *Rev. Sci. Instrum.* **77** *054701*
- [16] Nijhuis A Ilyin Y ten Kate HHJ 'Spatial periodic bending and critical current of bronze and PIT Nb3Sn strand in steel tube' *IEEE Trans Appl Supercond* **17** in press
- [17] Nijhuis A Ilyin A Wessel WAJ 2006 'Spatial Periodic Contact Stress and Critical Current of a Nb3Sn Strand measured in TARSIS*'* Supercond. Sci. Technol. **19** in press
- [18] van den Eijnden NC Nijhuis A Ilyin Y Wessel W A J and ten Kate H H J 2005 *Supercond. Sci. Technol.* **18** 1523
- [19] Ilyin Y Nijhuis A van den Eijnden NC Wessel WAJ and ten Kate HHJ 2006 'Axial tensile stressstrain characterisation of a 36 Nb3Sn strands cable', *IEEE Trans. Appl. Supercond.* **16** *1249-1252*
- [20] Nijhuis A Ilyin Y et. al. 2006 'Critical current of ITER TFMC Nb3Sn strand under influence of periodic bending strain and contact stress*' Cryogenics* in press
- [21] Nijhuis A Ilyin Y ten Kate HHJ 2006 'Critical current and strand stiffness of three types Nb3Sn strand subjected to spatial periodic bending*'* Supercond Sci Technol **19** in press
- [22] Zani L Cloez H Tena M della Corte A Muzzi L di Zenobio A, 2005 *Supercon Science Technol*. **18** S390
- [23] Senkowicz BJ Takayasu M Lee PJ Minervini JV Larbalestier DC 2005 *IEEE Trans Appl Supercond* **15** 3470
- [24] Ulbricht A Duchateau JL Fietz WH Ciazynski D Ilyin Y Nijhuis A et al., 2005 *Fusion Eng and Design* **73/2-4** 189
- [25] Nijhuis A Ilyin Y Abbas W ten Haken B ten Kate HHJ, 2004 *IEEE Trans Appl Supercond* **14** 1489
- [26] Bruzzone P Wesche R Stepanov B, 2003 *IEEE Trans Appl Superc* **13** 1452
- [27] della Corte A, 2006 private communication
- [28] Godeke A Dhalle M Morelli A Stobbelaar L van Weeren H van Eck H J N Abbas W Nijhuis A den Ouden A and ten Haken B, 2004 *Rev of Scient Instr* **75** 5112
- [29] Mitchell N, 2005 *Cryogenics* **45** 501
- [30] Mitchell N, 2005 *IEEE Trans Appl Supercond* **15** 3572
- [31] Ekin J W, 1978 *J Appl Phys* **49** 3406
- [32] Specking W, Goldacker W and Flukiger R, 1988 *Adv Cryog Eng (Materials)* **34** 569
- [33] Ekin JW Bray SL Bahn WL, 1991 *J. Appl. Phys*. **69** 4436
- [34] Boschman H van de Klundert LJM, 1990 *Adv Cryog Eng* **36** 93
- [35] Ilyin Y Nijhuis A ten Kate HHJ 2006 'Interpretation of conduit voltage measurements on the Poloidal Field Insert Sample using the CUDI-CICC numerical code' *Cryogenics* **46** *517-529*
- [36] Nijhuis A Ilyin Y Abbas W ten Kate H H J Ricci M V and della Corte A, 2005 *IEEE Trans Appl Supercond* **15** 1633
- [37] Hamada K Takahashi Y Matsui K Kato T Okuno K, 2004 *Cryogenics* **44** 45
- [38] Nijhuis A Noordman NHW ten Kate HHJ Mitchell N Bruzzone P, 1998 'A Press for Mechanical and Electrical Testing of Full-size ITER Conductors under Transverse Loading', *Proc 15th Conf Magnet Techn*, Science Press, Beijing, China, 441
- [39] Nijhuis A Noordman NHW ten Kate HHJ Mitchell N and Bruzzone P 1999 'Electromagnetic and Mechanical Characterisation of ITER CS-MC Conductors Affected by Transverse Cyclic Loading Part 2: Interstrand Contact Resistances' IEEE Trans. Appl. Supercond **9** 754-757