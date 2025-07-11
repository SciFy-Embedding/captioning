# arXiv:2403.06389

**Paper ID:** e80cbee1e4e8f8e5103dd775d8d6feaa

**PDF Path:** apl/Superconductors/arXiv:2403.06389.pdf

**Processing Status:** complete

**Captions Added:** 13

**Generated:** 2025-06-24T13:44:27.987052

---

# arXiv:2403.06389v2 [cond-mat.mtrl-sci] 4 Jun 2024

# Suppression of flux jumps in high-J<sup>c</sup> Nb3Sn conductors by ferromagnetic layer

Cun Xue,1, [∗](#page-0-0) Kai-Wei Cao,<sup>1</sup> Tian He,<sup>2</sup> Chong Wei,<sup>1</sup> Wei Liu,3, [†](#page-0-1) and Jun-Yi Ge2, [‡](#page-0-2)

<sup>1</sup>School of Mechanics, Civil Engineering and Architecture,

Northwestern Polytechnical University, Xi'an 710072, China

<sup>2</sup>Materials Genome Institute, Shanghai University, Shanghai 200444, China

<sup>3</sup>Western Superconducting Technologies Co., Ltd.,

and Xi'an Superconducting Magnet Technology Co., Ltd, Xi'an 710014, China

(Dated: June 5, 2024)

Flux jumps observed in high-J<sup>c</sup> Nb3Sn conductors are urgent problems to construct high field superconducting magnets. The low-field instabilities usually reduce the current-carrying capability and thus cause the premature quench of Nb3Sn coils at low magnetic field. In this paper, we explore suppressing the flux jumps by ferromagnetic (FM) layer. Firstly, we experimentally and theoretically investigate the flux jumps of Nb3Sn/FM hybrid wires exposed to a magnetic field loop with constant sweeping rate. Comparing with bare Nb3Sn and Nb3Sn/Cu wires, we reveal two underlying mechanisms that the suppression of flux jumps is mainly attributed to the thermal effect of FM layer for the case of lower sweeping rate, whereas both thermal and electromagnetic effects play a crucial role for the case of higher sweeping rate. Furthermore, we explore the flux jumps of Nb3Sn/FM hybrid wires exposed to AC magnetic fields with amplitude Ba<sup>0</sup> and frequency f. We build up the phase diagrams of flux jumps in the plane f-Ba<sup>0</sup> for bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire, respectively. We stress that the region of flux jumps of Nb3Sn/FM wire is much smaller than the other two wires, which indicates that the Nb3Sn/FM wire has significant advantage over merely increasing the heat capacity. The findings shed light on suppression of the flux jumps by utilizing FM materials, which is useful for developing new type of high-J<sup>c</sup> Nb3Sn conductors.

### I. INTRODUCTION

With high critical current density (non-matrix J<sup>c</sup> exceeds 3000 A/mm<sup>−</sup><sup>2</sup> at 4.2 K, 12 T) and upper critical field (28-30 T) [\[1](#page-8-0)[–4\]](#page-8-1), Nb3Sn conductor is a preferred material to construct high magnetic field (10-16 T) that is beyond the limit of NbTi [\[5\]](#page-8-2). Additionally, a hybrid solution of high temperature superconductor (HTS) and Nb3Sn materials is usually used to achieve much higher magnetic field (> 20 T) in order to avoid significantly high cost of HTS [\[6\]](#page-8-3). However, Nb3Sn magnets are susceptible to the flux jumps [\[7–](#page-8-4)[9\]](#page-9-0), which seriously limits the Nb3Sn performance. After magnetic flux jumps were first observed and investigated in 1960s [\[10\]](#page-9-1), the underlying physical mechanisms [\[11–](#page-9-2)[14\]](#page-9-3) and the relevant physical parameters controlling the thermomagnetic instability (temperature [\[15\]](#page-9-4), ramping rate [\[16\]](#page-9-5), sample size [\[17\]](#page-9-6), border defects [\[18\]](#page-9-7)) were gradually revealed. For composite superconducting wires, the early criterion for triggering the magnetic flux jump was proposed by Swartz & Bean [\[19\]](#page-9-8) and Wilson [\[20\]](#page-9-9). Subsequently, in order to further improve the performance of Nb3Sn high-field magnets, researchers conducted a series of experimental studies [\[21–](#page-9-10)[23\]](#page-9-11) and theoretical analyses [\[24–](#page-9-12)[26\]](#page-9-13) to describe the characteristics of low-field flux jumps. During the flux jumps, substantial energy is released in the form of Joule heating within a very short time due to the rapid motion of magnetic flux trapped inside the filaments. In this case, the flux instability behaviors usually lead to undesirable problem of premature quenches [\[27\]](#page-9-14).Experimental results demonstrated

that the superconducting wires that carry high currents at high fields (greater than 10 T) cannot sustain these same currents at low fields (1–3 T) when the sample current is fixed and the magnetic field is ramped [\[28\]](#page-9-15). Moreover, frequent and sudden spikes in voltage signals sometimes are very high, which can cause difficulties for the digital quench detection system [\[29\]](#page-9-16).

In recent years, researchers have shown more and more academic interests in exploring effective methods to prevent the flux instabilities of Nb3Sn. Experiments by Fermi national accelerator laboratory [\[30,](#page-9-17) [31\]](#page-9-18) and Brookhaven national laboratory [\[32,](#page-9-19) [33\]](#page-9-20) show that both reducing the filament size (def f ) and increasing residual resistivity ratio (RRR) of the copper can significantly improve stability of Nb3Sn at low field, which can indeed improve Nb3Sn conductor performance and current carry ability. On one hand, more subelements can reduce def f . On the other hand, however, it may also easily cause the problems of contamination and thus raise manufacturing difficulties [\[34\]](#page-9-21). The higher RRR can be achieved by reducing the reaction time or by modifying the design with adjusting the metal ratio of Nb and Sn, which is at the cost of reducing the critical current density [\[35\]](#page-9-22). Wilson et al demonstrated that the intrinsic stability of high-J<sup>c</sup> Nb3Sn conductors can be improved by adjusting the temperature margin and the dynamic cooling [\[21\]](#page-9-10). However, further increasing the temperature margin beyond a certain limit becomes increasingly challenging for Nb3Sn utilized in generating high fields. Condisering Nb3Sn coils are usually impregnated with epoxy, the quite small thermal diffusion coefficient may weaken the effectiveness of dynamic cooling. The experiments by Goldfarb et al [\[36\]](#page-9-23) showed that the number of flux jumps decreases remarkably when the high-J<sup>c</sup> Nb3Sn wire exposed to liquim helium at 2.1 K, which indicates that improvement of heat transfer capac-

<span id="page-0-0"></span><sup>∗</sup> [xuecun@nwpu.edu.cn](mailto:xuecun@nwpu.edu.cn)

<span id="page-0-1"></span><sup>†</sup> [liuwei17@c-wst.com](mailto:liuwei17@c-wst.com)

<span id="page-0-2"></span><sup>‡</sup> [junyi˙ge@t.shu.edu.cn](mailto:junyi_ge@t.shu.edu.cn)

ity indeed can suppress the low-field instabilities. This method needs very costly lower operation temperature. Additionally, it may not suitable for Nb3Sn magnets with densely packed coils where heat dissipation conditions of inner Nb3Sn wires is completely different from the sample of single Nb3Sn wire in experiments. Very recently, Xu et al [\[37\]](#page-9-24) investigated the influences of heat treatment temperature and Ti-dopping on the low-field flux jumps. It was found that the suppression of low-field flux jumps by heat treatment temperature is attributed to both reducing the Jc(B) curve slope and increasing the heat capacity. The negative result is a significant reduction of J<sup>c</sup> at low fields. Another promising idea to improve the thermomagnetic stability stems from increasing the specific heat. Xu et al proposed to introduce high specific heat substances (Gd2O3) in certain positions [\[38\]](#page-9-25). The experiments show that although many small flux jumps are still observed at low fields the amplitude of flux jumps can be indeed suppressed significantly. Therefore, it is still desirable to explore new methods to suppress the flux jumps.

It is well-known that the superconductors and ferromagnetic (FM) materials exhibit completely different electromagnetic behaviors. When coupling the two materials, it was expected to observe interesting phenomena in hybrid superconductors/ferromagnetic (SC/FM) structures. Many researchers developed theoretical and experimental approaches to explore the impacts of ferromagnetic substrate on the electromagnetic responses of superconductors. The complex-field approach with conformal mapping was used to analyze the effect of soft ferromagnetic substrate on the electromagnetic response of superconducting strips [\[39\]](#page-9-26) and ac loss of 2G HTS power transmission cables [\[40\]](#page-9-27). Recently, Prigozhin et al proposed a thin shell model and numerical scheme of Chebyshev spectral method for electromagnetic response of a coated conductor with magnetic substrate to variations of the transport current and applied field. The influence of a magnetic substrate on the superconducting current and ac losses is investigated[\[41\]](#page-9-28). Silhanek et al developed a quantitative magneto-optical imaging technique to directly visualize the thermomagnetic instabilities of superconductor/ferromagnet hybrid structures [\[42\]](#page-9-29). Another fascinating phenomenon is the so-called magnetic cloak to exactly cloak uniform static magnetic fields by specially designed superconductor-ferromagnetic structures [\[43,](#page-9-30) [44\]](#page-9-31). Due to the electromagnetic interactions, it is expected that the FM materials have clear impact on the low-field instabilities of Nb3Sn wires, which has not been well explored yet.

In this paper, we explore the thermomagnetic instabilities of Nb3Sn/ferromagnetic (Nb3Sn/FM) hybrid wires exposed to time-varying magnetic field. The magnetization measurements show that the flux jumps of Nb3Sn wires can be significantly suppressed by the FM layer. Combining with numerical simulations, we reveal different underlying mechanisms for suppression of flux jumps in the cases of low and high magnetic field sweeping rates, respectively. Then we further explore the flux jumps of Nb3Sn/FM hybrid wire exposed to the AC magnetic field. We demonstrate that the flux jumps depends on both

![](_page_1_Figure_4.jpeg)

**Caption:** Figure 1 showcases illustrative schematics of a bare Nb3Sn wire and a Nb3Sn wire with an FM strip. Panels (b) and (c) represent the setup used for magnetization tests, where wires are exposed to an external magnetic field with a sweeping rate of 0.02 T/s. These diagrams establish the physical model and experimental parameters for studying flux jumps, foundational for analyzing FM layers' impacts on thermomagnetic stability.


FIG. 1. (a) Bare Nb3Sn wire with 54 subelements (left) and hybrid-structured Nb3Sn wire wrapped by FM strip (Nb3Sn/FM wire, right). (b-c) The samples are exposed to an applied magnetic field with sweeping rate of 0.02 T/s.

amplitude Ba<sup>0</sup> and frequency f of AC magnetic field. The phase diagram of flux jumps in the plane f-Ba<sup>0</sup> for bare Nb3Sn wire, Nb3Sn/cooper (Nb3Sn/Cu) wire and Nb3Sn/FM wire are presented. The paper is organised as follows: we present the experimental measurements and theoretical method that are used to study the flux jumps of Nb3Sn/FM wires in Section II. Then we discuss the suppression of flux jumps for two cases of a linear ramping magnetic field and AC magnetic filed. The summary is given in the final section.

# II. EXPERIMENTS SETUP AND THEORETICAL FORMALISM

To prepare the samples used in the experimental measurements, we choose the multi-filamentary Nb3Sn wire manufactured with the so-called internal-tin approach. The diameter of Nb3Sn wire is about 1.3 mm and its subelement structure is shown in Fig. 1(a). In this work, two kinds of samples are fabricated, i.e., bare Nb3Sn wire (see left panel in Fig. 1(a)) and Nb3Sn wire warped by a FM layer made of 1J22 soft magnetic alloy (right panel in Fig. 1(a)). The data of the samples are shown in the Appendix. We prepare three samples of hybrid Nb3Sn/FM wires and the thickness of the FM layer is 0.1 mm, 0.2mm and 0.3 mm, respectively. A polishing procedure is performed for both ends of all samples. In real engineering applications, the Nb3Sn wire is generally exposed to an external magnetic field and transport current with ramping rates normally less than 5A/s. Actually, simulation results show that the effect of transport current on number of flux jumps and amplitude of flux jumps is very slight when ramping rates of transport current is less than 5 A/s [\[45\]](#page-9-32).

![](_page_2_Figure_0.jpeg)

**Caption:** Figure 2 presents the experimental (blue) and simulated (orange) magnetization (M) versus applied magnetic field (Ba) for Nb3Sn wire wrapped with an FM layer, plotted in part (a). Observations indicate a smooth magnetization curve with reduced flux jumps due to the FM layer's stabilizing effects. In part (b), the relative permeability (μr) of the FM material versus Ba shows a steep decline after 0.25 T, suggesting the FM layer's decreasing influence in high magnetic fields due to saturation, which is crucial for understanding its segregation of effects under varying field intensities.


FIG. 2. (a) The magnetization loop of the FM layer obtained by MPMS measurements. The fitting curve (blue) is in good agreement with the experimental data. (b) The relative permeability variations with applied magnetic field used in the numerical simulations.

Therefore, in order to detect the thermomagnetic instabilities, we perform the magnetization measurements for all samples only exposed to a transverse magnetic field loop with Magnetic Property Measurement System (MPMS) at T<sup>0</sup> = 4.2 K. Note that a zero-field cooling down from 20 K to 4.2 K is first performed before each magnetization measurements. As shown in Fig. 1(b), the magnetic field is varying between +3 T and -3 T with a constant field-sweeping rate of 20 mT/s, i.e., B<sup>a</sup> = 0 T → 3 T → -3 T → 3 T.

The maximum field-sweeping rate of MPMS is 0.02 T/s. In this case, we were not able to carry out the experiments with higher field-sweeping rates. Actually, the experimental results are mainly used to verify and confirm the validation of our numerical model and the parameters used in the simulations. Alternatively, to deeply understand the effects of FM layer on the thermomagnetic instabilities of Nb3Sn wire as well as to explore more complicated cases, as schematically shown in Fig. 1(c), we consider a 2D numerical model in the numerical simulations. With the Hformula [\[46\]](#page-9-33), the electromagnetic responses of bare Nb3Sn wire and Nb3Sn/FM wire exposed to a time-varying transverse magnetic field can be obtained by solving the following equations,

$$\begin{aligned} \mu\_0 \left(\mu\_r \left(H\right) + H\_x^2 f\left(H\right)\right) \frac{\partial H\_x}{\partial t} + \mu\_0 H\_x H\_y f\left(H\right) \frac{\partial H\_y}{\partial t} \\ + \rho \frac{\partial J\_z}{\partial y} = 0 \end{aligned} (1)$$

$$\begin{aligned} \mu\_0 \left(\mu\_r \left(H\right) + H\_y^2 f\left(H\right)\right) \frac{\partial H\_y}{\partial t} + \mu\_0 H\_x H\_y f\left(H\right) \frac{\partial H\_x}{\partial t} \\ -\rho \frac{\partial J\_z}{\partial x} = 0 \end{aligned} (2)$$

$$\frac{\partial H\_y}{\partial x} - \frac{\partial H\_x}{\partial y} = J\_z \tag{3}$$

where µ<sup>0</sup> is the vacuum permeability, µ<sup>r</sup> (H) the relative permeability and ρ the resistivity. Hx(x, y) and Hy(x, y) are the components of local magnetic field and thus the total magnetic field H = q H<sup>2</sup> <sup>x</sup> + H<sup>2</sup> y . f (H) = dµ<sup>r</sup> (H) / (HdH).

For the soft 1J22 FM layer, the relationship can be obtained by the magnetization measurements as shown in Fig. 2(a). One can see that a fitting curve agrees with the experimental results quite well. Therefore, similar to Ref [\[47\]](#page-9-34), the following relative permeability is employed in the simulation for the ferromagnetic material,

$$
\mu\_r\left(H\right) = 1 + 1200000\left(1 - \exp\left(-\left(H/70\right)^{3.2}\right)\right)H^{-0.99} \tag{4}
$$

The constitutive relationship between current density and electric field is given as

$$\mathbf{E} = \rho \mathbf{J} \tag{5}$$

The law of resistance in the superconducting region can be written as [\[48\]](#page-9-35)

$$
\rho\_{SC} = \frac{\rho\_0 \rho\_{SC0}}{\rho\_0 + \rho\_{SC0}} \tag{6}
$$

$$
\rho\_{SC0} = \frac{E\_c}{J\_c} \left(\frac{|J|}{|J\_c|}\right)^{n-1} \tag{7}
$$

where ρ<sup>0</sup> is a constant resistivity, E<sup>c</sup> the critical electric field. The critical current density J<sup>c</sup> is generally dependent on the temperature and local magnetic field, which can be calculated using the formulae in Appendix. The flux creep exponent n is taken as

$$n = n\_0 \left( 1 - \left(\frac{T}{T\_c}\right)^4 \right) \left( 1 - \frac{B}{B\_{c2}} \right) \frac{T\_c}{T} \tag{8}$$

where n<sup>0</sup> is the flux creep exponent at working temperature T0, and T<sup>c</sup> is the critical temperature, Bc<sup>2</sup> is the upper critical magnetic field. The magnetic field B is taken as

$$B = \mu\_0 \mu\_r H \tag{9}$$

In order to study the thermomagnetic instability, the electromagnetic equations should be solved by coupling the heating diffusion equations. The heat transport in the system is governed by the equation,

$$c\frac{\partial T}{\partial t} = \nabla \cdot (\kappa \sqcap T) + E \cdot \mathbf{J} \tag{10}$$

where E·J is the Joule heating source. The heat diffusion equation can be solved by considering the following heat exchange boundary condition

$$-\kappa \left( \bigtriangledown T \cdot \mathbf{n} \right) = h \left( T - T\_0 \right) \tag{11}$$

where c, κ, h are the specific heat, thermal conductivity and heat transfer coefficient, respectively. The thermal parameters are assumed to be proportional to T 3 , i.e., c = c<sup>0</sup> (T /T0) 3 , κ = κ<sup>0</sup> (T /T0) 3 , h = h<sup>0</sup> (T /T0) 3 [\[49\]](#page-9-36).

The coupling equations above can be solved by PDE mode in Comsol Multiphysics [\[50,](#page-9-37) [51\]](#page-9-38). In our numerical simulations, the electromagnetic parameters are Jc<sup>0</sup> = 4 × 10<sup>10</sup> A · m<sup>−</sup><sup>2</sup> , T0=4.2 K, Tc=18.2 K, n<sup>0</sup> = 15, E<sup>c</sup> = 1 × 10<sup>−</sup><sup>4</sup> V/m, ρair = 1 Ω · m, ρSC = 2.5 × 10<sup>−</sup><sup>7</sup> Ω · m, ρCu = 3 × 10<sup>−</sup><sup>10</sup> Ω · m, ρFM = 9 × 10<sup>−</sup><sup>7</sup> Ω · m. After calculation: the temperature of Nb3Sn is the same as that of Cu, the thermal conductivity of Nb3Sn has no effect on the simulation results, and the thermal conductivity of Nb3Sn and Cu is the same. The thermal parameters are κSC<sup>0</sup> = 500 W/ (m · K), κCu<sup>0</sup> = 500 W/ (m · K), κFM<sup>0</sup> = 110 W/ (m · K), cSC<sup>0</sup> = 560 J/ K · m<sup>3</sup> , cCu<sup>0</sup> = 560 J/ K · m<sup>3</sup> , cFM = 500 J/ K · m<sup>3</sup> [\[52](#page-9-39)[–55\]](#page-9-40).

### III. RESULTS AND DISCUSSIONS

## A. suppression of flux jumps in Nb3Sn/FM wire exposed to a linear ramping magnetic field loop

We first start with discussing the flux jumps induced by thermomagnetic instabilities of Nb3Sn/FM wire exposed to a magnetic field loop, i.e., 0 T → 3T → -3 T → 3 T with a constant field sweeping rate. Fig. 3(a-d) represent the experimental magnetization curves of bare Nb3Sn wire, Nb3Sn/FM wire with FM layer thickness of 0.1-0.3 mm, respectively. The sample is in the cold gas helium and the width of the gap between FM layer and Nb3Sn wire is about 0.07 mm-0.16 mm during experimental measurements. In order to simulate the partial connection between FM layer and Nb3Sn wire, we assumed that the small gap is filled up with cold gas helium in the numerical model. The thermal parameters of gas helium can be seen in Ref [\[56\]](#page-9-41). Numerical simulations indicated that the temperature is almost the same on the whole crosssection before flux jumps occur. This means that the heat exchange between Nb3Sn wire and FM layer (or Cu layer) is good enough. In this work, we mainly focuse on the flux jumps, i.e., the sharp peaks observed in the magnetization curves. As such, the magnitude of magnetization is not important and the magnetization is normalized by the maximum value M<sup>0</sup> for the convenient comparison with different samples. By comparing with bare Nb3Sn wire (Fig. 3(a)), we can see a most distinct characteristic of hybrid Nb3Sn/FM wire (Fig. 3(b-d)) that the magnetization curves can only be observed in the first and third quadrants. This is attributed to the strong paramagnetism of 1J22 FM materials and thus the total magnetization M is reversed. In addition, by comparing the normalized magnetization curves in the four cases, it can be seen that the jump amplitude of Nb3Sn/FM wire is smaller than that of the Nb3Sn wire. Moreover, amplitude of flux jumps is smaller and smaller with increasing thickness of ferromagnetic layer. Importantly, as shown in Fig. 3(a), frequent flux jumps are observed in bare Nb3Sn wire. However, for the Nb3Sn wire wrapped by FM layer with thickness of 0.1 mm, the number of the peaks decreases and the flux jumps are suppressed, especially for the decreasing branches of magnetic fields. Moreover, it is surprising to note that the magnetization curves become quite smooth and only 4 flux jumps observed in hybrid Nb3Sn/FM wire with FM thickness of 0.3 mm, which indicates that the thick FM layer can significantly suppress the undesired flux jumps triggered in Nb3Sn wire.

On one hand, it is apparent that suppression of flux jumps by wrapping FM layer should first be attributed to the thermal effect because it increases the total heating capacity, which definitely enhances the thermomagnetic stability of Nb3Sn wire exposed to a ramping magnetic fields. The main origin of the idea is based on the well-known fact that the flux jumps are thermomagnetic coupling problems. As a matter of fact, it had been demonstrated that power-doping with high specific heating capacity in Nb3Sn wire, such as Gd2O3, is an effective way to suppress the flux jumps [\[38\]](#page-9-25) . On the other hand, one could possibly argue that, besides the indirect effect, the FM layer should also have an extra impact on the flux jumps due to its electromagnetic effect. The main argument behind this belief leans on the fact that FM layer is a typical paramagnetic material and thus can directly affect the electromagnetic responses of the Nb3Sn wire. In order to clarify the underlying physics, we performed numerical simulations for the bare Nb3Sn wire and hybrid Nb3Sn/FM wire exposed to the same magnetic field loop with the aforementioned magnetization experiments. As shown in Fig. 3(e-h), the simulated magnetization of bare Nb3Sn wire and Nb3Sn/FM wire can reproduce the experimental results quite well. For Nb3Sn/FM system, both experimental and simulated results demonstrate that the flux jumps are triggered at the increasing branches (0 → ±3) T more frequently than at the decreasing branches (±3 → 0) T.

To further explore the thermal effect and electromagnetic effect of FM layer on suppression of flux jumps, as illustrated in Fig. 4(a), we calculate the magnetization of Nb3Sn wire with Cu layer which has the same structure size with Nb3Sn/FM wire. Although the Cu can lead to modifications of the current streamlines over the superconducting wire after quench, it is worth noting that, as shown in Fig. 3, the magnetization of Nb3Sn wire does not decrease to zero during the flux jumps and thus Nb3Sn wire does not quench yet. So the induced current still flows in the Nb3Sn filaments rather than in cooper during the flux jumps and this is also verified by our numerical

![](_page_4_Figure_1.jpeg)

**Caption:** Figure 3 presents both experimental and simulated magnetization curves for various configurations of Nb3Sn wires subjected to a transversal magnetic field. Panels (a) to (d) show magnetization measurements at 4.2 K for bare Nb3Sn wires and those wrapped with FM layers of varying thicknesses (0.1 mm to 0.3 mm), revealing that the FM layers reduce the number and amplitude of flux jumps. The simulated results (e) to (h) mirror these trends, affirming that thicker FM layers significantly suppress flux jumps due to enhanced thermal stability, demonstrating the FM layer's dual role in thermal and electromagnetic stabilization.


FIG. 3. (a-d) Experimental magnetization curves of bare Nb3Sn wire (sample A) and hybrid Nb3Sn wire wrapped by FM layer with thickness of 0.1-0.3 mm (sample B-D) exposed to a transversal magnetic field with a ramping rate of 0.02 T/s by MPMS measurements at 4.2 K. (e-h) Simulated magnetization for sample A-D under the same parameters with experiments. The red curves denote the magnetization during the process of 3 T to −3 T. Similarly, the blue curves denote the magnetization during the process of −3 T to 3 T.

![](_page_4_Figure_3.jpeg)

**Caption:** Figure 4 features a schematic comparison between Nb3Sn wires with FM and copper layers, and associated magnetization curves. Panel (a) presents design schematics for the numerical simulations, while panel (b) shows magnetization characteristics for Nb3Sn/Cu wires across a range of copper layer thicknesses (0.1-0.6 mm). The results demonstrate that increasing copper thickness enhances flux jump suppression, primarily through thermal capacity expansion. Panel (c) quantitatively relates the number of observed flux jumps to the thickness of Cu and FM layers, elucidating the layers' respective contributions to superconducting stability enhancements.


FIG. 4. (a) Schematic diagrams of two same-sized Nb3Sn/FM wire (left) and Nb3Sn/Cu wire (right) used in the numerical simulations. (b) The magnetization curves of Nb3Sn/Cu wire with Cu thickness of 0.1-0.6 mm. (c) The number of flux jumps (NFJ) as a function of the thickness of Cu and FM layer.

simulations. In this case, the Cu layer only exhibits thermal effect on the flux jumps while its electromagnetic effect can be neglected. Fig. 4(b) shows the magnetization of Nb3Sn/Cu wire with Cu layer thickness of 0.1-0.6mm. It can be seen that the number and amplitude of the flux jumps of the Nb3Sn/Cu wire decrease with increasing the thickness of the Cu layer. Particularly, the flux jumps of the hybrid structure can be completely suppressed when the thickness of Cu layer is increased up to 0.6 mm. Fig. 4(c) indicates that the flux jumps are less likely to occur when both increasing the thickness of Cu layer and FM layer. Additionally, the Cu layer exhibits a slightly more powerful suppression of flux jumps due to its larger specific heating capacity of Cu as mentioned in section II. To conclude, the suppression of flux jumps by FM layer is mainly attributed to its thermal effect, while its electromagnetic effect is ineffective.

0.0 0.2 0.4 0.6 0.8 SC/FM SC/Cu In order to deeply understand the phenomenon that seems to be against the common sense, Fig. 5 depicts the magnetic flux lines and the induced current density distributions over the bare Nb3Sn wire, Nb3Sn/FM wire and Nb3Sn/Cu wire exposed to a lower and higher magnetic fields, respectively. In the following text, the thicknesses of FM layer and Cu layer are both fixed to be 0.3 mm in all numerical simulations unless otherwise stated. For the case of lower magnetic field, it can be found that the induced current density in bare Nb3Sn wire and Nb3Sn/Cu wire are nearly the same, which are much higher than that in Nb3Sn/FM wire. Moreover, most of magnetic flux lines pass through the FM layer, which demonstrates a significant magnetic shielding effect. However, the Cu layer has no impact on the distributions of magnetic flux lines. In contrast to the low magnetic field, the impact of FM layer on the electromagnetic responses of Nb3Sn becomes much weaker in the case of higher magnetic fields. As shown in Fig. 2, the experimental measurements show that the magnetization of 1J22 FM material is saturated and its relative permeability decreases rapidly with increasing magnetic field when B<sup>a</sup> > 0.25 T. As a consequence, in the case of small ramping rate B˙ a, the ferromagnetic layer has nearly no extra advantage over increasing the ratio of cooper to Nb3Sn due to the rapid decrease of relative permeability of ferromagnetic material at higher magnetic fields.

![](_page_5_Figure_1.jpeg)

**Caption:** Figure 5 depicts the simulated distributions of magnetic flux lines and induced current densities over three configurations of Nb3Sn wires: bare Nb3Sn wire (a, d), Nb3Sn wire with a copper layer (b, e), and Nb3Sn wire with a ferromagnetic (FM) layer (c, f). The left panels present results at a low magnetic field (Ba = 0.2 T) and the right panels at a high field (Ba = 3.0 T). At low fields, the FM layer effectively shields magnetic flux and reduces current density within the superconducting core, demonstrating its magnetic shielding capacity. This shielding effect diminishes at higher fields, where the FM material's permeability decreases due to approaching saturation. These results highlight the FM layer's role in enhancing the thermomagnetic stability through magnetic shielding, significantly at low field conditions.


FIG. 5. The simulated magnetic flux lines and the induced current density distributions over bare Nb3Sn wire (a, d), Nb3Sn/Cu wire (b, e) and Nb3Sn/FM wire (c, f) exposed to Ba= 0.2 T (left panels) and 3.0 T (right panels).

Now we explore the case of three kinds of superconducting wires exposed to a magnetic field loop with a higher field-sweeping rate. Fig. 6 shows simulated magnetization of bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire with B˙ <sup>a</sup>=0.1 T/s. Counting the peaks in magnetization curves, one can see that the flux jumps of Nb3Sn/FM wire is less than that of Nb3Sn/Cu wire, which indicates that the FM layer has better suppression of flux jumps than Cu layer for higher field-ramping rate. It seems that the electromagnetic effect of FM layer on Nb3Sn wire is enhanced for the cases of higher field-ramping rate. Note that it is not an accident because the similar results can also be observed by varying the heating transfer coefficient as shown in Fig. 6(b). In order to extract the electromagnetic effect of FM layer on the flux jumps, we remove the heating effect in the numerical simulations. As shown in Fig. 7, the mere electromagnetic effect of FM layer can indeed suppress the flux jumps partially. One may suppose that the suppression of flux by ferromagnetic layer depends on the field-ramping rate. Fig. 7 represents the number of flux jumps in the bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire with increasing the field-ramping rate of applied magnetic field B˙ <sup>a</sup>. The results demonstrate that flux jumps are more likely triggered in the three kinds of Nb3Sn wires with increasing B˙ <sup>a</sup>. It is interesting to note that the flux jumps triggered in Nb3Sn/FM wire is

![](_page_5_Figure_4.jpeg)

**Caption:** Figure 6 illustrates simulated magnetization characteristics under varying field sweep rates for bare Nb3Sn wire, Nb3Sn wire with a copper layer, and Nb3Sn with an FM layer. The higher ramping rates reveal the FM layer's enhanced suppression of flux jumps compared to the other materials, indicating that electromagnetic effects of the FM material become more influential at increased sweep rates, with significant implications for high-field applications.


FIG. 6. (a-b) The simulated magnetization of bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire exposed to a magnetic field with a higher field-sweeping rate of B˙ <sup>a</sup>=0.1 T/s for h = 4 and 8 W/ K · m<sup>2</sup> , respectively. The thickness of Cu and FM layers are 0.3 mm. For comparison, both cases of the Nb3Sn/FM wire with/without thermal effect are presented.

![](_page_5_Figure_6.jpeg)

**Caption:** Figure 7 charts the number of flux jumps in bare Nb3Sn wire, Nb3Sn/Cu, and Nb3Sn/FM wire during a magnetic field loop with different ramping rates. The FM layer's more significant suppression of flux jumps in comparison to the Cu layer as ramping rates increase highlights its dual thermal and electromagnetic stabilizing effects, crucial for applications needing reliable superconductivity at varying field dynamics.


FIG. 7. The number of flux jumps in bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire triggered during one applied magnetic field loop with different ramping rate.

![](_page_6_Figure_1.jpeg)

**Caption:** Figure 8 illustrates the time-varying AC magnetic field profile and the resulting temperature evolution in a Nb3Sn wire, as used in numerical simulations. The study utilizes a sinusoidal AC magnetic field to investigate thermomagnetic behavior, showing that induced temperature changes within the wire are closely tied to the AC field's dynamics. This analysis is critical for evaluating how AC fields influence thermal management of Nb3Sn/FM wires in practical superconducting applications.


FIG. 8. The variation of AC magnetic field with time used in the numerical simulations and the time evolutions of simulated temperature of Nb3Sn wire.

less than the Nb3Sn/Cu wire in the case of higher fieldramping rate. This demonstrates that advantage of FM layer over the Cu layer becomes more and more apparent with increasing B˙ <sup>a</sup> although the FM layer and Cu layer have nearly the same suppression of flux jumps for small B˙ a.

# B. Suppression of flux jumps in Nb3Sn/FM wire exposed to an AC magnetic field

In this subsection, we further explore the suppression of flux jumps in Nb3Sn wire exposed to an AC magnetic field by FM layer. Due to the magnetization loss of ferromagnetic materials under AC magnetic fields, local heating may occur. However, we choose soft ferromagnetic material. Unlike hard ferromagnetic material, the soft ferromagnetic material has very small magnetization loop (see experimental magnetization of only ferromagnetic layer in Fig. 2). Furthermore, we also calculated the magnetization loss of ferromagnetic layer with the numerical simulations. As shown in Fig. A1 in the Appendix, the magnetization loss of ferromagnetic layer is very small, which can be neglected. Unlike the cases of a constant field ramping rate in subsection A, the main characteristic of AC magnetic field is that the field ramping rate always varies with time, and it also depends on the present magnetic field. As shown in Fig. 8, we consider an AC magnetic field with sinusoidal variations, i.e., Ba(t) = Ba0sin(2πf t). The frequency f and the amplitude Ba<sup>0</sup> of the AC magnetic field can be tunable in the numerical simulations. Although the signals of flux jumps are not strictly periodic with AC magnetic field, it can be found that the number of simulated flux jumps are nearly the same in each period except the initial increasing branch. Therefore, we count the flux jumps within a period of AC magnetic field after the initial increasing branch (see colored box in Fig. 8).

We first explore a simple case of AC magnetic field by fixing the frequency and amplitude. Fig. 9(a) shows the time evolutions of temperature in bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire exposed to an AC magnetic field with f = 0.5 Hz and Ba<sup>0</sup> = 0.75 T. The averaged ramping rate of magnetic field is about 1.5 T/s, which is much greater than the field-sweeping rates in-

![](_page_6_Figure_7.jpeg)

**Caption:** Figure 9 highlights temperature and current density dynamics in Nb3Sn wires exposed to AC magnetic fields. Part (a) exhibits temperature changes over time among different wire types, while parts (b) to (d) show current density distributions when Ba(t) = 0.75 T. The diagram emphasizes the FM layer's role in reducing temperature variations and modifying current distributions, ultimately showcasing its effectiveness in preventing flux jumps by enhancing thermal and electromagnetic stability.


FIG. 9. (a) The time evolution of temperature in bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire exposed to an AC magnetic field with frequency f = 0.5 Hz and Ba<sup>0</sup> = 0.75 T. (b-d) The current density distributions over the three kinds of wires when Ba(t) = 0.75 T. (e-f) The local magnetic field evolutions with time at point 1 and 2 indicated in (b-d). The dashed black curves are the variations of AC magnetic field with time.

vestigated above. By noting that flux jumps triggered in Nb3Sn/Cu wire is less than that in bare Nb3Sn wire, one can see that the Cu layer can partially suppress the flux jumps, which is similar to the cases in subsection A. By contrast, it is surprising that the FM layer can completely eliminate the flux jumps. To understand the difference of thermomagnetic instabilities in three kinds of Nb3Sn wires exposed to AC magnetic field, Fig. 9(bd) represents the current density distributions over bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire when Ba(t) = 0.75 T, respectively. It can be observed that both bare Nb3Sn wire and Nb3Sn/Cu wire are completely penetrated by magnetic flux. However, the magnetic flux just partially penetrates the Nb3Sn/FM wire. The maximum current density of Nb3Sn/FM wire is greater than the bare Nb3Sn wire and Nb3Sn/Cu wire because flux jumps occur before Ba(t) = 0.75 T in the latter two cases. Additionally, one may expect to find more evidence from the local magnetic field to interpret the suppression of flux jumps by FM layer. Fig. 9(e-f) shows the time evolutions of two representative local magnetic field at point 1 and 2 indicated in Fig. 9(b-d). It is evident that the local magnetic field BSC/FM at point 1 and 2 is always much less than BSC and BSC/Cu. Therefore, the FM layer decreases not only

![](_page_7_Figure_1.jpeg)

**Caption:** Figure 10 illustrates the suppression of flux jumps in superconductor (SC) wires, comparing three configurations: bare Nb3Sn wire (SC), Nb3Sn wire with a copper layer (SC/Cu), and Nb3Sn wire with a ferromagnetic (FM) layer (SC/FM). Part (a) shows the number of flux jumps as a function of frequency (f) for an AC magnetic field with an amplitude fixed at Ba^0 = 0.75 T, demonstrating that the FM layer effectively eliminates flux jumps across all frequencies, while jumps are observed in SC and SC/Cu wires. Parts (b) and (c) plot temperature evolutions for f = 0.5 Hz and 10 Hz, respectively, revealing higher temperature ranges in SC and SC/Cu wires compared to SC/FM, suggesting FM's superior thermal management. Part (d) shows the effect of the magnetic field amplitude on flux jumps at a fixed frequency of 0.5 Hz, where flux jumps begin to occur in SC and SC/Cu wires above 0.3 T, but not until higher amplitudes in SC/FM wires. These findings support the FM layer's role in enhancing thermal stability and suppressing unwanted magnetic instabilities.


FIG. 10. (a) The number of flux jumps as a function of frequency f in bare SC wire, SC/Cu wire and SC/FM wire exposed to AC magnetic field by fixing Ba<sup>0</sup> = 0.75 T. (b-c) The time evolutions of temperature in three kinds of wires for f = 0.5 Hz and 10 Hz, respectively. (d)The number of flux jumps as a function of amplitude of ac magnetic field by fixing f = 0.5 Hz.

the local magnetic fields, but also the field ramping rate.

For Ba<sup>0</sup> = 0.75 T, Fig. 10(a) indicates that the flux jumps are completely eliminated by the FM layer for all frequencies of AC magnetic field, while the flux jumps are observed in bare Nb3Sn wire and Nb3Sn/Cu wire. Furthermore, the number of flux jumps triggered in both bare Nb3Sn wire and Nb3Sn/Cu wire is not a monotonous function with frequency of AC magnetic field. This can be interpreted by the temperature. Fig. 10(b-c) shows the time evolutions of temperature in three kinds of wires for f = 0.5 Hz and f = 10 Hz, respectively. By taking the Nb3Sn/Cu wire as an example, the temperature varies in the range of 4.2-7.6 K for f = 10 Hz (see Fig. 10(c)), while the maximum temperature is less than 5 K before triggering flux jumps for f = 0.5 Hz (see Fig. 10(b)). As studied in previous work (see Refs. [\[57\]](#page-9-42)), the pinning effect on magnetic vortices is weakened and the critical current density is reduced at higher temperature. This leads to the magnetic flux penetrates into the superconductor easier and the magnetic pressure is not sufficiently large to triggered the flux jumps. As a consequence, the smooth flux penetrations are statistically more likely to occur and the thermomagnetic instabilities are less likely triggered in this case. Due to the field-shielding effect of FM layer, one can observe that the temperature of Nb3Sn/FM wire is much smaller than the other two kinds of wires. Although the minimum temperature is about 4.2 K in Nb3Sn/FM wire for f = 10 Hz, the ramping rate of magnetic field is very small and flux jumps cannot be triggered at that time.

For the case of fixing frequency of AC magnetic field f = 0.5 Hz, Fig. 10(d) shows that the flux jumps are

![](_page_7_Figure_6.jpeg)

**Caption:** Figure 11 illustrates a phase diagram showing the thresholds for flux jumps as a function of frequency (f) and amplitude (Ba^0) of the AC magnetic field for three wire types: bare Nb3Sn, Nb3Sn/Cu, and Nb3Sn/FM. The diagram distinctly shows that SC/FM wires have reduced regions where flux jumps occur, indicating superior thermomagnetic stability compared to bare Nb3Sn and Nb3Sn/Cu wires. The FM layer's effectiveness in minimizing flux jump regions supports its role in enhancing the applicational robustness of superconducting wires in dynamic electromagnetic environments.


FIG. 11. The phase diagram of flux jumps in the plane f-Ba<sup>0</sup> for bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire. The different colored curves represent the threshold frequencies as functions of amplitude of AC magnetic field.

triggered in bare Nb3Sn wire when the AC magnetic field amplitude Ba<sup>0</sup> > 0.3 T. By contrast, the threshold amplitude of AC magnetic field can be increased to 0.6 T and 0.8 T in Nb3Sn/Cu wire and Nb3Sn/FM wire, respectively. On one hand, this again demonstrates that the Cu layer and FM layer can enhance the thermomagnetic stabilities of Nb3Sn wire exposed to AC magnetic field. On the other hand, by combining with the fact that the specific heating capacity of cooper is a bit greater than that of ferromagnetic material 1J22, it also provides the evidence for that the mechanism of better suppression of flux jumps by FM layer than the Cu layer is attributed to both thermal effect and electromagnetic effect.

In the last, we explore more complicated situations by varying both amplitude and frequency of AC magnetic fields. Fig. 11 shows the phase diagram of thermomagnetic instabilities in plane f-Ba<sup>0</sup> and the different colors denote the regions of flux jumps triggered in bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire, respectively. Both lower and upper threshold frequencies can be observed in the phase diagrams, which indicates that the flux jumps can only be observed in AC magnetic fields with a specific range of frequencies by fixing the amplitude. Additionally, the threshold frequencies for all three kinds of wires depends on the amplitude of AC magnetic fields. Most importantly, the region of flux jumps for Nb3Sn/FM wire is much smaller than other two kinds of wires. As a consequence, the hybrid Nb3Sn/FM wire indeed has more advantage to improve the thermomagnetic stability than bare Nb3Sn wire and Nb3Sn/Cu wire when it is used in AC magnetic fields.

### IV. CONCLUSIONS

In summary, we experimentally and theoretically explore suppressing the undesirable low-field instabilities of high-J<sup>c</sup> Nb3Sn conductors by FM materials. We fabricate two kinds of sample of bare Nb3Sn and Nb3Sn/FM hybrid wires and implement magnetization measurements. Compared with bare Nb3Sn wire, the experiments indicate that the flux jumps decrease significantly in Nb3Sn/FM hybrid wires. We find that the thicker the FM layer is, the less flux jumps are observed. In order to reveal the underlying mechanism, we also implement numerical simulations. The validation of the numerical simulation are verified by reproducing the experiments quite well. We further numerically study two cases of Nb3Sn/FM hybrid wires exposed to magnetic field loop with constant ramping rate and AC magnetic field with time-varying ramping rate, respectively. Even though the effective critical current density of Nb3Sn wire is decreased by the outer layer because the cross-section of the wire is increased. As shown in Fig. A2 in the Appendix, simulation results indicate that when the critical current of the bare Nb3Sn wire is reduced to the same as the FM/Nb3Sn wire equivalent critical current density, the FM layer is still effective in suppressing flux hopping due to the additional electromagnetic effect. The main results are concluded as follows.

(a) When the field-sweeping rate is small, we find that the flux jumps of Nb3Sn suppressed by FM layer are mainly attributed to the thermal effect, i.e., increasing the total heat capacity. As for higher field-sweeping rates, less flux jumps are observed in Nb3Sn/FM hybrid wire than in Nb3Sn/Cu wire, which indicates that both thermal and eletromagnetic effects of FM materials are crucial to suppress the flux jumps in this case.

(b) For Nb3Sn/FM hybrid wires exposed to AC magnetic field, it is found that the flux jumps can be suppressed completely, which depends on the amplitude and frequency of AC magnetic field. The local magnetic field inside Nb3Sn/FM hybrid wire and the temperature rise is significantly smaller than that in bare Nb3Sn wire and Nb3Sn/Cu wire. We further build up a phase diagram in the plane f-Ba<sup>0</sup> for bare Nb3Sn wire, Nb3Sn/Cu wire and Nb3Sn/FM wire, respectively. The region of flux jumps in Nb3Sn/FM wire is much smaller than other two kinds of wires, which demonstrates that the FM materials exhibit much better suppression of low-field instabilities than merely increasing the heat capacity.

# Acknowledgement

The authors acknowledge support by the the National Natural Science Foundation of China (Grant Nos. 12372210, 11972298, 12174242).

The critical current density J<sup>c</sup> is generally dependent on the temperature and local magnetic field, which can

| TABLE A1. Sample data |                                          |        |    |        |
|-----------------------|------------------------------------------|--------|----|--------|
|                       | Sample FM thickness Diameter Subelements |        |    | Gap    |
| SC                    | \                                        | 1.3mm  | 54 | \      |
| SC/FM                 | 0.1mm                                    | 1.82mm | 54 | 0.16mm |
|                       | 0.2mm                                    | 1.92mm | 54 | 0.11mm |
|                       | 0.3mm                                    | 2.04mm | 54 | 0.07mm |

![](_page_8_Figure_8.jpeg)

**Caption:** Figure A1 depicts the simulated magnetization loss in the ferromagnetic (FM) layer and the associated Joule heating within the superconducting (SC) wire. The graph shows the comparative Joule heating impacts for SC wires unshielded and those with an FM layer, highlighting the FM layer's potential in reducing energy loss due to its magnetic shielding effects. This indicates the FM layer's efficacy in reducing thermomagnetic instability by absorbing unwanted magnetic losses.


FIG. A1. Simulated magnetization loss of ferromagnetic layer and Joule heating of superconducting wire.

be expressed as [\[58,](#page-9-43) [59\]](#page-9-44)

$$J\_c\left(B,T\right) = J\_c\left(B,T\_0\right) \frac{\left(1 + \left(\frac{T}{T\_c}\right)^2\right)^{-0.5} \left(1 - \left(\frac{T}{T\_c}\right)^2\right)^{2.5}}{\left(1 + \left(\frac{T\_0}{T\_c}\right)^2\right)^{-0.5} \left(1 - \left(\frac{T\_0}{T\_c}\right)^2\right)^{2.5}}\tag{A1}$$

with

$$J\_c\left(B, T\_0\right) = \frac{b}{A} \left(\frac{B}{B\_{c2}}\right)^{-p} \left(1 - \frac{B}{B\_{c2}}\right)^{q} \qquad (A2)$$

where a = 25, b = 7731.5, p = 0.4, q = 3.12, A = 6.65 × 10<sup>−</sup><sup>7</sup> .

- <span id="page-8-0"></span>[1] Lee P J, Larbalestier D C 2008 Cryogenics 48 283-292
- [2] Vostner A, Salpietro E 2006 Supercond. Sci. Technol. 19 S90
- [3] Jewell M C, Godeke A, Lee P J and Larbalestier D C 2004 AIP Conf Proc 711 474-484
- <span id="page-8-1"></span>[4] Godeke A, Jewell M C, Fischer C M, Squitieri A A, Lee
- P J and Larbalestier D C 2005 J. Appl. Phys. 97
- <span id="page-8-2"></span>[5] Xu X C 2017 Supercond. Sci. Technol. 30 093001
- <span id="page-8-3"></span>[6] Ferracin P, Ambrosio G, Arbelaez D, Brouwer L, Barzi E, Cooley L, et al. 2022 IEEE Trans. Appl. Supercond. 32 4000906
- <span id="page-8-4"></span>[7] Bruzzone P, March S, Sedlak K, Stepanov B, Wesche R

![](_page_9_Figure_0.jpeg)

**Caption:** Figure A2 compares temperature curves in bare superconducting wires and those with an equivalent FM layer under identical critical current conditions. The FM layer shows reduced peak temperatures, indicating better thermal management and stability under operational conditions. This affirms the FM layer's contribution to lowering temperature fluctuations, preventing flux jumps, and thus enhancing the wire's thermomagnetic stability.


FIG. A2. Temperature curves of bare SC wire and SC/FM wire with the same equivalent critical current.

<span id="page-9-29"></span><span id="page-9-28"></span><span id="page-9-27"></span><span id="page-9-26"></span><span id="page-9-25"></span>and Uglietti D 2014 IEEE Trans. Appl. Supercond. 25 1-4

- [8] Zlobin A V, Andreev N, Apollinari G, Auchmann B, Barzi E, Izquierdo Bermudez S, et al. 2014 IEEE Trans. Appl. Supercond. 25 1-9
- <span id="page-9-0"></span>[9] Feher S, Bossert R C, Ambrosio G, Andreev N, Barzi E, Carcagno R, et al. 2007 IEEE Trans. Appl. Supercond. 17 1126-1129
- <span id="page-9-1"></span>[10] Kim Y B, Hempstead C F and Strnad A R 1963 Phys. Rev. 129 528
- <span id="page-9-30"></span><span id="page-9-2"></span>[11] Wipf S L and Lubell M S 1965 Phys. Lett. 16 103-105
- <span id="page-9-31"></span>[12] Wipe S L 1967 Phys. Rev. 161 404
- [13] Mints R G 1996 Phys. Rev.B 53 12311
- <span id="page-9-3"></span>[14] Zhou Y H and Yang X 2006 Phys. Rev.B 74 054507
- <span id="page-9-32"></span><span id="page-9-4"></span>[15] Nabialek A, Niewczas M, Dabkowska H, Dabkowski A, Castellan J P and Gaulin B D 2003 Phys. Rev.B 67 024518
- <span id="page-9-33"></span><span id="page-9-5"></span>[16] Zebouni N H, Venkataram A, Rao G N, Grenier C G and Reynolds J M 1964 Phys. Rev. Lett. 13 606-609
- <span id="page-9-35"></span><span id="page-9-34"></span><span id="page-9-6"></span>[17] Guillot M, Potel M, Gougeon P, Noel H, Levet J C, Chouteau G and Tholence J L 1964 Phys. Lett. A 127 363-365
- <span id="page-9-36"></span><span id="page-9-7"></span>[18] Jiang L, Xue C, Burger L, et al. 2020 Phys. Rev.B 101 224505
- <span id="page-9-37"></span><span id="page-9-8"></span>[19] Swartz P S and Bean C P 1968 J. Appl. Phys. 39 4991- 4998
- <span id="page-9-38"></span><span id="page-9-9"></span>[20] Wilson M N 1983 Oxford: Oxford University Press pp 139-41
- <span id="page-9-39"></span><span id="page-9-10"></span>[21] Wertheimer M R, Gilchrist J G 1967 J Phys Chem Solids 28 2509-2524
- [22] Ambrosio G, Andreev N, Bartlett S E, et al 2005 IEEE Trans. Appl. Supercond. 15 1545-1549
- <span id="page-9-11"></span>[23] Ghosh A K, Cooley L D, Moodenbaugh A R 2005 IEEE Trans. Appl. Supercond. 15 3360-3363
- <span id="page-9-40"></span><span id="page-9-12"></span>[24] Zlobin A V, Kashikhin V V, Barzi E 2006 IEEE Trans. Appl. Supercond. 16 1308-1311
- <span id="page-9-41"></span>[25] Wang Q Y, Xue C, Zhou Y H 2022 Superconductivity 4 100031
- <span id="page-9-43"></span><span id="page-9-42"></span><span id="page-9-13"></span>[26] Sumption M D, Collings E W 2003 IEEE Trans. Appl. Supercond. 13 3394-3397
- <span id="page-9-44"></span><span id="page-9-14"></span>[27] Glowacki B A 2005 Heidelberg: Springer Berlin Heidelberg 697-738
- <span id="page-9-24"></span><span id="page-9-23"></span><span id="page-9-22"></span><span id="page-9-21"></span><span id="page-9-20"></span><span id="page-9-19"></span><span id="page-9-18"></span><span id="page-9-17"></span><span id="page-9-16"></span><span id="page-9-15"></span> SC/FM [28] Dietderich D R, Bartlett S E, Caspi S, Ferracin P, Gourlay S A, Higley H C, et al. 2005 IEEE Trans. Appl. Supercond. 15 1524-1528
	- [29] Orris D F, Carcagno R, Feher S, Makulski A and Pischalnikov Y M 2005 IEEE Trans. Appl. Supercond. 15 1205-1208
	- [30] Kashikhin V V, Zlobin A V 2005 IEEE Trans. Appl. Supercond. 15 1621-1624
	- [31] Barzi E, Andreev N, Kashikhin V V, Turrioni D and Zlobin A V 2005 IEEE Trans. Appl. Supercond. 15 1537- 1540
	- [32] Ghosh A K, Cooley L D, Moodenbaugh A R, Parrell J A,Field M B, Zhang Y and Hong S 2005 IEEE Trans. Appl. Supercond. 15 3494-7
	- [33] Cooley L D, Chang P S and Ghosh A K 2007 IEEE Trans. Appl. Supercond. 17 2706-9
	- [34] Ghosh A K, Sperry E A, Cooley L D, Moodenbaugh A M, Sabatini R L and Wright J L 2004 Supercond. Sci. Technol. 18 L5
	- [35] Field M B, Zhang Y Z, Miao H P, Gerace M and Parrell J A 2013 IEEE Trans. Appl. Supercond. 24 1-5
	- [36] Goldfarb R B, Goodrich L F, Pyon T, and Gregory E 2001 IEEE Trans. Appl. Supercond. 11 3679-3682
	- [37] Xu X, Sumption M D, Collings E W 2014 Supercond. Sci. Technol. 27 095009
	- [38] Xu X, Li P, Zlobin A V, and Peng X 2018 Supercond. Sci. Technol. 31 03LT02
	- [39] Mawatari Y 2008 Phys. Rev. B 77 104505
	- [40] He A, Xue C, Yong H D and Zhou Y H 2013 Supercond. Sci. Technol. 27 025004
	- [41] Prigozhin L, Sokolovsky V 2023 IEEE Trans. Appl. Supercond. 33 1-10
	- [42] Shaw G, Brisbois J, Pinheiro L, M¨uller J, Blanco Alvarez S, Devillers T, et al. 2018 Rev. Sci. Instrum 89
	- [43] Wang R F, Mei Z L, Cui T J. 2013 Appl. Phys. Lett. 102
	- [44] G¨om¨ory F, Solovyov M, Souc J, Navau C, Prat-Camps J ˇ and Sanchez A 2012 Science 335 1466-146
	- [45] Xun C, Ren H X, Jia P, et al. 2024 arXiv preprint arXiv: 2403.07666.
	- [46] Nguyen D N, Ashworth S P, Willis J O, Sirois F and Grilli F 2009 Supercond. Sci. Technol. 23 025001
	- [47] Liu G L, Zhang G M and Jing L W 2019 Supercond. Sci. Technol. 32 055002
	- [48] Morandi A, Fabbri M 2014 Supercond. Sci. Technol. 28 024004
	- [49] Vestg˚arden J I, Shantsev D V, Galperin Y M and Johansen T H 2011 Phys. Rev. B 84 054537
	- [50] Brambilla R, Grilli F, Martini L 2006 Supercond. Sci. Technol. 20 16
	- [51] Hong Z, Campbell A M, Coombs T A 2006 Supercond. Sci. Technol. 19 1246
	- [52] Iwasa Y 2009 Springer science & business media
	- [53] Koettig T, Maciocha W, Bermudez S, Rysti J, Tavares S, Cacherat F and Bremer J 2017 IOP Conf. Series: Mater. Sci. Eng 171 012103
	- [54] Kemp W R G, Klemens P G, White G K. 1956 Australian Journal of Physics 9 180-188
	- [55] Dixon M, Hoare F E, Holden T M and Moody D E 1965 Proc. Math. Phys. Eng. Sci. 285 561-580
	- [56] Iwasa Y 2009 Springer science & business media
	- [57] Jiang L, Xue C, Burger L, Vanderheyden B, Silhanek A V and Zhou Y H 2020 Phys. Rev. B 101 224505
	- [58] M¨uller K H 1989 Physica C Supercond 159 717-726
	- [59] Griessen1 R 1990 Phys. Rev. Lett 64 1674