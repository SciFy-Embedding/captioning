# arXiv:2310.18245

**Paper ID:** b229126b3e516334378b2f913a5e9ab3

**PDF Path:** apl/Superconductors/arXiv:2310.18245.pdf

**Processing Status:** complete

**Captions Added:** 15

**Generated:** 2025-06-24T13:44:27.906757

---

# Dirac surface states, multiorbital dimerization and superconductivity in Nb- and Ta-based A15 compounds

Raghottam M. Sattigeri [ID](https://orcid.org/0000-0002-6697-1876) , 1, [∗](#page-14-0) Giuseppe Cuono [ID](https://orcid.org/0000-0002-1550-5429) , 1, [†](#page-14-1) Ghulam Hussain [ID](https://orcid.org/0000-0003-1654-9156) , <sup>1</sup> Xing Ming,<sup>2</sup> Angelo Di Bernardo [ID](https://orcid.org/0000-0002-2912-2023) , <sup>3</sup> Carmine Attanasio [ID](https://orcid.org/0000-0002-3848-9169) , <sup>3</sup> Mario Cuoco [ID](https://orcid.org/0000-0002-7325-8331) , 4, 3, [‡](#page-14-2) and Carmine Autieri [ID](https://orcid.org/0000-0002-5008-8165) 1, [§](#page-14-3)

> 1 International Research Centre Magtop, Institute of Physics,

Polish Academy of Sciences, Aleja Lotnik´ow 32/46, PL-02668 Warsaw, Poland

<sup>2</sup>College of Science, Guilin University of Technology, Guilin 541004, People's Republic of China.

<sup>3</sup>Dipartimento di Fisica 'E.R. Caianiello', Universit´a degli Studi di Salerno,

via Giovanni Paolo II 132, I-84084 Fisciano (SA), Italy

<sup>4</sup>Consiglio Nazionale delle Ricerche CNR-SPIN, UOS Salerno, I-84084 Fisciano (Salerno), Italy

(Dated: March 26, 2025)

Using first-principle calculations, we investigate the electronic, topological and superconducting properties of Nb3X (X = Ge, Sn, Sb) and Ta3Y (Y = As, Sb, Bi) A15 compounds. We demonstrate that these compounds host Dirac surface states which are related to a nontrivial Z<sup>2</sup> topological value. The spin-orbit coupling (SOC) splits the highly degenerate R point close to the Fermi level enhancing the amplitude of the spin Hall conductance. Indeed, despite the moderate spin-orbit of the Nb-compounds, a large spin Hall effect is also obtained in Nb3Ge and Nb3Sn compounds. We show that the Coulomb interaction opens the gap at the R point thus making the occurrence of Dirac surface states more obvious. We then investigate the superconducting properties by determining the strength of the electron-phonon BCS coupling. The evolution of the critical temperature is tracked down to the 2D limit indicating a reduction of the transition temperature which mainly arises from the suppression of the density of states at the Fermi level. Finally, we propose a minimal tight-binding model based on three coupled Su-Schrieffer-Heeger chains with t2<sup>g</sup> Ta- and Nb-orbitals reproducing the spin-orbit splittings at the R point among the π-bond bands in this class of compounds. We separate the kinetic parameters in π and δ-bonds, in intradimer and interdimer hoppings and discuss their relevance for the topological electronic structure. We point out that Nb3Ge might represent a Z<sup>2</sup> topological metal with the highest superconducting temperature ever recorded.

#### I. INTRODUCTION

Topological superconductivity is a captivating phase of condensed matter physics characterized by unique electronic properties such as the Majorana fermions[1](#page-14-4) . These particles, distinct for their non-Abelian statistics, hold promise for fault-tolerant quantum computation. This enigmatic phase of matter has ignited a surge of research, with potential applications spanning quantum computing to quantum information storage. Understanding and harnessing topological superconductivity holds the key to obtaining new quantum technologies. Therefore, a growing interest in topological superconductivity has been rising in the last decade.[2,](#page-14-5)[3](#page-14-6) Topological superconductivity can arise from the coexistence of Bardeen–Cooper–Schrieffer (BCS) superconductivity and Dirac states which can lead to mixed pairing order parameters or topological superconductivity, therefore, the Dirac surface states of superconductors are platforms for investigating the interplay between superconductivity and topologically nontrivial Fermi surfaces[4](#page-14-7)[,5](#page-14-8) .

The most common k-space topological phase is characterized by a non-zero Z<sup>2</sup> topological invariant. The Z<sup>2</sup> topological insulators have gapped bulk band structure and gapless surface states. These surface states are protected by time-reversal symmetry. Z<sup>2</sup> topological metals are conducting materials with gapless bulk band structures and gapless surface states[6](#page-14-9)[–8](#page-14-10). In the last years, different families of materials have been proposed to be Z<sup>2</sup> topological metals with a superconductive ground state. We can mention some of the most representative members of these families as the kagome[9](#page-14-11) CsV3Sb5, the non-symmorphic ZrOSSi[10](#page-14-12) , beryllenes[11](#page-14-13) and KHgAs compounds[12](#page-14-14) and the van der Waals Ta2Pd3Te<sup>5</sup> material[13](#page-14-15)[,14](#page-14-16). Dirac surface states were also found in several undoped iron-based systems such as BaFe2As<sup>2</sup> and LiFeAs[15,](#page-14-17)[16](#page-14-18) .

The Nb-based A15 compounds were widely studied in the past due to their superconductivity with a high critical temperature (Tc). The superconductivity in Nb-based A15 compounds was found to be BCS-like[17](#page-14-19), namely the pairing of the superconducting electrons is via electron–phonon coupling. The Fermi level of these compounds is close to a peak in the density of states deriving from dimerized one-dimensional Nb chains. In silicides and germanides of transition metals, the highest T<sup>c</sup> was found in V3Si among all the known binary compounds[18](#page-14-20). The A15 claimed the title of the highest T<sup>c</sup> superconductors in 1954 when T<sup>c</sup> = 18 K was first observed in Nb3Sn[19](#page-14-21). Additionally, high-temperature superconductivity in H-based A15 compounds was recently found[20](#page-14-22) Other Nb-based superconductors were then found as for example Nb3Al with Tc=18.8 K[21](#page-14-23) , Nb3Ga with Tc=20.3 K in[22](#page-14-24) and Nb3Ge with Tc=22.3 K[23](#page-14-25). Recently, several Nb-based compounds were investigated for their exotic superconducting properties such

as the van der Waals NbX2(X=S, Se)[24–](#page-14-26)[26](#page-14-27) and the noncentrosymmetric NbRe[27](#page-14-28)[,28](#page-14-29) Both theoretical and experimental investigations have extensively explored the properties of A15 compounds[17,](#page-14-19)[29–](#page-15-0)[33](#page-15-1) based on niobium and tantalum, due to their high critical temperature and high critical magnetic fields. Due to the discovery of the unconventional high T<sup>c</sup> superconductivity in heavy fermions and cuprates, the superconductive phase of the A15 compounds has received less scientific attention in recent years. Recently, the A15 compounds regained attention due to the large degeneracy at the R point, these A15 compounds are multifold fermion metals[34](#page-15-2) with a notable spin-Hall effect[35](#page-15-3)[,36](#page-15-4) and non-trivial band structure topology[37](#page-15-5)[,38](#page-15-6). Dirac points are emergent along the R–M path due to the C<sup>4</sup> rotational symmetry.[36](#page-15-4) The large spin-Hall conductivity in these compounds is due to the fact that they have bands close to the Fermi level that present crossings unprotected under the action of the spin-orbit coupling interaction (SOC)[35](#page-15-3). The Ta-based A15 compounds have the same filling as the Nb-based A15 compounds, with the Ta having a larger SOC. The Ta3Sb compound with A15 crystal structure was proposed to be a topological superconductor[39](#page-15-7)[,40](#page-15-8) .

Regarding the realization of devices, Nb3Sn superconductors have significant applications in constructing high-field magnets[41](#page-15-9). Nb3Sn can be used as a coating for producing superconducting surfaces[42](#page-15-10) and for particle accelerators[43](#page-15-11). Nb3Sn thin films are promising candidates for future applications in superconducting radio frequency cavities[42](#page-15-10) .

In this paper, we study the electronic, topological and superconductive properties of Nb3X (X = Ge, Sn, Sb) and Ta3Y (Y = As, Sb, Bi) A15 compounds and we demonstrate that all these compounds are Z<sup>2</sup> topological metals hosting Dirac surface states. We study the interplay between the electronic and topological properties with the spin-Hall and BCS superconductivity. These topological properties can be explained by a tight-binding model with three coupled Su–Schrieffer–Heeger (SSH) chains. Nb3Sb and Ta3Y (Y = As, Sb, Bi) have halffilling p- and d-orbitals, while Nb3Ge and Nb3Sn have one electron less. The paper is organized as follows. In the next Section, the results of our ab initio calculations are reported. In more detail, this Section is divided into many Subsections: in Subsection A the structural and electronic properties of Nb3X and Ta3Y are investigated, in Subsection B we study the Spin Hall conductivity, while in Subsections C and D we discuss the topological properties for the Ta-based and Nb-based compounds, respectively. Subsection E is devoted to the superconductivity, while Subsection G is dedicated to the thicknessdependent density of states. In Section III, we report our tight-binding model composed of the three coupled chains of the SSH model with t2<sup>g</sup> orbital basis. Finally, Section IV is devoted to the discussion, conclusions and outlook.

![](_page_1_Figure_3.jpeg)

**Caption:** FIG. 1 illustrates the crystal structure of the A3B compound with 3D perspective views. Part (a) depicts the crystallographic orientation showing A sites in yellow and B sites in brown, with bonding directions indicated. Part (b) presents the Brillouin zone with high-symmetry points Γ, M, and X marked, as well as the kx, ky, kz axes. This structural representation is crucial for contextualizing the electronic properties and band structure calculations discussed in the study.


**c)**

**c**

**a)**

<span id="page-1-0"></span>A sites belong to the dimers along the a, b and c-axis of the unit cell. We indicate with a red bond the dimer along the a-axis, with a green bond the dimer along the b-axis and with a blue bond the dimer along the c-axis. (b) The irreducible bulk Brillouin zone with (001) surface projections is used for computing surface states. (c) Slab of A3B compounds periodic along (001) crystal direction with the top surface layer composed of A<sup>2</sup> and bottom surface layer composed of AB elements. The composition of the bottom and top surface layers is preserved along the paper.

## II. RESULTS

# A. Structural and Electronic properties of Nb3X and Ta3Y

A15 compounds are governed by the Pm3n (No. 223) space group which exhibits an intermetallic nature arising from a chemical composition of A3B, where site A is occupied by a transition metal/d-block element and site B is occupied by the p-block element. The crystal structure presented in Fig. [1\(](#page-1-0)a) is a typical unit cell of an A15 compound with inversion symmetry containing eight atoms with site A forming one-dimensional chains along the edges which are orthogonal to neighboring faces and the B site forming a body-centered cubic lattice. The presence of spatial inversion symmetry is due to the nonsymmorphic space group governing the system which involves a screw axis in the [001] crystal direction. The A sites in A3B composition occupy 6c Wyckoff positions (0.25,0.00,0.50) and the B sites occupy 2a Wyckoff po-

![](_page_2_Figure_0.jpeg)

**Caption:** FIG. 3 displays the electronic structures and surface states of Nb-based compounds: (a) Nb3Ge, (b) Nb3Sn, and (c) Nb3Sb. Panels (d)-(f) present the Dirac dispersions at the Γ point, while (g)-(i) illustrate slab band structures, where the red and blue bands indicate contributions from the top and bottom surface layers, respectively. This setup demonstrates the impact of spin-orbit coupling on electronic dispersion, essential for interpreting topological characteristics and potential surface state contributions to the transport properties of these compounds.


<span id="page-2-0"></span>FIG. 2. Electronic structures of (a) Ta3As, (b) Ta3Sb and (c) Ta3Bi with spin-orbit coupling. Surface states of (d) Ta3As, (e) Ta3Sb and (f) Ta3Bi indicating spin-momentum locked Dirac dispersions at Γ point. Slab band structures of (g) Ta3As, (h) Ta3Sb and (i) Ta3Bi with the red bands indicating the contribution from the top surface layer and the blue bands indicating the contribution from the bottom surface layer of the slab presented in Fig [1.](#page-1-0) The legends on the right in panels (d-i) indicate the spectral weights of the top and bottom layers. The Fermi level is set to zero in all panels.

sitions (0.00,0.00,0.00). The crystal structure presents three dimers of the A atoms, along the a, b and c axes, as shown in Fig. [1\(](#page-1-0)a). A typical slab of A15 compound periodic in [001] crystal direction is presented in Fig. [1\(](#page-1-0)c) with surface Brillouin zone highlighted in Fig. [1\(](#page-1-0)b). Such slab structure leads to the absence of the fractional translation symmetries which breaks the four-fold symmetry in bulk to two-fold symmetry on the surfaces.[37](#page-15-5) Therefore the surface Brillouin zone would be orthorhombic, such surface projection has the momentum path Γ → X → Γ → M/S (since on the surface the S and M points of the orthorhombic and cubic Brillouin zone are equivalent).[35,](#page-15-3)[37](#page-15-5) This exposes two unique surfaces, the top surface originates from the A<sup>2</sup> atomic arrangement and the bottom surface originates from the AB atomic

arrangement. The optimized lattice constants (a) after structural relaxation for Nb3Ge, Nb3Sn and Nb3Sb are 5.177 ˚A, 5.324 ˚A and 5.303 ˚A, respectively which are in agreement with literature.[44–](#page-15-12)[46](#page-15-13) While the optimized lattice constants (a) for Ta3As, Ta3Sb and Ta3Sn are, 5.203 ˚A, 5.329 ˚A and 5.394 ˚A, respectively.

The computational framework is described in Appendix B. In Figs. [2](#page-2-0) and [3](#page-3-0) the electronic structures of Ta3Y and Nb3X compounds are shown, respectively. Both band structures host fourfold rotational symmetries. This is due to the presence of non-symmorphic symmetry operations involving fourfold rotations and fractional translations with respect to the [001] crystal directions. Apart from the symmetries due to timereversal and spatial inversion symmetry, we observe ad-

![](_page_3_Figure_0.jpeg)

**Caption:** FIG. 3 shows the electronic structure and spin-momentum locked surface states of (a) Nb3Ge, (b) Nb3Sn, and (c) Nb3Sb. The middle panels (d)-(f) illustrate the Dirac dispersions at the Γ point, while the lower panels (g)-(i) exhibit the slab band structures, with red bands from the top surface and blue from the bottom. The Fermi level is set to zero. These results highlight the topological characteristics and surface state contributions essential for understanding the compounds' electronic environments


<span id="page-3-0"></span>FIG. 3. Electronic structures of (a) Nb3Ge, (b) Nb3Sn and (c) Nb3Sb with spin-orbit coupling. Surface states of (d) Nb3Ge, (e) Nb3Sn and (f) Nb3Sb indicating spin-momentum locked Dirac dispersions at Γ point. Slab band structures of (g) Nb3Ge, (h) Nb3Sn and (i) Nb3Sb with the red bands indicating the contribution from the top surface layer and the blue bands indicating the contribution from the bottom surface layer of the slab presented in Fig [1.](#page-1-0) The legends on the right in panels (d-i) indicate the spectral weights of the bottom and top layers. The Fermi level is set to zero in all panels.

ditional degeneracies in the momentum. Due to the nonsymmorphic symmetries and screw axes, we have a 4-fold degeneracy above Fermi which splits into 8-fold degeneracy due to SOC whereas, the 6-fold degeneracy below the Fermi level splits into 4-fold and 8-fold degeneracies with the 4-fold degeneracies lying above the 8-fold degenerate states at the R point due to SOC. However, in the case of Ta3Bi due to strong SOC originating from Bi atoms, the 6-fold degenerate states below Fermi splits similar to other compounds but with the exception that the 4-fold degeneracies are pushed below the 8-fold degeneracies. Such 8-fold degenerate points are known to represent double Dirac points.[38](#page-15-6) As we demonstrate in Appendix A, in the low-energy sector at the R-point two groups of bands are present: the bands related to the π

and δ intradimer bonds. In Figs. [2\(](#page-2-0)a,b,c) and [3\(](#page-3-0)a,b,c) the band structures in the range between -1 eV and 1 eV are shown, at the R point we can see in this range the π-bond bands. The δ-bond bands are around 1.5 eV below the Fermi level. The parabolic band appearing at the Γ point for the Ta-based compounds is the 6s band of Ta. The 6s band crosses the p-d bands for Ta3As and Ta3Sb and increases the density of states.

However, at the R point in the momentum space, we have mild variations in Ta3Y and Nb3X compounds with eigenvalues that are four and eight times degenerate as shown in Figs. [2](#page-2-0) and [3](#page-3-0) which is in agreement with the literature. While in Ta-based compounds, the strong SOC at the R-point opens the gap between π-orbital bands, as we can see in Fig. [2](#page-2-0) (a, b, c), in the Nb-based there is a smaller splitting at R but the bands with the camelback shape around R are still crossing keeping conduction and valence sectors entangled, as it is shown in Figs. [3](#page-3-0) (a, b,c).

The crystal symmetries are responsible for orbital hybridizations in the compounds, once we apply the SOC the anticrossings strongly contribute to the Berry curvature and spin Berry curvature. The multiple crossings observed in the electronic structures here give rise to large spin Berry curvatures which effectively translates to a large spin Hall effect in such compounds since the Fermi level lies within gaped crossings.

At the Fermi level, we observe a high density of states with large contributions (due to multiple crossings) from transition metals Ta or Nb and minor contributions from other constituent elements of the A3B composition, as shown in Appendix C. In the momentum space, the conduction band minima at Γ point originates from the sorbitals of group III elements or pnictogens in the A3B composition while the valence band maxima at Γ point originate from the d-orbitals of transition metals. Since these dispersions vary between the Ta3Y and Nb3X compositions, their relative positions define the density of states at the Fermi level and in turn the superconducting critical temperature (Tc) of the system.

In Ta3As and Ta3Bi, the conduction band minima with s-orbital contributions are below the valence band maxima with d-orbital contributions at Γ point as observed in Fig. [2](#page-2-0) (a,c). However, in the case of Ta3Sb and Nb3X, the conduction bands and valence bands are well separated throughout the Brillouin zone as presented in Fig. [2\(](#page-2-0)b) and Fig. [3,](#page-3-0) respectively. In this last case, the sorbital contributions are farther away from the Fermi level as compared to the Ta3Y family. The well-resolved band manifolds in Ta3Sb and Nb3X are accompanied by band inversions across the Fermi level which could produce topological properties. We will see that the presence of the s-orbitals band produces additional anticrossings and additional bands that increase the spin Hall conductivity (SHC) and T<sup>c</sup> respectively. Fig. [4.](#page-4-0) The SHC exhibits a wide peak that encompasses -

#### B. Spin Hall conductivity

It is clear from the electronic structures of these compounds that they host multiple crossings and anticrossings, therefore we have a large change in the spin Berry curvature which indicates that the spin Hall effect should be large. It is known that the SHC is inversely proportional to the spin-orbit induced gap. Accordingly, in the case of Ta3As, Ta3Sb and Ta3Bi (spin-orbit induced gap in electronic structure in increasing order) we find that the SHC at the Fermi level is − 1492.8 (ℏ/e) Scm<sup>−</sup><sup>1</sup> , − 1423.86 (ℏ/e) Scm<sup>−</sup><sup>1</sup> and − 1320.2 (ℏ/e) Scm<sup>−</sup><sup>1</sup> respectively (which has decreasing trend as compared to the increasing order of the spin-orbit induced gap). The SHC of Ta3As and Ta3Bi close to the Fermi level are shown in

![](_page_4_Figure_6.jpeg)

**Caption:** FIG. 4 depicts the spin Hall conductivity of Ta3As and Ta3Bi as a function of energy, highlighting its proximity to the Fermi level. This close proximity suggests potential for realizing strong spintronic properties, emphasizing the relevance of these materials in spintronics due to high SHC associated with robust orbital textures.


<span id="page-4-0"></span>FIG. 4. Spin Hall conductivity as a function of the energy for Ta3As and Ta3Bi compounds. The black solid line indicates Fermi level.

the energy range where the gapped crossings are located. In both cases of Ta3As and Ta3Bi the peak is very close to the Fermi level since the gapped crossings are located near the Fermi level, as shown in Fig. [2\(](#page-2-0)a,c). A similar trend is observed in the case of Nb3X (with group III elements) i.e., for Nb3Ge and Nb3Sn (with the spin-orbit induced gap in electronic structure in increasing order) we have SHC at the Fermi level of − 1691.4 (ℏ/e) Scm<sup>−</sup><sup>1</sup> and − 983.1 (ℏ/e) Scm<sup>−</sup><sup>1</sup> respectively. In the composition of Nb3X, Nb3Sb is an outlier with significantly low and positive SHC of 155.3 (ℏ/e) Scm<sup>−</sup><sup>1</sup> since it is a pnictogen substitution as compared to the other two which are group III elements.

Large values of the SHC are usually associated with robust orbital texture. Indeed, the A15 systems host a robust orbital texture.[37](#page-15-5) Even if the orbital moment is zero, since the compounds present inversion symmetry and highly symmetric crystal structure, the breaking of the inversion symmetry at the surface will generate an orbital magnetic moment. This makes the study of the surface states of A15 compounds interesting.

## C. Topological properties of Ta3Y (Y = As, Sb, Bi)

In this subsection, we discuss first the topological properties of the Ta3Sb that is the ideal case, later, we discuss the topological properties of Ta3As and Ta3Bi with the presence of the s-band at the Fermi level. Since the Ta3Sb compound shows well-resolved band manifolds, we compute the surface states projected on [001] crystal direction as presented in Fig. [2\(](#page-2-0)e). Clearly, this compound hosts spin-momentum locked surface states with Dirac dispersions at Γ point. We also represent the corresponding slab band structure in Fig. [2\(](#page-2-0)h), which shows that the Dirac dispersion at Γ originates from the top surface

![](_page_5_Figure_0.jpeg)

**Caption:** FIG. 5 focuses on the electronic structure of Nb3Ge for U values 0, 2, 3, and 4 eV, embedding the impact of spin-orbit coupling. The surface state panel further showcases the presence of Dirac dispersions, shifting noticeably as U increases, facilitating the understanding of correlations between surface states and electronic properties in topological metals. The spectral weight distributions on the right panels enhance comprehension of how energy interaction varies with U increments.


<span id="page-5-0"></span>FIG. 5. (a) Electronic structure of Nb3Ge for U = 0, 2, 3 and 4 eV with spin-orbit coupling. (b) Surface states of Nb3Ge for U = 0, 2, 3 and 4 eV indicating the presence of spin-momentum locked Dirac dispersions shifting farther from the Fermi level with the increase in U. The legends on the right in panel (b) indicate the spectral weights. The Fermi level is set to zero in all panels.

![](_page_5_Figure_2.jpeg)

**Caption:** FIG. 5b visualizes the surface states of Nb3Ge for various U (0, 2, 3, and 4 eV) values, showcasing spin-momentum locked Dirac dispersions. The spectral weight distribution is indicated on the right, and the Fermi level aligned to zero energy. These findings illustrate how Coulomb repulsion influences the visibility and positioning of Dirac dispersions relative to the Fermi level, providing insights into topological state stabilization in correlated systems like Nb3Ge.


<span id="page-5-1"></span>FIG. 6. (a) Electronic structure of Nb3Sn for U = 0, 2, 3 and 4 eV with spin-orbit coupling. (b) Surface states of Nb3Sn for U = 0, 2, 3 and 4 eV indicating the presence of spin-momentum locked Dirac dispersions shifting farther from the Fermi level with the increase in U. The legends on the right in panel (b) indicate the spectral weights. The Fermi level is set to zero in all panels.

layers. Although these are slightly away from the Fermi level, one can realize them at the Fermi level in experimental conditions by varying the carrier concentrations. Albeit, as the band manifolds are well resolved in the case of Ta3Sb, we compute the Z<sup>2</sup> topological invariants using the Wilson loop method around the Wannier charge centers. Therefore, for Ta3Sb, the four Z<sup>2</sup> 3D topological invariants are (ν0,ν1ν2ν3)=(1;000) indicating a strong topological insulator character.

Since the conduction bands and valence bands are de-

generate at the Γ point due to the presence of the 6s band in the case of Ta3As and Ta3Bi, we do not calculate the Z<sup>2</sup> invariants for these compounds. Although Z<sup>2</sup> is not well-defined, from the band structure we can see the Dirac surface states for Ta3As and Ta3Bi as shown in Figs. [2\(](#page-2-0)d,g) and [2\(](#page-2-0)f,i), respectively, where we show both the surface states and the slab band structures. However, when we include the 6s band in the tight-binding model with the Wannier basis, the Dirac surface states are blurred by the hybridization with the 6s band (see

![](_page_6_Figure_1.jpeg)

**Caption:** FIG. 7 illustrates the electronic structure and surface states of Nb3Sb for U values of 0, 2, 3, and 4 eV. The right panel indicates the spectral weights within the band structure, where spin-momentum locked Dirac dispersions shift further from the Fermi level as U increases. The depiction of these changes in spectral distribution emphasizes the relationship between electronic correlation (U) and surface state evolution, pivotal for topological insulator examination in Nb3Sb.


<span id="page-6-0"></span>FIG. 7. (a) Electronic structure of Nb3Sb for U = 0, 2, 3 and 4 eV with spin-orbit coupling. (b) Surface states of Nb3Sb for U = 0, 2, 3 and 4 eV indicating the presence of spin-momentum locked Dirac dispersions shifting farther from the Fermi level with the increase in U. The legends on the right in panel (b) indicate the spectral weights. The Fermi level is set to zero in all panels.

![](_page_6_Figure_3.jpeg)

**Caption:** FIG. 8 presents phonon dispersion curves coupled with anisotropic Migdal-Eliashberg spectral functions and phonon density of states for (a) Nb3Ge and (b) Nb3Sn. These graphs are essential for understanding lattice dynamics and their impact on superconducting properties, illustrating that these compounds feature distinct vibrational properties which could influence their electron-phonon interactions and critical temperatures.


<span id="page-6-1"></span>FIG. 8. Phonon dispersion curves of (a) Nb3Ge and (b) Nb3Sn alongside the corresponding anisotropic Migdal-Eliashberg spectral functions (α <sup>2</sup>F(ω)) and phonon density of states (F(ω)).

Appendix E for more details). The Dirac surface states would be difficult to detect in Ta3As and Ta3Bi, while they should be observable in all other compounds investigated in this paper.

# D. Topological properties of Nb3X (X = Ge, Sn, Sb): effects of Coulomb repulsion

Although Z<sup>2</sup> invariants are not well defined for Nb3X compounds in the absence of Coulomb repulsion (U), from the band structures, we can see the Dirac surface states as shown in Fig. [3\(](#page-3-0)d,g) for Nb3Ge, [3\(](#page-3-0)e,h) for Nb3Sn and [3\(](#page-3-0)f,i) for Nb3Sb. To investigate further, we perform DFT + U calculations for Nb3Ge, Nb3Sn, and Nb3Sb. The band structures within DFT + U are shown in Figs. [5,](#page-5-0) [6](#page-5-1) and [7.](#page-6-0) The Coulomb repulsion opens a global gap in the momentum space which is obvious from the evolution of bulk band structures for different values of U as shown in Figs. [5\(](#page-5-0)a) for Nb3Ge, [6\(](#page-5-1)a) for Nb3Sn and [7\(](#page-6-0)a) for Nb3Sb. The corresponding surface states are instead shown in Figs. [5\(](#page-5-0)b) for Nb3Ge, [6\(](#page-5-1)b) for Nb3Sn and [7\(](#page-6-0)b) for Nb3Sb. As we can see from the surface states, for all three compounds, the Dirac point at Γ is buried in the bulk at U = 0 eV, while already at U=2 eV it is clearly visible and at U = 4 eV there is a global gap in the momentum space making the calculation of four Z<sup>2</sup> invariants possible.

After the opening of the gap, the calculation of Z<sup>2</sup> is well-defined and we obtain the four Z<sup>2</sup> topological invariants (ν0,ν1ν2ν3)=(1;000), showing that all these Nbbased compounds are Z<sup>2</sup> topological metals hosting Dirac surface states similarly to Ta3Sb compound. This is a clear signature of non-trivial topological states appearing not only in heavy Ta-based compounds but also in Nb-based compounds.

## E. Bulk superconductivity in Nb3X and Ta3Y

Generally, superconductivity requires metallic states at the Fermi level whereas topological insulators are gaped due to spin-orbit interactions, and they present conducting surface states. Hence, finding a stoichiometric composition where bulk superconductivity and topological surface states coexist is a tough task i.e., the surface

FIG. 9. Total density of states (DOS) for bulk Nb3Sn (a) and for various thicknesses (b,c,d,e). The Fermi level is set to zero in all panels highlighted in red dashed line. The trend of increasing DOS as a function of thickness is shown in panel (f). The red line represents the DOS value of the bulk.

states should lie at the Fermi level while the bulk remains fully gaped superconductor in the critical temperature regime. Typically in such systems the Dirac points exist farther from the Fermi level in the conduction bands making it challenging to be observed in experiments like Angle-resolved photoemission spectroscopy (ARPES).

Several compounds have been investigated to this effect with A15 compounds not being an exception due to their metallic character.[35](#page-15-3)[,37](#page-15-5) Studies have been dedicated to Ta3Sb as a potential candidate for topological superconductivity due to the presence of well-resolved spinorbit induced band manifolds and a superconducting critical temperature of 0.7 K.[37](#page-15-5) We revisited this compound and find that in agreement with the previous studies, the Dirac dispersions in surface states are around 500 meV away from the Fermi level in the conduction bands as presented in Fig [2\(](#page-2-0)e) with a superconducting critical temperature of 0.81 K. However, these Dirac dispersions on the top surface merge with the s-bands of the pnictogens and on the bottom surface merge with the d-bands of Ta at the Γ point in the momentum space. Hence it is highly unlikely that Ta3Sb will become a topological superconductor as has been observed in some noncentrosymmetric binary compound BiPd where the Dirac dispersions lie away from the Fermi level in the superconducting regime. This explanation holds true for Ta3As and Ta3Bi as well which have similar characteristics on the surface states (presented in Fig. [2](#page-2-0) (d,f)) with superconducting critical temperatures of 3.0 K and 1.16 K respectively. The sbands in the conduction band minimum (CBM) of Ta3Bi and Ta3As are responsible for higher T<sup>c</sup> as compared to Ta3Sb since they increase the density of states near the

Fermi level.

On the other hand, the T<sup>c</sup> for Nb3Ge and Nb3Sn compounds is higher as compared to Ta3Y and Nb3Sb. Owing to the electronic structure near the Fermi level, Nb3Sb has a T<sup>c</sup> of 2.21 K which is comparable to that of Ta3As and Ta3Bi. However, Nb3Sb is an outlier when compared to Nb3Ge and Nb3Sn which have a T<sup>c</sup> of 15.25 K and 15.66 K respectively. This distinction originates from the lattice dynamics which is evident from the phonon dispersion curves. As compared to Nb3Sn which exhibits anomalous vibrational properties such as soft modes in the Γ → X → M directions in the momentum space (as presented in Fig. [8\(](#page-6-1)b)), the phonon softening is not observed in Nb3Sb presented in Fig. [14\(](#page-12-0)d) (see Appendix D). The anomaly of longitudinal acoustic modes of Nb3Sn softening at lower temperature scales is accompanied by large neutron scattering linewidths which is a function of the electron-phonon coupling coefficient λ(ω), hence resulting in higher Tc. [17](#page-14-19)

We present the phonon dispersion curves alongside the Migdal-Eliashberg spectral functions α <sup>2</sup>F(ω) and phonon density of states F(ω) for Nb3Ge and Nb3Sn are presented in Fig. [8](#page-6-1) and for Ta3Y, Nb3Sb are presented in Fig. [14](#page-12-0) (see Appendix D). The electron-phonon coupling coefficients λ(ω) for Nb3Ge and Nb3Sn are 1.41 and 2.03, respectively. The bulk superconductivity in Nb3X compounds is quite sensitive to the method of crystal growth and various experimental conditions vary the experimental Tc. However, Nb3Ge has been found to exhibit a maximum T<sup>c</sup> of 23.2 K which along with Nb3Sn gives further scope to explore the interplay between the topology and superconductivity owing to large SHC at the Fermi level.
# F. Thickness dependence of the electronic and superconducting properties in Nb3Sn thin films

We calculate the DOS as a function of the thickness to asses the superconducting properties of the Nb3Sn thin film. The electronic properties reported in this subsection were calculated with the computational framework described in Appendix C. In Fig. [9\(](#page-7-0)a-e), we show the DOS calculated for the bulk and for the stoichiometric slabs with different thicknesses. The DOS as a function of the thickness is reported in Fig. [9\(](#page-7-0)f), we observe a decreasing value of the DOS when the thickness got reduced, this will reflect in a reduction of the critical temperature for the thin films of Nb3Sn. For ultra-thin films, the DOS is reduced approximately by half with respect to the bulk value. Consequently, the observed decline in superconductivity in films with reduced thickness can be explained by a weakening of the DOS. While the large DOS in three dimensions can be attributed to the presence of the van Hove singularities[47](#page-15-0), in two dimensions we observed a smoothing of the van Hove singularity. These calculations show that Nb3Sn has a different behavior due to its complex band structure properties compared to other BCS superconductors like V, Ta or Nb where one normally finds that, as soon as the thin film thickness is equal or greater than the coherence length, T<sup>c</sup> is very close to the bulk Tc. The experimental superconducting critical temperature of the thin films strongly depends on the sample quality, of the substrate and usually tends to reduce with the thickness reduction.[48,](#page-15-1)[49](#page-15-2)

# III. TIGHT-BINDING MODEL WITH THREE SSH CHAINS FOR t2<sup>g</sup> ORBITALS

Here, we report a model that includes only the t2<sup>g</sup> orbitals of the Nb/Ta atoms. This tight-binding model allows us to understand the character of the orbitals and the Fermi level and which hopping parameters tune the opening of the topological gap. If we want to produce a minimal model for the d-orbitals of the Nb/Ta atoms, we must include all the 6 Nb/Ta atoms per unit cell. If we could decouple p- and d-electrons, e<sup>g</sup> are below in energy with respect to t2<sup>g</sup> given the crystal field due to the position of other atoms, or anyway, we can assume this in first approximation. Since the charge transfer is zero, the Nb/Ta atoms are in d<sup>5</sup> electronic configuration. We developed a tight-binding model for the t2<sup>g</sup> subset for the 6 Nb/Ta-atoms. Within the t2<sup>g</sup> tight-binding model we can easily include the spin-orbit coupling. The crystal structure presents three dimers of Nb/Ta atoms, along the a, b and c axes, as shown in Fig. [1a](#page-1-0)). We consider in our model the intradimer hybridizations and the interdimer hoppings by including the first and the second nearest neighbours.

The spinful tight-binding model is reported in more detail in Appendix A and it is composed of three coupled SSH chains with t2<sup>g</sup> orbitals. The SSH chain along the

![](0__page_8_Figure_6.jpeg)

**Caption:** FIG. 10 depicts a top view of the A3B compound crystal structure, with A sites in yellow and B sites in brown. The figure shows a green bond illustrating the dimer along the b-axis, emphasizing the geometric arrangement analogous to an SSH chain. Hopping parameters, t1α and t1β, highlight the essential geometric configuration influencing electronic properties without considering SOC effects. This structural representation plays a crucial role in investigating the material's electronic properties influenced by dimer bonds and the tight-binding model framework where SOC and hopping parameters control the topology.


<span id="page-8-0"></span>FIG. 10. Top view of the crystal structure of the A3B compound. The A sites are shown in yellow and B sites are shown in brown. We indicate with a green bond the dimer along the b-axis, we have deleted the other dimers for a better visualization. The hopping parameters t1<sup>α</sup> and t1<sup>β</sup> have the same geometric structure as the hopping in the SSH chain.

b-axis is shown in Fig. [10.](#page-8-0) Very few recent examples of a three-dimensional SSH model have been reported in the literature[50](#page-15-3)[,51](#page-15-4), with significant differences from the case proposed in this paper. The parameters of the model are described in detail in Appendix A, we have two onsite energies E<sup>1</sup> and E<sup>2</sup> for the π and δ-bonds, respectively. The hopping parameters t1α, t1β, t2α, t2<sup>β</sup> for the intradimer elements and t<sup>3</sup> and t<sup>4</sup> for the interdimer elements. The band structure reproduced of the model without including SOC is reported in Figs. [11.](#page-9-0) A magnification of the band structure at the R point is shown in [12a](#page-9-1)) and [12b](#page-9-1)), without and with SOC respectively. We include the SOC within the t2<sup>g</sup> as the SOC of L=−1. In our model, when t1α=t1<sup>β</sup> and t2α=t2β, the π bands are 24 times degenerate without SOC, while when SOC is applied, they are splitted in two sets 12 times degenerate. The δ bands are 12 times degenerate both without and with SOC. In the realistic case, namely when t1<sup>α</sup> is different from t1<sup>β</sup> and t2<sup>α</sup> is different from t2β, the π bands are 12 times degenerate without SOC while with SOC they are sixfold degerate. The δ bands are six-fold degenerate both without and with SOC. When t2α=t2<sup>β</sup> and t1<sup>α</sup> is different from t1β, the π bands are 12 times degenerate and the δ bands are 12 times degenerate both without SOC and with SOC. When t1α=t1<sup>β</sup> and t2<sup>α</sup> is different from t2β, the π are 24 times degenerate without SOC and 12 times degenerate with SOC, while the the δ bands are 12 times degenerate without SOC and six times degenerate with SOC. The opening of the topological gap between the conduction and the valence band at R is controlled by both the difference t1α-t1<sup>β</sup> and the spin-orbit coupling, also the Coulomb repulsion controls the gap as it was proved within DFT+U. The opening of the gap at R allows us to explicitly calculate the topological invariants. The SSH model is a spinless and chiral 1D model, while the model that we have proposed is spinful and not chiral. Despite these differences regarding the material class, the relevant quantity for the topological properties is the difference between the hopping parameters due to the dimerization in both cases.

![](0__page_9_Figure_0.jpeg)

**Caption:** FIG. 11 shows the band structure of a model without spin-orbit coupling (SOC) for a system characterized by parameters: E1=6.359 eV, E2=3.205 eV, t1α=1.691 eV, t1β=1.594 eV, t2α=-0.516 eV, t2β=-0.518 eV, t3=-0.065 eV, t4=-0.495 eV. The band positions, relative to the Fermi level set at zero energy, indicate the control of E1 and E2 on the bands near 1 eV and -3 eV at the R point, respectively. This figure is significant for understanding the electronic properties of the material without SOC, providing a baseline for how SOC affects band degeneracies and topological gap openings.


<span id="page-9-0"></span>FIG. 11. Band structure reproduced with the minimal model without including the SOC interaction. The parameters used are: E1=6.359 eV, E2=3.205 eV, t1α=1.691 eV, t1β=1.594 eV, t2α=− 0.516 eV, t2β=− 0.518 eV, t3=−0.065 eV and t4=−0.495 eV. The value of the onsite energy E<sup>1</sup> controls the position of the bands that are at around 1 eV at the R point, while the value of E<sup>2</sup> controls the position of the bands manifold which is at around -3 eV at the R point. The Fermi level is set at zero energy.

![](0__page_9_Figure_2.jpeg)

**Caption:** This figure focuses on magnifications of the band structure at the R point without and with SOC. It highlights the band degeneracy differences when SOC is applied, showing the π band splits and δ band split characteristics. These adjustments provide insights into topological gap openings controlled by spin-orbit coupling and differences in hopping parameters, essential for understanding the evolution of topological properties in the investigated materials.


<span id="page-9-1"></span>FIG. 12. Zoom of the band structure reproduced with the minimal model along the lines Γ-R-X a) without SOC and b) with the SOC interaction included. The parameters used are: E1=6.359 eV, E2=3.205 eV, t1α=1.691 eV, t1β=1.594 eV, t2α=− 0.516 eV, t2β=− 0.518 eV, t3=−0.065 eV and t4=−0.495 eV. The SOC parameter is λ=0.3 eV. The energy difference between the bands at the R point without SOC is roughly proportional to the module of the difference between the hoppings t1<sup>α</sup> and t1β. The Fermi level is set at zero energy.

### IV. DISCUSSION, CONCLUSIONS AND OUTLOOK

Ta3Sb shows robust orbital textures in the topological surface states[37](#page-15-5). Since the material class is the same and topological surface states are qualitatively the same, we expect the orbital texture even in Nb3Ge and Nb3Sn. The proximity-induced superconductivity in the Dirac surface states can generate Majorana zero mode localized at the vortex[37,](#page-15-5)[52](#page-15-6). Further studies using the

tight-binding model for the bulk and the surface of A15 could shed light on the interplay between topological and superconductive properties.

In conclusion, using a first-principle approach, we provide an extensive description of the electronic, topological and superconductive properties of the Nb- and Ta-based A15 compounds by calculating band structure, SHC, superconducting Tc, and topological surface states. All compounds Nb3X (X = Ge, Sn, Sb) and Ta3Y (Y = As, Sb, Bi) have metallic band structures. Nb3Ge and Nb3Sn have one electron less respect to the other compounds, they have a larger density of states and high superconducting Tc. Ta3As and Ta3Bi host the Ta-6s band at the Fermi level producing larger T<sup>c</sup> and larger SHC compared to Ta3Sb due to the presence of additional DOS and additional anticrossings close to the Fermi level. The spin-Hall conductivity is relatively large also for the lighter elements due to the presence of several anticrossings in the Brillouin zone. All the superconducting T<sup>c</sup> are sizeable. One of the most interesting results among our outcomes is the presence of Dirac surface states at the Γ point for all compounds at the same filling. The ideal case is the Ta3Sb where we have Z<sup>2</sup> metallic compounds with net separation between conduction and valence bands. In all the other compounds, the conduction and the valence bands cross due to the Ta-6s for the Ta-compounds and due to the weaker SOC in the Nb-compounds. Despite the crossing between conduction and valence bands, the band inversion at the high-symmetry points persists in Nb-based compounds producing always the Dirac surface states at the Γ point. Unfortunately, the Dirac surface states are obstructed when we include the hybridization with the s-bands in the Ta-based compounds. The surface Dirac points can be tuned by the Coulomb repulsion, in the case of transition metal termination the Dirac point is around the Fermi level. Even if the Z<sup>2</sup> topological invariant cannot be calculated in all cases, the presence of the Dirac surface states is persistent. Therefore, we can assume that the Nb3X (X = Ge, Sn, Sb) and Ta3Y (Y = As, Sb, Bi) compounds are all Z<sup>2</sup> topological metals. These Dirac surface states could explain the low resistivity of the Nb3Sn surface observed experimentally[42](#page-15-7). Additionally, we provide a minimal tight-binding model composed of three coupled SSH chains and based on the t2<sup>g</sup> orbitals. This tight-binding reproduces the relevant electronic and topological features for these compounds at the Fermi level. With a T<sup>c</sup> of 23.2 K, Nb3Ge could be the Z<sup>2</sup> topological metal with the highest T<sup>c</sup> ever reported amongst the A15 compounds.

The surfaces of the A15 will host an interplay between Z<sup>2</sup> topology, robust orbital texture, breaking of the inversion symmetry and BCS superconductivity with relatively large Tc. The non-trivial topology originates from the well resolved bands in the momentum space which dues to bulk-boundary correspondence host Dirac dispersions along the surfaces. Also, alongside this topological phenomena, due to multiple bands crossing the Fermi energy, the superconductivity is retained simultaneously. Therefore, the surfaces of A15 are a platform to search for exotic topological superconductivity. Once it will be grown the thin film of A15, it will be possible to construct superlattices, junctions, or heterostructures of superconductors and topological compounds in order to study the topological superconductivity via the proximity effect. The non-trivial surface properties of A15 thin films can also represent an interesting platform for the realization of gate-controllable superconducting devices, where recent studies have suggested that surface properties are key for the observation of the suppression of a critical current under an applied gate voltage[53](#page-15-8) .

#### ACKNOWLEDGMENTS

We acknowledge B. Wieder for useful discussions. The work is supported by the Foundation for Polish Science through the International Research Agendas program co-financed by the European Union within the Smart Growth Operational Programme (Grant No. MAB/2017/1). A.D.B. and M.C. acknowledge partial support by the EU's Horizon 2020 Research and Innovation Framework Program under Grant Agreement No. 964398 (SUPERGATE). C. A. acknowledges Erasmus+ for a training scholarship. We acknowledge the access to the computing facilities of the Interdisciplinary Center of Modeling at the University of Warsaw, Grant g91-1418, g91-1419 and g91-1426 for the availability of high-performance computing resources and support. We acknowledge the CINECA award under the ISCRA initiative IsC99 "SILENTS", IsC105 "SILENTSG", IsB26 "SHINY" and IsB27 "SLAM" grants for the availability of high-performance computing resources and support. We acknowledge the access to the computing facilities of the Poznan Supercomputing and Networking Center Grant No. 609 (pl0223-01) and pl0267-01.

### Appendix A: Three coupled SSH chains with a t2<sup>g</sup> orbital basis

Our model includes just the t2<sup>g</sup> electrons for the 6 Nbatoms. The crystal structure presents three dimers of Nb atoms, along the a, b and c axes. We consider in our model the intradimer hybridizations and the interdimer hoppings by considering the first and the second nearest neighbours.

The lattice constant is d=5.139 ˚A. The basis in the Hilbert space is given by the vector

$$\phi\_i^\dagger = (\phi\_{1a}^\dagger, \phi\_{2a}^\dagger, \phi\_{3c}^\dagger, \phi\_{4c}^\dagger, \phi\_{5b}^\dagger, \phi\_{6b}^\dagger),\tag{A1}$$

where

$$\phi\_{1a}^{\dagger} = (\phi\_{1a,xz}^{\dagger}, \phi\_{1a,yz}^{\dagger}, \phi\_{1a,xy}^{\dagger}),\tag{A2}$$

the indices 1,.., 6 indicate the Nb atoms from the first to the sixth, the letters a, c, b indicate the fact that these sites belong to the dimers along the cell directions a, c and b, while xz, yz and xy indicate the t2<sup>g</sup> orbitals dxz, dyz and dxy.

Our Hamiltonian is the following:

$$H = \begin{bmatrix} H\_{aa} & H\_{ac} & H\_{ab} \\ H\_{ca} & H\_{cc} & H\_{cb} \\ H\_{ba} & H\_{bc} & H\_{bb} \end{bmatrix},$$

where the subscript a indicates the dimer along the a direction, composed of Nb<sup>1</sup> and Nb<sup>2</sup> atoms, c the dimer along the c direction, composed of Nb<sup>3</sup> and Nb<sup>4</sup> atoms and b indicates the dimer along the b direction, namely the dimer of Nb<sup>5</sup> and Nb<sup>6</sup> atoms. The aa, bb and cc blocks of the Hamiltonian contain the intradimer hybridizations, while the off-diagonal blocks contain the interdimer hoppings.

The intradimer submatrices are of this type:

$$H\_{aa} = \begin{bmatrix} H\_{1a1a} & H\_{1a2a} \\ H\_{2a1a} & H\_{2a2a} \end{bmatrix},$$

where

$$H\_{1a1a} = \begin{bmatrix} E\_{1a\_{xz}} & 0 & 0 \\ 0 & E\_{1a\_{yz}} & 0 \\ 0 & 0 & E\_{1a\_{xy}} \end{bmatrix},$$

and

$$H\_{1a2a} = \begin{bmatrix} H\_{1a\_{xz}2a\_{xz}} & 0 & 0\\ 0 & H\_{1a\_{yz}2a\_{yz}} & 0\\ 0 & 0 & H\_{1a\_{xy}2a\_{xy}} \end{bmatrix},$$

The on-site energies belong to two groups. The energy E<sup>1</sup> belongs to the orbitals that form intradimer π-bond, while the energy E<sup>2</sup> belongs to the orbitals that form intradimer δ-bond :

$$\begin{aligned} E\_{1a\_{xz}} &= E\_1 & E\_{1a\_{yz}} &= E\_2 & E\_{1a\_{xy}} &= E\_1\\ E\_{2a\_{xz}} &= E\_1 & E\_{2a\_{yz}} &= E\_2 & E\_{2a\_{xy}} &= E\_1\\ E\_{3c\_{xz}} &= E\_1 & E\_{3c\_{yz}} &= E\_1 & E\_{3c\_{xy}} &= E\_2\\ E\_{4c\_{xz}} &= E\_1 & E\_{4c\_{yz}} &= E\_1 & E\_{4c\_{xy}} &= E\_2\\ E\_{5b\_{xz}} &= E\_2 & E\_{5b\_{yz}} &= E\_1 & E\_{5b\_{xy}} &= E\_1\\ E\_{6b\_{xz}} &= E\_2 & E\_{6b\_{yz}} &= E\_1 & E\_{6b\_{xy}} &= E\_1 \end{aligned}$$

The intradimer elements have a hopping form similar to the SSH model[54](#page-15-9) as we can see in Fig. [10,](#page-8-0) therefore, the tight-binding model that describes the A15 is composed of three coupled SSH chains. The intradimer Hamiltonian elements have the following form:

$$\begin{aligned} H\_{1a\_{xz}2a\_{xz}} &= & t\_{1\alpha}e^{-ik\_{x}d/2} + t\_{1\beta}e^{ik\_{x}d/2} \\ H\_{1a\_{yz}2a\_{yz}} &= & t\_{2\alpha}e^{-ik\_{x}d/2} + t\_{2\beta}e^{ik\_{x}d/2} \\ H\_{1a\_{xy}2a\_{xy}} &= & t\_{1\beta}e^{-ik\_{x}d/2} + t\_{1\alpha}e^{ik\_{x}d/2} \\ H\_{3c\_{xz}4c\_{xz}} &= & t\_{1\beta}e^{-ik\_{x}d/2} + t\_{1\alpha}e^{ik\_{x}d/2} \\ H\_{3c\_{yz}4c\_{yz}} &= & t\_{1\alpha}e^{-ik\_{x}d/2} + t\_{1\beta}e^{ik\_{x}d/2} \\ H\_{3c\_{xy}4c\_{xy}} &= & t\_{2\alpha}e^{-ik\_{x}d/2} + t\_{2\beta}e^{ik\_{x}d/2} \\ H\_{5b\_{xz}6b\_{xz}} &= & t\_{2\alpha}e^{-ik\_{x}d/2} + t\_{2\beta}e^{ik\_{x}d/2} \\ H\_{5b\_{yz}6b\_{yz}} &= & t\_{1\alpha}e^{-ik\_{x}d/2} + t\_{1\beta}e^{ik\_{x}d/2} \\ H\_{5b\_{xy}6b\_{xy}} &= & t\_{1\beta}e^{-ik\_{x}d/2} + t\_{1\alpha}e^{ik\_{x}d/2} \\ \end{aligned}$$

where t<sup>1</sup> are π-bonds hoppings and t<sup>2</sup> are δ-bonds hoppings. As expected, we have t<sup>1</sup> > t<sup>2</sup> if we extract the parameters from the wannierization of the DFT band structure. The configuration of α and β are the left and right hopping, their configuration is related to the symmetries of the system. The topological gap at the R-point is controlled by the difference between t1<sup>α</sup> - t1<sup>β</sup> and enhanced by the spin-orbit coupling.

The interdimer submatrices are of the type:

$$H\_{ac} = \begin{bmatrix} H\_{1a3c} & H\_{1a4c} \\ H\_{2a3c} & H\_{2a4c} \end{bmatrix},$$

where we have:

$$H\_{1a3c} = \begin{bmatrix} H\_{1a\_{xz}3c\_{xz}} & H\_{1a\_{xz}3c\_{yz}} & H\_{1a\_{xz}3c\_{xy}} \\ H\_{1a\_{yz}3c\_{xz}} & H\_{1a\_{yz}3c\_{yz}} & H\_{1a\_{yz}3c\_{xy}} \\ H\_{1a\_{xy}3c\_{xz}} & H\_{1a\_{xy}3c\_{yz}} & H\_{1a\_{xy}3c\_{xy}} \end{bmatrix},$$

Regarding the interdimer hybridizations, the intraorbital elements that are different from zero in our model are the following:

$$\begin{array}{rcl} H\_{1a\_{xx}3c\_{xx}} &=& 2t\_3 e^{i(k\_x+k\_z)d/4} \cos(k\_y d/2) \\ H\_{1a\_{xx}4c\_{xx}} &=& 2t\_3 e^{i(k\_x-k\_z)d/4} \cos(k\_y d/2) \\ H\_{1a\_{xy}5b\_{xy}} &=& 2t\_3 e^{i(-k\_x+k\_y)d/4} \cos(k\_z d/2) \\ H\_{1a\_{xy}6b\_{xy}} &=& 2t\_3 e^{i(-k\_x-k\_y)d/4} \cos(k\_z d/2) \\ H\_{2a\_{xx}3c\_{xx}} &=& 2t\_3 e^{i(-k\_x+k\_z)d/4} \cos(k\_y d/2) \\ H\_{2a\_{xx}4c\_{xx}} &=& 2t\_3 e^{i(-k\_x-k\_z)d/4} \cos(k\_y d/2) \\ H\_{2a\_{xy}5b\_{xy}} &=& 2t\_3 e^{i(k\_x+k\_y)d/4} \cos(k\_z d/2) \\ H\_{2a\_{xy}6b\_{xy}} &=& 2t\_3 e^{i(k\_x-k\_y)d/4} \cos(k\_z d/2) \\ H\_{3c\_{yx}5b\_{yz}} &=& 2t\_3 e^{i(-k\_y+k\_z)d/4} \cos(k\_x d/2) \\ H\_{3c\_{yx}6b\_{yz}} &=& 2t\_3 e^{i(k\_y+k\_z)d/4} \cos(k\_x d/2) \\ H\_{4c\_{yx}5b\_{yz}} &=& 2t\_3 e^{i(-k\_y-k\_z)d/4} \cos(k\_x d/2) \\ H\_{4c\_{xy}6b\_{yz}} &=& 2t\_3 e^{i(k\_y-k\_z)d/4} \cos(k\_x d/2) \end{array}$$

and the interorbital Hamiltonian elements are the following:

$$\begin{aligned} H\_{1a\_{yz}3c\_{xy}} &= 2t\_4 e^{i(k\_x+k\_z)d/4} \cos(k\_y d/2) \\ H\_{1a\_{yz}4c\_{xy}} &= -2t\_4 e^{i(k\_x-k\_z)d/4} \cos(k\_y d/2) \\ H\_{1a\_{yz}5b\_{xz}} &= -2t\_4 e^{i(-k\_x+k\_y)d/4} \cos(k\_z d/2) \\ H\_{1a\_{yz}6b\_{xz}} &= 2t\_4 e^{i(-k\_x-k\_y)d/4} \cos(k\_z d/2) \\ H\_{2a\_{yz}3c\_{xy}} &= -2t\_4 e^{i(-k\_x+k\_z)d/4} \cos(k\_y d/2) \\ H\_{2a\_{yz}4c\_{xy}} &= 2t\_4 e^{i(-k\_x-k\_z)d/4} \cos(k\_y d/2) \\ H\_{2a\_{yz}5b\_{xz}} &= 2t\_4 e^{i(k\_x+k\_y)d/4} \cos(k\_z d/2) \\ H\_{2a\_{yz}5b\_{xz}} &= -2t\_4 e^{i(k\_x-k\_y)d/4} \cos(k\_z d/2) \\ H\_{3c\_{xy}5b\_{xz}} &= -2t\_4 e^{i(-k\_y+k\_z)d/4} \cos(k\_x d/2) \\ H\_{3c\_{xy}6b\_{xz}} &= 2t\_4 e^{i(k\_y+k\_z)d/4} \cos(k\_x d/2) \\ H\_{4c\_{xy}5b\_{xz}} &= 2t\_4 e^{i(-k\_y-k\_z)d/4} \cos(k\_x d/2) \\ H\_{4c\_{xy}6b\_{xz}} &= -2t\_4 e^{i(k\_y-k\_z)d/4} \cos(k\_x d/2) \end{aligned}$$

# Appendix B: Computational details for the Quantum Espresso calculations

To compute material properties, we employed density functional theory based first-principles calculations as implemented within the Quantum ESPRESSO code.[55](#page-15-10) We use generalized-gradient approximations with Perdew-Burke-Ernzerhof (PBE) exchange-correlation functional[56](#page-15-11) implemented in norm-conserving pseudopotential with optimized kinetic energy cut-off of 110 Ry for Nb3X (X = Ge, Sb, Sn) and 70 Ry Ta3Y (Y = Bi, As, Sb). A uniform Monkhorst-Pack grid of 20 × 20 × 20 was used in all the calculations which corresponds to 1342 special k-points in the irreducible Brillouin zone. In phonons of Nb3Sn, we observed a numerical artifact at high symmetry point Γ with PBE pseudopotential, hence made additional calculations using local-density approximation with Perdew-Wang 91 gradient-corrected functional only for this system.[57](#page-15-12) The optimized kinetic energy cut-off of 70 Ry was used for this purpose.

The dynamical calculations pertaining to vibrational properties have been performed within the density functional perturbation theory[58](#page-15-13) wherein the dynamical matrices were sampled within the irreducible Brillouin zone using a q-mesh of at least 3 × 3 × 3. The Migdal-Eliashberg spectral functions were calculated using Eq. [B1](#page-11-0) presented below.[58](#page-15-13)

<span id="page-11-0"></span>
$$\alpha^2 F(\omega) = \frac{1}{2\pi N(E\_F)} \sum\_{qv} \frac{\gamma\_{qv}}{\omega\_{qv}} \delta(\omega - \omega\_{qv}) \tag{B1}$$

where, γqv is phonon linewidth and ωqv is the phonon eigenfrequency. Integrating these spectral functions over

![](0__page_12_Figure_0.jpeg)

**Caption:** FIG. 14 displays the phonon dispersion curves and corresponding Migdal-Eliashberg spectral functions for (a) Ta3As, (b) Ta3Sb, (c) Ta3Bi and (d) Nb3Sb. Such spectral analyses underpin the understanding of phonon interactions with electrons, essential for predicting superconducting behavior and critical temperatures based on vibrational properties revealed by these curves.


<span id="page-12-2"></span>FIG. 13. Local density of states of (a) Nb3Ge, (b) Nb3Sn and (c) Nb3Sb. The Fermi level is set to zero in all panels.

the frequencies as presented in Eq. [B2,](#page-12-0) we obtain the electron-phonon coupling coefficient λ(ω).

<span id="page-12-0"></span>
$$
\lambda(\omega) = 2 \int\_0^\infty \frac{\alpha^2 F(\omega)}{\omega} d\omega \tag{B2}
$$

Following this, the superconducting critical temperature (Tc) was calculated using Allen-Dynes modification of the McMillan formula[59](#page-15-14) presented in Eq. [B3](#page-12-1) where µ ∗ is effective Coulomb repulsion parameter ωln is weighted logarithmic average phonon frequencies.

<span id="page-12-1"></span>
$$T\_c = \frac{\omega\_{log}}{1.2} \exp\left(-\frac{1.04(1+\lambda)}{\lambda - \mu^\*(1+0.62\lambda)}\right) \qquad \text{(B3)}$$

To compute the topological surface states, the Z<sup>2</sup> topological invariants and spin Hall conductivity, we used the Wannier90[60](#page-15-15) and WannierTools codes[61](#page-15-16). The basis of the tight-binding model was composed of the Wannier functions[62](#page-15-17) obtained from the d-orbitals of Nb or Ta and the p-orbitals of the other atoms. The atomic orbital configuration is s2d 3 for Nb and Ta, but it becomes d<sup>5</sup> in crystals, therefore Ta and Nb have half-filled d-orbitals. The tight-binding model is composed of 60 d-bands and 12 p-bands for a total of 72 bands. The topological gap is observed at half-filling. Of these 36 occupied bands, 6 are mainly p-bands and 30 are mainly d-bands. The momentum mesh used for calculating spin Hall conductivity σ spinz xy (ω) was 200 × 200 × 200. This was done using the Kubo-Greenwood formula as implemented in Wannier90[63](#page-15-18)[,64](#page-15-19) .

![](0__page_12_Figure_7.jpeg)

**Caption:** FIG. 14 presents the phonon dispersion and Migdal-Eliashberg spectral functions for (a) Ta3As, (b) Ta3Sb, (c) Ta3Bi, and (d) Nb3Sb, with phonon density of states F(ω) included. The graphs underscore lattice dynamics and electron-phonon coupling coefficients relevant for predicting superconducting behavior and critical temperatures. This comprehensive analysis of vibrational modes offers insights into the superconductivity potential of A15 type structures and their variabilities.


<span id="page-12-3"></span>FIG. 14. Phonon dispersion curves of (a) Ta3As, (b) Ta3Sb, (c) Ta3Bi and (d) Nb3Sb alongside the corresponding anisotropic Migdal-Eliashberg spectral functions α <sup>2</sup>F(ω) and phonon density of states F(ω).

![](0__page_13_Figure_0.jpeg)

**Caption:** FIG. 13 showcases the local density of states (LDOS) in (a) Nb3Ge, (b) Nb3Sn, and (c) Nb3Sb with the Fermi level set at zero. These LDOS plots are critical in determining how electronic states at the atomic level contribute to overall material properties like conductivity and potential for hosting topological states.


![](0__page_13_Figure_1.jpeg)

**Caption:** FIG. 14 (c, d) illustrates the phonon dispersion curves and corresponding Migdal-Eliashberg spectral functions for Ta3Bi and Nb3Sb. These graphs are vital for analyzing phonon-related physical properties such as superconductivity, defining how vibrational dynamics may affect electron pairing mechanisms and superconducting T_c


<span id="page-13-0"></span>FIG. 15. Surface states of (a) Ta3As, (b) Ta3Sb and (c) Ta3Bi indicating spin-momentum locked Dirac dispersions at Γ point submerged below the s-bands. Slab band structures of (d) Ta3As, (e) Ta3Sb and (f) Ta3Bi with the red bands indicating the contribution from the top surface layer and the blue bands indicating the contribution from the bottom surface layer of the slab presented in Fig [1.](#page-1-0) The legends on the right in all the panels indicate the spectral weights. The Fermi level is set to zero in all panels.

# Appendix C: Slab calculation and density of states of the Nb-based compounds using VASP

For the slab calculations, additional first-principles calculations are performed via the Vienna ab initio simulation package (VASP)[65,](#page-15-20)[66](#page-15-21) using density functional theory. The generalized gradient approximation with PBE form[56](#page-15-11) and PBEsol[67](#page-15-22) is adopted to calculate the lattice parameters and the density of states (DOS). An energy cutoff of 350 eV and a mesh of 16 × 16 × 1 k-points were chosen for the different thicknesses of Nb-based compounds. The calculations were converged with the convergence criteria of 0.01 eV/˚A, and energy 10<sup>−</sup><sup>5</sup> eV. We have constructed stoichiometric slabs of Nb3Sn with Nb<sup>2</sup> and NbSn terminations as shown in Fig. [1\(](#page-1-0)c) and the internal degrees of freedom were relaxed.

The density of states (DOS) is a relevant quantity to examine the superconducting properties. We report the DOS Nb3Ge, Nb3Sn and Nb3Sb in Fig. [13\(](#page-12-2)a-c). From the DOS, we can see that we have the d-orbitals roughly between -3 eV and +5 eV sandwiched between p-orbitals spectral weight. In these cases, it is extremely challenging to decouple the d-orbitals from the p-orbitals due to their strong hybridization as shown for other materials with such p-d spectral weight[68,](#page-15-23)[69](#page-15-24). To perform the wannierization, we must include the d-orbitals of Nb and the p-orbitals of Ge, Sn and Sb. The DOS results were

obtained within the VASP code and matched with the results obtained with Quantum Espresso.

We report the calculation of the equilibrium lattice constant within PBEsol and PBE. The system is cubic and does not have internal degrees of freedom, therefore, the only parameter to optimize is the lattice constant a. The lattice constant results were obtained within the VASP code and matched with the results obtained with Quantum espresso. The PBE exchange-correlation functional overestimates the experimental lattice constant by 1%, indeed the lattice constant for Nb3Ge, Nb3Sn and Nb3Sb are 5.188 ˚A, 5.339 ˚A and 5.314 ˚A, respectively. The PBEsol gives slightly better results underestimating the lattice constant by 0.1% (i.e., lattice constant for Nb3Ge, Nb3Sn and Nb3Sb are 5.135 ˚A, 5.280 ˚A and 5.259 ˚A, respectively.

# Appendix D: Phonon dispersions and Migdal-Eliashberg spectral functions for Ta3Y (Y = Ge,Sn,Sb) and Nb3Sb compounds

In this Section, we report the phonon dispersion curves, phonon density of states, Migdal-Eliashberg spectral functions and the electron-phonon coupling values for all the compounds except that for Nb3Ge and Nb3Sn which are described in the main text. The results are

reported in Fig. [14.](#page-12-3) The electron-phonon coupling coefficient λ(ω) for Ta3As, Ta3Sb, Ta3Bi and Nb3Sb are 0.56, 0.41, 0.45 and 0.47, respectively. Correspondingly the T<sup>c</sup> is slightly higher in Ta3As, Ta3Bi and Nb3Sb as compared to Ta3Sb (see the main text for the numerical values), this is essentially due to the density of states at the Fermi level in the electronic structure. Although, Ta3Sb shows softening of the optical phonon modes which implies that the T<sup>c</sup> should be as high as Ta3Bi and Nb3Sb, however, the density of states at the Fermi level in the electronic structure dominates the superconducting behavior leading to lower Tc.

### Appendix E: Hybridization of the Dirac surface states with the s-bands for Ta-based systems

In this Section, we want to include the s-band in our tight-binding model and see the effect on the Dirac surface states. While the results of the main text are obtained with high numerical accuracy, the process of including the s-band in the tight-binding Hamiltonian is obtained within strong approximations in the wannierization process (just including the s-band in the frozen window). The presence of the s-band in the Ta-based system affects the Dirac surface states by submerging the Dirac points below them. We report in Fig. [15](#page-13-0) the surface states for the Ta3Y which include the s-bands. In our calculations, we can observe the Dirac surface states are pushed below the s-bands and the Dirac surface states are blurred by the hybridization with the s-bands.

The parity of the s-band and the parity of the other highest valence bands are both positive, therefore, we expect that this s-band would produce an adiabatic transition without affecting the topology[70,](#page-15-25)[71](#page-15-26). Indeed, in the literature, it was shown that the Dirac surface states of Ta3Sb can coexist with the s-bands[35](#page-15-27) .

- ∗ [rsattigeri@magtop.ifpan.edu.pl](mailto:rsattigeri@magtop.ifpan.edu.pl)
- † [gcuono@magtop.ifpan.edu.pl](mailto:gcuono@magtop.ifpan.edu.pl)
- ‡ [mario.cuoco@spin.cnr.it](mailto:mario.cuoco@spin.cnr.it)
- § [autieri@magtop.ifpan.edu.pl](mailto:autieri@magtop.ifpan.edu.pl)
- <sup>1</sup> C. Beenakker, [Annual Review of Condensed Matter](http://dx.doi.org/10.1146/annurev-conmatphys-030212-184337) Physics 4[, 113 \(2013\),](http://dx.doi.org/10.1146/annurev-conmatphys-030212-184337) [https://doi.org/10.1146/annurev](http://arxiv.org/abs/https://doi.org/10.1146/annurev-conmatphys-030212-184337)[conmatphys-030212-184337.](http://arxiv.org/abs/https://doi.org/10.1146/annurev-conmatphys-030212-184337)
- <sup>2</sup> S. Yonezawa, K. Tajiri, S. Nakata, Y. Nagai, Z. Wang, K. Segawa, Y. Ando, and Y. Maeno, [Nature Physics](http://dx.doi.org/10.1038/nphys3907) 13, [123 \(2017\).](http://dx.doi.org/10.1038/nphys3907)
- <sup>3</sup> M. M. Sharma, P. Sharma, N. K. Karn, and V. P. S. Awana, 35[, 083003 \(2022\).](http://dx.doi.org/10.1088/1361-6668/ac6987)
- <sup>4</sup> S. A. Yang, H. Pan, and F. Zhang, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.113.046401) 113, [046401 \(2014\).](http://dx.doi.org/10.1103/PhysRevLett.113.046401)
- <sup>5</sup> Z. S. Gao, X.-J. Gao, W.-Y. He, X. Y. Xu, T. K. Ng, and K. T. Law, [Quantum Frontiers](http://dx.doi.org/ 10.1007/s44214-022-00001-1) 1, 3 (2022).
- <sup>6</sup> W. Cheng, A. Cerjan, S.-Y. Chen, E. Prodan, T. A. Loring, and C. Prodan, [Nature Communications](http://dx.doi.org/ 10.1038/s41467-023-38862-2) 14, 3071 (2023).
- <sup>7</sup> Y. X. Zhao and Z. D. Wang, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.116.016401) 116, 016401 [\(2016\).](http://dx.doi.org/10.1103/PhysRevLett.116.016401)
- <sup>8</sup> A. Cerjan and T. A. Loring, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.106.064109) 106, 064109 [\(2022\).](http://dx.doi.org/10.1103/PhysRevB.106.064109)
- <sup>9</sup> B. R. Ortiz, S. M. L. Teicher, Y. Hu, J. L. Zuo, P. M. Sarte, E. C. Schueller, A. M. M. Abeykoon, M. J. Krogstad, S. Rosenkranz, R. Osborn, R. Seshadri, L. Balents, J. He, and S. D. Wilson, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.125.247002) 125, 247002 (2020).
- <sup>10</sup> S. K. Ghosh1, B. Li, C. Xu, A. D. Hillier, P. K. Biswas, X. Xu, and T. Shiroka, [Frontiers in Physics](http://dx.doi.org/ 10.3389/fphy.2023.1256166) 11 (2023), [10.3389/fphy.2023.1256166.](http://dx.doi.org/ 10.3389/fphy.2023.1256166)
- <sup>11</sup> J. Li, M. Guo, J. Si, L. Shi, X. Shi, J.-J. Ma, Q. Zhang, D. J. Singh, P.-F. Liu, and B.-T. Wang, [Materials Today](http://dx.doi.org/https://doi.org/10.1016/j.mtphys.2023.101257) Physics 38[, 101257 \(2023\).](http://dx.doi.org/https://doi.org/10.1016/j.mtphys.2023.101257)
- <sup>12</sup> G. Dai, Y. Jia, B. Gao, Y. Peng, J. Zhao, Y. Ma, C. Chen, J. Zhu, Q. Li, R. Yu, and C. Jin, [NPG Asia Materials](http://dx.doi.org/10.1038/s41427-023-00496-7) 15, [52 \(2023\).](http://dx.doi.org/10.1038/s41427-023-00496-7)
- <sup>13</sup> N. Higashihara, Y. Okamoto, Y. Yoshikawa, Y. Yamakawa, H. Takatsu, H. Kageyama, and K. Takenaka, [Journal of](http://dx.doi.org/ 10.7566/jpsj.90.063705) [the Physical Society of Japan](http://dx.doi.org/ 10.7566/jpsj.90.063705) 90, 063705 (2021).
- <sup>14</sup> H. Yang, Y. Zhou, S. Wang, J. Wang, X. Chen, L. Zhang,
- C. Xu, and Z. Yang, Phys. Rev. B 107[, L020503 \(2023\).](http://dx.doi.org/ 10.1103/PhysRevB.107.L020503) <sup>15</sup> P. Zhang, Z. Wang, X. Wu, K. Yaji, Y. Ishida, Y. Kohama, G. Dai, Y. Sun, C. Bareille, K. Kuroda, T. Kondo, K. Okazaki, K. Kindo, X. Wang, C. Jin, J. Hu, R. Thomale, K. Sumida, S. Wu, K. Miyamoto, T. Okuda, H. Ding, G. D. Gu, T. Tamegai, T. Kawakami, M. Sato, and S. Shin, [Nature Physics](http://dx.doi.org/10.1038/s41567-018-0280-z) 15, 41 (2018).
- <sup>16</sup> X. Ma, G. Wang, R. Liu, T. Yu, Y. Peng, P. Zheng, and Z. Yin, Phys. Rev. B 106[, 115114 \(2022\).](http://dx.doi.org/10.1103/PhysRevB.106.115114)
- <sup>17</sup> G. Stewart, [Physica C: Superconductivity and its Appli](http://dx.doi.org/https://doi.org/10.1016/j.physc.2015.02.013)cations 514[, 28 \(2015\).](http://dx.doi.org/https://doi.org/10.1016/j.physc.2015.02.013)
- <sup>18</sup> G. F. Hardy and J. K. Hulm, Phys. Rev. 93[, 1004 \(1954\).](http://dx.doi.org/10.1103/PhysRev.93.1004)
- <sup>19</sup> B. T. Matthias, T. H. Geballe, S. Geller, and E. Corenzwit, Phys. Rev. 95[, 1435 \(1954\).](http://dx.doi.org/10.1103/PhysRev.95.1435)
- <sup>20</sup> S. Cross, J. Buhot, A. Brooks, W. Thomas, A. Kleppe, O. Lord, and S. Friedemann, "High-temperature superconductivity in a15-type la4h23 below 100 gpa," (2023), [arXiv:2308.02977 \[cond-mat.supr-con\].](http://arxiv.org/abs/2308.02977)
- <sup>21</sup> R. Willens, T. Geballe, A. Gossard, J. Maita, A. Menth, G. Hull, and R. Soden, [Solid State Communications](http://dx.doi.org/ https://doi.org/10.1016/0038-1098(69)90773-X) 7, [837 \(1969\).](http://dx.doi.org/ https://doi.org/10.1016/0038-1098(69)90773-X)
- <sup>22</sup> G. Webb, L. Vieland, R. Miller, and A. Wicklund, [Solid](http://dx.doi.org/ https://doi.org/10.1016/0038-1098(71)90314-0) [State Communications](http://dx.doi.org/ https://doi.org/10.1016/0038-1098(71)90314-0) 9, 1769 (1971).
- <sup>23</sup> J. R. Gavaler, [Applied Physics Letters](http://dx.doi.org/10.1063/1.1654966) 23, [480 \(2003\),](http://dx.doi.org/10.1063/1.1654966) [https://pubs.aip.org/aip/apl/article](http://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/23/8/480/7732534/480_1_online.pdf)[pdf/23/8/480/7732534/480](http://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/23/8/480/7732534/480_1_online.pdf) 1 online.pdf.
- <sup>24</sup> W. Wang, C. Si, W. Lei, F. Xiao, Y. Liu, C. Autieri, and X. Ming, Phys. Rev. B 105[, 035119 \(2022\).](http://dx.doi.org/10.1103/PhysRevB.105.035119)
- <sup>25</sup> W. Wang, B. Wang, Z. Gao, G. Tang, W. Lei, X. Zheng, H. Li, X. Ming, and C. Autieri, [Phys. Rev. B](http://dx.doi.org/ 10.1103/PhysRevB.102.155115) 102, 155115 [\(2020\).](http://dx.doi.org/ 10.1103/PhysRevB.102.155115)
- <sup>26</sup> F. Cossu, K. Palot´as, S. Sarkar, I. D. Marco, and A. Akbari, NPG Asia Materials 12 [\(2020\), 10.1038/s41427-020-](http://dx.doi.org/10.1038/s41427-020-0207-x) [0207-x.](http://dx.doi.org/10.1038/s41427-020-0207-x)
- <sup>27</sup> M. Caputo, C. Cirillo, and C. Attanasio, [Applied Physics Letters](http://dx.doi.org/10.1063/1.4997675) 111, 192601 [\(2017\),](http://dx.doi.org/10.1063/1.4997675) [https://pubs.aip.org/aip/apl/article](http://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/doi/10.1063/1.4997675/14102743/192601_1_online.pdf)[pdf/doi/10.1063/1.4997675/14102743/192601](http://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/doi/10.1063/1.4997675/14102743/192601_1_online.pdf) 1 online.pdf.
- <sup>28</sup> Z. M. Kakhaki, A. Leo, F. Chianese, L. Parlato, G. P. Pepe,

A. Nigro, C. Cirillo, and C. Attanasio, [Beilstein Journal](http://dx.doi.org/10.3762/bjnano.14.5) [of Nanotechnology](http://dx.doi.org/10.3762/bjnano.14.5) 14, 45 (2023).

- <sup>29</sup> M. Sundareswari, S. Ramasubramanian, and M. Rajagopalan, [Solid State Communications](http://dx.doi.org/https://doi.org/10.1016/j.ssc.2010.08.004) 150, 2057 (2010).
- <sup>30</sup> J. Muller, [Reports on Progress in Physics](http://dx.doi.org/10.1088/0034-4885/43/5/003) 43, 641 (1980).
- <sup>31</sup> Y. Wu, L. Bao, X. Wang, Y. Wang, M. Peng, and Y. Duan, [Materials Today Communications](http://dx.doi.org/ https://doi.org/10.1016/j.mtcomm.2020.101410) 25, 101410 (2020).
- <sup>32</sup> Y. Ding, S. Deng, and Y. Zhao, [Journal of Modern Trans](http://dx.doi.org/ 10.1007/s40534-014-0045-z)portation 22[, 183 \(2014\).](http://dx.doi.org/ 10.1007/s40534-014-0045-z)
- <sup>33</sup> G. D. Marzi, L. Morici, L. Muzzi, A. della Corte, and M. B. Nardelli, [Journal of Physics: Condensed Matter](http://dx.doi.org/ 10.1088/0953-8984/25/13/135702) 25, [135702 \(2013\).](http://dx.doi.org/ 10.1088/0953-8984/25/13/135702)
- <sup>34</sup> B. Bradlyn, J. Cano, Z. Wang, M. G. Vergniory, C. Felser, R. J. Cava, and B. A. Bernevig, Science 353[, aaf5037 \(2016\),](http://dx.doi.org/10.1126/science.aaf5037) [https://www.science.org/doi/pdf/10.1126/science.aaf5037.](http://arxiv.org/abs/https://www.science.org/doi/pdf/10.1126/science.aaf5037)
- <span id="page-15-27"></span><sup>35</sup> E. Derunova, Y. Sun, C. Felser, S. Parkin, B. Yan, and M. Ali, Science Advances 5, eaav8575 (2019).
- <sup>36</sup> W. Hou, J. Liu, X. Zuo, J. Xu, X. Zhang, D. Liu, M. Zhao, Z.-G. Zhu, H.-G. Luo, and W. Zhao, [npj Computational](http://dx.doi.org/ 10.1038/s41524-021-00504-w) Materials 7[, 37 \(2021\).](http://dx.doi.org/ 10.1038/s41524-021-00504-w)
- <span id="page-15-5"></span><sup>37</sup> M. Kim, C.-Z. Wang, and K.-M. Ho, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.99.224506) 99, [224506 \(2019\).](http://dx.doi.org/10.1103/PhysRevB.99.224506)
- <sup>38</sup> B. J. Wieder, Y. Kim, A. M. Rappe, and C. L. Kane, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.116.186402) 116, 186402 (2016).
- <sup>39</sup> R. Chapai, A. Rydh, M. P. Smylie, D. Y. Chung, H. Zheng, A. E. Koshelev, J. E. Pearson, W.-K. Kwok, J. F. Mitchell, and U. Welp, Phys. Rev. B 105[, 184510 \(2022\).](http://dx.doi.org/10.1103/PhysRevB.105.184510)
- <sup>40</sup> R. Chapai, M. P. Smylie, H. Hebbeker, D. Y. Chung, W.- K. Kwok, J. F. Mitchell, and U. Welp, [Phys. Rev. B](http://dx.doi.org/ 10.1103/PhysRevB.107.104504) 107, [104504 \(2023\).](http://dx.doi.org/ 10.1103/PhysRevB.107.104504)
- <sup>41</sup> X. Xu, [Superconductor Science and Technology](http://dx.doi.org/10.1088/1361-6668/aa7976) 30, 093001 [\(2017\).](http://dx.doi.org/10.1088/1361-6668/aa7976)
- <span id="page-15-7"></span><sup>42</sup> N. Sch¨afer, N. Karabas, J. P. Palakkal, S. Petzold, M. Major, N. Pietralla, and L. Alff, [Journal of Applied Physics](http://dx.doi.org/10.1063/5.0015376) 128[, 133902 \(2020\),](http://dx.doi.org/10.1063/5.0015376) [https://pubs.aip.org/aip/jap/article](http://arxiv.org/abs/https://pubs.aip.org/aip/jap/article-pdf/doi/10.1063/5.0015376/15253550/133902_1_online.pdf)[pdf/doi/10.1063/5.0015376/15253550/133902](http://arxiv.org/abs/https://pubs.aip.org/aip/jap/article-pdf/doi/10.1063/5.0015376/15253550/133902_1_online.pdf) 1 online.pdf.
- <sup>43</sup> C. Sundahl, J. Makita, P. B. Welander, Y.-F. Su, F. Kametani, L. Xie, H. Zhang, L. Li, A. Gurevich, and C.-B. Eom, Scientific Reports 11 [\(2021\), 10.1038/s41598-](http://dx.doi.org/10.1038/s41598-021-87119-9) [021-87119-9.](http://dx.doi.org/10.1038/s41598-021-87119-9)
- <sup>44</sup> H. Devantay, J. Jorda, M. Decroux, J. Muller, and R. Fl¨ukiger, Journal of Materials Science 16, 2145 (1981).
- <sup>45</sup> Y. Tarutani and M. Kudo, Journal of the Less Common Metals 55, 221 (1977).
- <sup>46</sup> S. Furuseth and A. Kjekshus, ACTA CHEMICA SCAN-DINAVICA 18, 1180 (1964).
- <span id="page-15-0"></span><sup>47</sup> J. Labb´e, S. Bariˇsi´c, and J. Friedel, [Physical Review Let](http://dx.doi.org/10.1103/physrevlett.19.1039)ters 19[, 1039–1041 \(1967\).](http://dx.doi.org/10.1103/physrevlett.19.1039)
- <span id="page-15-1"></span><sup>48</sup> K. Il'in, D. Rall, M. Siegel, A. Engel, A. Schilling, A. Semenov, and H.-W. Huebers, [Physica C: Superconductivity](http://dx.doi.org/ https://doi.org/10.1016/j.physc.2010.02.042) 470[, 953 \(2010\),](http://dx.doi.org/ https://doi.org/10.1016/j.physc.2010.02.042) vortex Matter in Nanostructured Superconductors.
- <span id="page-15-2"></span><sup>49</sup> H. Asano, K. Tanabe, Y. Katoh, and O. Michikami, [Japanese Journal of Applied Physics](http://dx.doi.org/ 10.1143/JJAP.27.35) 27, 35 (1988).
- <span id="page-15-3"></span><sup>50</sup> J. Kim, C.-Y. Huang, H. Lin, D. Vanderbilt, and N. Kious-

sis, Phys. Rev. B 107[, 045135 \(2023\).](http://dx.doi.org/ 10.1103/PhysRevB.107.045135)

- <span id="page-15-4"></span><sup>51</sup> A. K losi´nski, W. Brzezicki, A. Lau, C. E. Agrapidis, A. M. Ole´s, J. van Wezel, and K. Wohlfeld, [Phys. Rev. B](http://dx.doi.org/ 10.1103/PhysRevB.107.125123) 107, [125123 \(2023\).](http://dx.doi.org/ 10.1103/PhysRevB.107.125123)
- <span id="page-15-6"></span><sup>52</sup> L. Fu and C. L. Kane, [Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.100.096407) 100, 096407 (2008).
- <span id="page-15-8"></span><sup>53</sup> L. Ruf, T. Elalaily, C. Puglia, Y. P. Ivanov, F. Joint, M. Berke, A. Iorio, P. Makk, G. De Simoni, S. Gasparinetti, G. Divitini, S. Csonka, F. Giazotto, E. Scheer, and A. Di Bernardo, [APL Materials](http://dx.doi.org/ 10.1063/5.0159750) 11, [091113 \(2023\),](http://dx.doi.org/ 10.1063/5.0159750) [https://pubs.aip.org/aip/apm/article](http://arxiv.org/abs/https://pubs.aip.org/aip/apm/article-pdf/doi/10.1063/5.0159750/18132042/091113_1_5.0159750.pdf)[pdf/doi/10.1063/5.0159750/18132042/091113](http://arxiv.org/abs/https://pubs.aip.org/aip/apm/article-pdf/doi/10.1063/5.0159750/18132042/091113_1_5.0159750.pdf) 1 5.0159750.pdf.
- <span id="page-15-9"></span><sup>54</sup> W. P. Su, J. R. Schrieffer, and A. J. Heeger, [Phys. Rev.](http://dx.doi.org/10.1103/PhysRevLett.42.1698) Lett. 42[, 1698 \(1979\).](http://dx.doi.org/10.1103/PhysRevLett.42.1698)
- <span id="page-15-10"></span><sup>55</sup> P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, G. L. Chiarotti, M. Cococcioni, I. Dabo, et al., Journal of physics: Condensed matter 21, 395502 (2009).
- <span id="page-15-11"></span><sup>56</sup> J. P. Perdew, K. Burke, and M. Ernzerhof, Physical Review Letters 77, 3865 (1996).
- <span id="page-15-12"></span><sup>57</sup> J. P. Perdew and Y. Wang, Physical Review B 45, 13244 (1992).
- <span id="page-15-13"></span><sup>58</sup> S. Baroni, S. De Gironcoli, A. Dal Corso, and P. Giannozzi, Reviews of modern Physics 73, 515 (2001).
- <span id="page-15-14"></span><sup>59</sup> P. B. Allen and R. Dynes, Physical Review B 12, 905 (1975).
- <span id="page-15-15"></span><sup>60</sup> A. A. Mostofi, J. R. Yates, G. Pizzi, Y.-S. Lee, I. Souza, D. Vanderbilt, and N. Marzari, Computer Physics Communications 185, 2309 (2014).
- <span id="page-15-16"></span><sup>61</sup> Q. Wu, S. Zhang, H.-F. Song, M. Troyer, and A. A. Soluyanov, Computer Physics Communications 224, 405 (2018).
- <span id="page-15-17"></span><sup>62</sup> N. Marzari and D. Vanderbilt, Physical Review B 56, 12847 (1997).
- <span id="page-15-18"></span><sup>63</sup> G.-Y. Guo, S. Murakami, T.-W. Chen, and N. Nagaosa, Physical Review Letters 100, 096401 (2008).
- <span id="page-15-19"></span><sup>64</sup> J. Qiao, J. Zhou, Z. Yuan, and W. Zhao, Physical Review B 98, 214402 (2018).
- <span id="page-15-20"></span><sup>65</sup> P. E. Bl¨ochl, Physical Review B 50, 17953 (1994).
- <span id="page-15-21"></span><sup>66</sup> G. Kresse and J. Furthm¨uller, Physical Review B 54, 11169 (1996).
- <span id="page-15-22"></span><sup>67</sup> J. P. Perdew, A. Ruzsinszky, G. I. Csonka, O. A. Vydrov, G. E. Scuseria, L. A. Constantin, X. Zhou, and K. Burke, Physical Review Letters 100, 136406 (2008).
- <span id="page-15-23"></span><sup>68</sup> G. Cuono, F. Forte, M. Cuoco, R. Islam, J. Luo, C. Noce, and C. Autieri, [Phys. Rev. Mater.](http://dx.doi.org/ 10.1103/PhysRevMaterials.3.095004) 3, 095004 (2019).
- <span id="page-15-24"></span><sup>69</sup> A. S. Wadge, G. Grabecki, C. Autieri, B. J. Kowalski, P. Iwanowski, G. Cuono, M. F. Islam, C. M. Canali, K. Dybko, A. Hruban, A. Lusakowski, T. Wojciechowski, R. Diduszko, A. Lynnyk, N. Olszowska, M. Rosmus, J. Ko lodziej, and A. Wi´sniewski, [Journal of Physics: Con](http://dx.doi.org/10.1088/1361-648X/ac43fe)densed Matter 34[, 125601 \(2022\).](http://dx.doi.org/10.1088/1361-648X/ac43fe)
- <span id="page-15-25"></span><sup>70</sup> L. Fu and C. L. Kane, Phys. Rev. B 76[, 045302 \(2007\).](http://dx.doi.org/10.1103/PhysRevB.76.045302)
- <span id="page-15-26"></span><sup>71</sup> M. Dzero, J. Xia, V. Galitski, and P. Coleman, [Annual Review of Condensed Matter Physics](http://dx.doi.org/ 10.1146/annurev-conmatphys-031214-014749) 7, [249 \(2016\),](http://dx.doi.org/ 10.1146/annurev-conmatphys-031214-014749) [https://doi.org/10.1146/annurev-conmatphys-](http://arxiv.org/abs/https://doi.org/10.1146/annurev-conmatphys-031214-014749)[031214-014749.](http://arxiv.org/abs/https://doi.org/10.1146/annurev-conmatphys-031214-014749)