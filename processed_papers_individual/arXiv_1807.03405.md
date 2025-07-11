# arXiv:1807.03405

**Paper ID:** 2d6ab9dee49ad428ca67804d29c14d6c

**PDF Path:** apl/Superconductors/arXiv:1807.03405.pdf

**Processing Status:** complete

**Captions Added:** 4

**Generated:** 2025-06-24T13:44:27.215002

---

## Topological states in A15 superconductors

Minsung Kim,<sup>1</sup> Cai-Zhuang Wang,<sup>1</sup> and Kai-Ming Ho1, 2

<sup>1</sup>Ames Laboratory – U.S. Department of Energy, Iowa State University, Ames, Iowa 50011, USA <sup>2</sup>Department of Physics and Astronomy, Iowa State University, Ames, Iowa 50011, USA

(Dated: September 16, 2024)

## Abstract

Superconductors with the A15 structure are prototypical type-II s-wave superconductors which have generated considerable interest in early superconducting material history. However, the topological nature of the electronic structure remains unnoticed so far. Here, using first-principles band structure calculations based on density functional theory, we show that the A15 superconductors (Ta3Sb, Ta3Sn, and Ta3Pb) have nontrivial band topology in the bulk electronic band structures, leading to the formation of topological surface states near the Fermi energy. Due to the bulk superconductivity, the proximity effect in the topological surface states would induce topological superconductivity even without heterostructure of a topological insulator and an s-wave superconductor. Our results indicate that the A15 superconductors are promising candidates for the realization of the topological superconductivity and the Majorana fermion.

Topological phase of a matter has been a central theme of recent condensed matter physics [1, 2]. An intriguing aspect of the topological states is that they are promising platforms to realize novel excitations such as Majorana fermions in condensed matter systems whereas they are elusive in high-energy physics [3–12]. One of the propositions to realize Majorana fermions is making heterostructures between strong topological insulators and s-wave superconductors, where the proximity-induced superconductivity in the topological surface states is analogous to that in a spinless p<sup>x</sup> + ip<sup>y</sup> superconductor that is expected to have Majorana bound states at the vortices [4, 5]. A straightforward extension of the idea would be to search for conventional s-wave superconductors which also have topological surface states [13–15]. The realization of topological superconductivity in a single material is advantageous considering the possible complexity at the interfaces of heterostructures.

A natural strategy for the search in this direction would be to examine existing superconductor material families in view of topological band theory. In conventional Bardeen-Cooper-Schrieffer (BCS) superconductors, the band structure in the normal phase is metallic and sizable density-of-states exists at the Fermi level. In contrast, the topological characterization of the band structure requires a band gap in order to define a topological invariant of the occupied band manifold since the band topology is defined using continuous deformation of the Hamiltonian without closing a gap. However, we note that it is still possible to define the band topology as long as there is a gap in the energy spectrum at each k-point (i.e., separate band manifolds). While the band structure of the superconductor would have metallic bands at the Fermi level, it is possible to have topological surface states in part of the Brillouin zone (BZ) where the metallic bulk bands do not overlap with them for appropriate surface termination. Thus, if we find a superconductor which has gaps at every k-point in the BZ with nontrivial band topology, it would be a potential candidate for the topological superconductor.

Here, we show that superconductors with the A15 structure, which are representative metal-based superconductors mostly discovered from 1950's to 1970's [16–18], are promising candidates for topological superconductors. Specifically, Ta3Sb, Ta3Sn, and Ta3Pb are shown to have topological bulk band structures characterized by nontrivial Z<sup>2</sup> invariants [19]. We find that the topological surface bands appear near the Fermi energy as dictated by the bulk-boundary correspondence [1]. Interestingly, since we have lower symmetry in the [001] surface, the topological surface states show non-helical spin texture unlike, e.g., Bi2Se<sup>3</sup> [20, 21]. We also discuss the electronic band structures of Nb compounds in the A15 family.

We performed first-principles electronic structure calculations based on density functional theory as implemented in Vienna ab initio simulation package (VASP) [22, 23]. We employed the projector augmented-wave method [24], and for the exchange correlation functional we adopted Perdew-Burke-Ernzerhof revised for solid (PBEsol) [25]. We used a plane-wave basis set with the energy cutoff of 325 eV, and 10 × 10 × 10 k-point meshes were exploited. The experimental lattice constants were used [26–29] except for the case where we could not find an experimental value. For the surface state calculations, we used 21-layer-thick slab models in the [001] direction with sufficiently large vacuum regions (' 23 ˚A) to prevent the spurious interactions between the periodic images. The electronic structure was also checked and the symmetry property was analyzed using Quantum espresso package [30].

The superconductors with the A15 structure have cubic symmetry with the space group Pm3n #223 (Fig. 1c). They are intermetallic compounds with chemical formulae A3B where A atoms lie at the cube face of the unit cell forming mutually orthogonal one-dimensional chains along edges, and B atoms constitute a bcc lattice. The crystal structure belongs to a nonsymmorphic space group where for instance we have a symmetry operation involving a screw axis,

$$
\begin{pmatrix} x \\ y \\ z \end{pmatrix} \rightarrow \begin{pmatrix} \frac{1}{2} - y \\ \frac{1}{2} + x \\ \frac{1}{2} + z \end{pmatrix} = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y - \frac{1}{2} \\ z \end{pmatrix} + \begin{pmatrix} 0 \\ \frac{1}{2} \\ \frac{1}{2} \end{pmatrix},\tag{1}
$$

with the lattice constant set to 1 for simplicity. This operation is a 4-fold rotation around z axis with respect to (0, 1/2, 0) followed by a fractional translation along z direction. Accordingly, we have 4-fold rotational symmetry in the bulk electronic band structure although we do not have the strict 4-fold rotation in the crystal structure.

The electronic band structure of the representative superconductor Ta3Sb shows metallic feature with separate band manifolds, i.e., separation of the "valence" and the "conduction" bands in the whole BZ as shown in Fig. 1a. We find that spin-orbit coupling (SOC) is essential to have the separation of the band manifolds (Fig. 1a, b). The bands near the Fermi energy mostly have Ta d character. Since the crystal structure possesses the spatial inversion symmetry, the band topology of the valence bands can be calculated from the parities of the wavefunctions at the time-reversal invariant momenta (TRIM) [31]. The parity products are +,+,−, and + for Γ, 3X, 3M, and R, respectively, which results in a

![](_page_3_Figure_0.jpeg)

**Caption:** Figure 1 presents the electronic band structure and atomic structure of Ta3Sb. (a) shows the band structure along high-symmetry paths in the Brillouin Zone (BZ) with spin-orbit coupling (SOC) accounted for, revealing a separation between 'valence' and 'conduction' bands indicative of metallic characteristics. The inset highlights that bands near the Fermi energy predominantly originate from Ta d-orbitals. (b) displays the band structure without SOC, demonstrating the essential role of SOC in the band manifold separation. The crystal configuration is depicted in (c), indicating the cubic and nonsymmorphic space group Pm3n structural details of Ta3Sb. (d) illustrates parity products at time-reversal invariant momenta (TRIM) within the BZ, leading to a strong topological phase characterized by a Z2 index (ν0; ν1ν2ν3) = (1; 000). These findings underscore Ta3Sb's nontrivial topology and potential for topological surface state realization, important for quantum computing applications.


FIG. 1. Electronic band structure and atomic structure of Ta3Sb. The band structure along highsymmetry paths in the BZ (a) with and (b) without SOC. (c) The atomic structure of Ta3Sb. (d) The parity products in the BZ. The Fermi energy is set to 0.

strong topological phase with (ν0; ν1ν2ν3) = (1; 000) where ν<sup>0</sup> is the strong Z<sup>2</sup> index and ν1, ν2, ν<sup>3</sup> are the weak ones (Fig. 1d).

One of the direct consequences of the nontrivial bulk band topology is the existence of the topological surface bands. In the [001] surface of Ta3Sb, we indeed find topological surface states as shown in Fig. 2a. The number (mod 2) of crossings between the surface states and a line connecting two TRIM in the gaps is dictated by the projected parity products. For example, since we have different parities at Γ and X (i.e., + at Γ and − at X), an odd number of crossings occurs along Γ–X (Fig. 2a), which characterizes topological surface states. This confirms the bulk-boundary correspondence between the bulk topological invariants and the

![](_page_4_Figure_0.jpeg)

**Caption:** Figure 2 illustrates the surface states of Ta3Sb on the [001] surface. (a) Highlights the electronic band structure along high-symmetry paths in the surface BZ, with green circles identifying surface-related features and the inset projecting parity products at TRIM. Comparisons are drawn between surface and bulk states. Spin angular momentum textures are displayed, with (b) and (d) depicting the x-component and (c) and (e) the y-component. Red and blue circles indicate positive and negative values, respectively, with grey regions corresponding to bulk states. This complex spin configuration stems from the reduced symmetry in the [001] surface, further enhanced by SOC, emphasizing its potential in spintronic applications.


FIG. 2. Surface states of Ta3Sb in the [001] surface. (a) Electronic band structure along highsymmetry paths in the surface BZ. Green circles denote the surface contribution. The inset shows the projected parity products at TRIM. (b), (d) x-component of the spin angular momentum. (c), (e) y-component of the spin angular momentum. The red and blue circles denote the positive and negative values, respectively. The grey regions correspond to the projected bulk states.

surface state configurations.

The spin-polarized surface states have non-helical spin structure (as opposed to typical helical surface states, e.g., in Bi2Se3) due to the reduced symmetry at the [001] surface (Fig 2). In general, topological surface states are spin-polarized due to the SOC and the breaking of the inversion symmetry at the surface. The specific spin configuration is determined by the relevant symmetry of the terminated surface. Whereas we have the 4-fold symmetry in the bulk electronic bands, we have only 2-fold symmetry in the [001] surface since the symmetry involving the fractional translation is broken. The symmetry reduction at the surface is im-

![](_page_5_Figure_0.jpeg)

**Caption:** Figure 3 details the spin angular momentum texture near Γ in the [001] surface of Ta3Sb. In (a) and (b), the spin texture of the lower and upper bands are presented, highlighting a unique mixture of Rashba and Dresselhaus spin textures due to surface symmetry reduction from C4v to C2v. Schematic illustrations in (c) show a Rashba-type helical spin texture, while (d) denotes a linear Dresselhaus-type texture. The study provides insights into Dirac cone anisotropy and electronic configurations at the surface, important for understanding spin polarization effects in topological materials.


FIG. 3. Spin angular momentum texture near Γ in the [001] surface of Ta3Sb. The spin texture of the (a) lower and (b) upper bands. The schematic illustration of (c) helical (Rashba-type) spin texture (d) linear Dresselhaus-type one.

portant for the qualitative understanding of the spin configuration. To see this, we consider the effective Hamiltonian Heff = Akyσ<sup>x</sup> − Bkxσ<sup>y</sup> which is the general form under C2<sup>v</sup> (the relevant symmetry of the surface states near Γ) and the time-reversal symmetry up to linear order in k. Here, σ's are Pauli matrices that describe the spin degree of freedom. Heff gives an anisotropic Dirac cone with the energy eigenvalues E = ±|k| p A<sup>2</sup> sin<sup>2</sup> θ<sup>k</sup> + B<sup>2</sup> cos<sup>2</sup> θk, in which cos θ<sup>k</sup> = kx/|k| and sin θ<sup>k</sup> = ky/|k|. When the local k axis is rotated by π/4, the Hamiltonian takes the form αR(kyσ<sup>x</sup> − kxσy) + αD(kxσ<sup>x</sup> − kyσy) with α<sup>R</sup> = (A + B)/2 and α<sup>D</sup> = (A−B)/2. Here, the α<sup>R</sup> term gives rise to helical (Rashba-type) spin texture (Fig. 3c), and the α<sup>D</sup> term leads to linear Dresselhaus-type one (Fig. 3d) [32, 33]. Indeed, our firstprinciples calculations show that the resulting spin configuration of the surface states near the Fermi energy around Γ is a mixture of the two types of spin textures (Fig. 3a). Note that under higher symmetry C4<sup>v</sup> or C3<sup>v</sup> we should have A = B, i.e., only the α<sup>R</sup> (helical) term survives.

Now we discuss the proximity induced superconductivity in the surface states. Here, we consider the simplest case under the C2<sup>v</sup> symmetry and the time-reversal symmetry, where we have a single Dirac cone at the Γ point. If we substitute k 0 <sup>x</sup> = B v k<sup>x</sup> and k 0 <sup>y</sup> = A v ky with v = √ A<sup>2</sup> + B<sup>2</sup> in the effective Hamiltonian, the Hamiltonian becomes mathematically
analogous to the isotropic helical Dirac cone described by v(k 0 <sup>y</sup>σ<sup>x</sup> − k 0 <sup>x</sup>σy). The analysis can be given in a similar way with Ref. [5] except for the spin helicity. We consider the Cooper pairs tunneling into the surface states due to the proximity effect by introducing H<sup>p</sup> = ∆ψ † ↑ψ † <sup>↓</sup> + H.c. with ∆ = ∆0e iφ. Then the Bogoliubov–de Gennes (BdG) Hamiltonian is given by HBdG = 1 <sup>2</sup>Ψ†HBdGΨ where

$$\mathcal{H}\_{BdG} = \begin{pmatrix} -iv(\sigma\_x \partial\_y - \sigma\_y \partial\_x) - \mu & \Delta \\ \Delta^\* & iv(\sigma\_x \partial\_y - \sigma\_y \partial\_x) + \mu \end{pmatrix},\tag{2}$$

and Ψ = (ψ↑, ψ↓, ψ† ↓ , −ψ † ↑ ) <sup>T</sup> with µ being the chemical potential. By solving the BdG equation HBdGζ = Eζ for a vortex-like condition ∆ = ∆0(r)e iθ in polar coordinates, the zero energy bound states can be found. The solution for µ = 0 case is simple and can be written as

$$\zeta(r,\theta) = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \\ -1 \end{pmatrix} \exp(-\int\_0^r ds \frac{\Delta\_0(s)}{v}),\tag{3}$$

which represents the Majorana zero mode localized at the vortex.

The electronic band structure of Ta3Sn shows qualitatively the same band topology. Since we substitute Sb (Group VA in the periodic table) with Sn (Group IVA), the Fermi energy moves downwards below the surface states (Fig. 4). The experimental superconducting transition temperatures (Tc) of Ta3Sb and Ta3Sn are 0.7K and 8.3K, respectively [16]. We also find nontrivial band topology in Ta3Pb (Pb also belongs to Group IVA) which has T<sup>c</sup> = 17K [17]. The compounds with Sn or Pb have the advantage that they have higher T<sup>c</sup> while electron doping would be needed to adjust the Fermi level in the surface. The manipulation of the Fermi level in the surface layers could be possible using chemical substitution or liquid gating [34, 35].

Another variation of the composition would be the substitution of Ta with Nb since they belong to the same group in the periodic table. However, we find that the valence and the conduction bands in Nb3Sb and Nb3Sn are not separated unlike the Ta compounds probably due to the weaker SOC strength of Nb. Thus, the Z<sup>2</sup> bulk band topology is not properly defined in the Nb compounds.

![](0__page_7_Figure_0.jpeg)

**Caption:** Figure 4 analyzes the electronic structure of Ta3Sn. (a) Shows the bulk band structure along high-symmetry paths, revealing similar topology to Ta3Sb with separate band manifolds crucial for defining nontrivial topological invariants. (b) Depicts the [001] surface band structure, mapping surface states with green circles and setting the Fermi level at zero, reinforcing the compound's topological classification. The study supports Ta3Sn's inclusion as a potential candidate for hosting Majorana modes due to topological surface states, advancing its applicability in topological quantum computing.


FIG. 4. Electronic structure of Ta3Sn. (a) Bulk band structure along high-symmetry paths in Ta3Sn. (b) Band structure of the [001] surface. Green circles denote the surface contribution. The Fermi level is set to 0.

In conclusion, we reported hitherto unnoticed topological phases in A15 superconductors, Ta3Sb, Ta3Sn, and Ta3Pb. The topological band structure was characterized by the nontrivial strong Z<sup>2</sup> topological invariant ν<sup>0</sup> = 1. Corresponding topological surface states appear in the [001] surface, and they show non-helical spin texture due to the reduced symmetry at the surface. The proximity induced superconductivity at the surface would give rise to the Majorana zero mode. The topological surface bands of the proposed compounds could be experimentally verified using angle-resolved photoemission spectroscopy (ARPES) [1, 36]. Our study shows that the A15 superconductors are promising candidates for the realization of the topological superconductivity and Majorana fermions, and potentially useful for topological quantum computation.

## ACKNOWLEDGMENTS

We thank Suk Bum Chung for fruitful discussions. This work was supported by the U.S. Department of Energy (DOE), Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division. The research was performed at Ames Laboratory, which is operated for the U.S. DOE by Iowa State University under Contract No. DE-AC02-07CH11358. Computations were performed through the support of the National Energy Research Scientific Computing Center, which is a DOE Office of Science User Facility operated under Contract No. DE-AC02-05CH11231.

- [1] M. Z. Hasan and C. L. Kane, Rev. Mod. Phys. 82, 3045 (2010).
- [2] X.-L. Qi and S.-C. Zhang, Rev. Mod. Phys. 83, 1057 (2011).
- [3] E. Majorana, Nuovo Cimento 14, 171 (1937).
- [4] N. Read and D. Green, Phys. Rev. B 61, 10267 (2000).
- [5] L. Fu and C. L. Kane, Phys. Rev. Lett. 100, 096407 (2008).
- [6] G. E. Volovik, JETP Letters 90, 398 (2009).
- [7] J. D. Sau, R. M. Lutchyn, S. Tewari, and S. Das Sarma, Phys. Rev. Lett. 104, 040502 (2010).
- [8] J. Alicea, Phys. Rev. B 81, 125318 (2010).
- [9] J. Linder, Y. Tanaka, T. Yokoyama, A. Sudbø, and N. Nagaosa, Phys. Rev. Lett. 104, 067001 (2010).
- [10] M. Wimmer, A. R. Akhmerov, M. V. Medvedyeva, J. Tworzyd lo, and C. W. J. Beenakker, Phys. Rev. Lett. 105, 046803 (2010).
- [11] R. M. Lutchyn, J. D. Sau, and S. Das Sarma, Phys. Rev. Lett. 105, 077001 (2010).
- [12] Y. Oreg, G. Refael, and F. von Oppen, Phys. Rev. Lett. 105, 177002 (2010).
- [13] B. Yan, M. Jansen, and C. Felser, Nature Physics 9, 709 (2013).
- [14] G. Xu, B. Lian, P. Tang, X.-L. Qi, and S.-C. Zhang, Phys. Rev. Lett. 117, 047001 (2016).
- [15] S.-Y. Guan, P.-J. Chen, M.-W. Chu, R. Sankar, F. Chou, H.-T. Jeng, C.-S. Chang, and T.-M. Chuang, Science Advances 2 (2016), 10.1126/sciadv.1600894.
- [16] A. Narlikar, Superconductors (OUP Oxford, 2014).
- [17] K. Buschow, Concise Encyclopedia of Magnetic and Superconducting Materials, Advances in

Materials Sciences and Engineering (Elsevier Science, 2005).

- [18] H. Hosono, A. Yamamoto, H. Hiramatsu, and Y. Ma, Materials Today (2017), https://doi.org/10.1016/j.mattod.2017.09.006.
- [19] L. Fu, C. L. Kane, and E. J. Mele, Phys. Rev. Lett. 98, 106803 (2007).
- [20] H. Zhang, C.-X. Liu, X.-L. Qi, X. Dai, Z. Fang, and S.-C. Zhang, Nature Physics 5, 438 (2009).
- [21] W. Zhang, R. Yu, H.-J. Zhang, X. Dai, and Z. Fang, New Journal of Physics 12, 065013 (2010).
- [22] G. Kresse and J. Hafner, Phys. Rev. B 47, 558 (1993).
- [23] G. Kresse and J. Furthm¨uller, Phys. Rev. B 54, 11169 (1996).
- [24] P. E. Bl¨ochl, Phys. Rev. B 50, 17953 (1994).
- [25] J. P. Perdew, A. Ruzsinszky, G. I. Csonka, O. A. Vydrov, G. E. Scuseria, L. A. Constantin, X. Zhou, and K. Burke, Phys. Rev. Lett. 100, 136406 (2008).
- [26] H. L. Luo, E. Vielhaber, and E. Corenzwit, Zeitschrift f¨ur Physik A Hadrons and nuclei 230, 443 (1970).
- [27] T. H. Courtney, G. W. Pearsall, and J. Wulff, Journal of Applied Physics 36, 3256 (1965).
- [28] S. Furuseth and A. Kjekshus, Acta Chem. Scand. 18, 1180 (1964).
- [29] I. Guseva, Y. Seropegin, and E. Sokolovskaya, Journal of the Less Common Metals 87, 109 (1982).
- [30] P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, G. L. Chiarotti, M. Cococcioni, I. Dabo, A. D. Corso, S. de Gironcoli, S. Fabris, G. Fratesi, R. Gebauer, U. Gerstmann, C. Gougoussis, A. Kokalj, M. Lazzeri, L. Martin-Samos, N. Marzari, F. Mauri, R. Mazzarello, S. Paolini, A. Pasquarello, L. Paulatto, C. Sbraccia, S. Scandolo, G. Sclauzero, A. P. Seitsonen, A. Smogunov, P. Umari, and R. M. Wentzcovitch, Journal of Physics: Condensed Matter 21, 395502 (2009).
- [31] L. Fu and C. L. Kane, Phys. Rev. B 76, 045302 (2007).
- [32] M. Kim, J. Ihm, and S. B. Chung, Phys. Rev. B 94, 115431 (2016).
- [33] A. Stroppa, D. Di Sante, P. Barone, M. Bokdam, G. Kresse, C. Franchini, M.-H. Whangbo, and S. Picozzi, Nat Comms 5, 5900 (2014).
- [34] J. T. Ye, S. Inoue, K. Kobayashi, Y. Kasahara, H. T. Yuan, H. Shimotani, and Y. Iwasa, Nature Materials 9, 125 (2010).
- [35] J. Jeong, N. Aetukuri, T. Graf, T. D. Schladt, M. G. Samant, and S. S. P. Parkin, Science 339, 1402 (2013), http://science.sciencemag.org/content/339/6126/1402.full.pdf.
- [36] D. Hsieh, D. Qian, L. Wray, Y. Xia, Y. S. Hor, R. J. Cava, and M. Z. Hasan, Nature 452, 970 (2008).