# arXiv:2103.04960

**Paper ID:** c20eb63a4661f60d6213e2b2ec639a84

**PDF Path:** apl/Superconductors/arXiv:2103.04960.pdf

**Processing Status:** complete

**Captions Added:** 11

**Generated:** 2025-06-24T13:44:27.510785

---

# arXiv:2103.04960v1 [cond-mat.str-el] 8 Mar 2021

# Spin-orbital model for fullerides

Ryuta Iwazaki and Shintaro Hoshino

Department of Physics, Saitama University, Shimo-Okubo, Saitama 338-8570, Japan

(Dated: February 25, 2022)

The multiorbital Hubbard model in the strong coupling limit is analyzed for the effectively antiferromagnetic Hund's coupling relevant to fulleride superconductors with three orbitals per molecule. The localized spin-orbital model describes the thermodynamics of the half-filled (three-electron) state with total spin-1/2, composed of singlon and doublon placed on the two of three orbitals. The model is solved using the mean-field approximation and magnetic and electric ordered states are clarified through the temperature dependences of the order parameters. Combining the model with the band structure from ab initio calculation, we also semi-quantitavely analyze the realistic model and the corresponding physical quantities. In the A15-structure fulleride model, there is an antiferromagnetic ordered state, and subsequently the two orbital ordered state appears at lower temperatures. It is argued that the origin of these orbital orders is related to the T<sup>h</sup> point group symmetry. As for the fcc-fulleride model, the time-reversal broken orbital ordered state is identified. Whereas the spin degeneracy remains in our treatment for the geometrically frustrated lattice, it is expected to be lifted by some magnetic ordering or quantum fluctuations, but not by the spin-orbital coupling which is effectively zero for fullerides in the strong-coupling regime.

# I. INTRODUCTION

Strongly correlated electron systems with multiple orbital degrees of freedom show a variety of intriguing phenomena, and are realized in a wide range of materials such as iron-pnictides, heavy-electron materials, and molecular-based organic materials. The alkali-doped fullerides are also the typical cases where the strong correlation effects with multiorbitals are relevant. This material has been attracting attention recent years for a lot of experimental findings. The superconductivity with the high transition temperature ∼ 40K is one of the characteristic feature [\[1](#page-13-0)[–8\]](#page-14-0). While the mechanism is identified as the electron-phonon interaction [\[9–](#page-14-1)[11\]](#page-14-2), the superconducting dome in the temperature-pressure phase diagram is found to be located near the Mott insulator and antiferromagnetic phase, featuring the typical behaviors of the strongly correlated superconductors [\[7,](#page-14-3) [12–](#page-14-4)[14\]](#page-14-5). In the Mott insulating phase, the localized electrons form a lowspin state and the imbalance of the occupancy in orbitals lead to the deformation of the fullerene molecule because of the coupling between electrons and anisotropic molecular distortions (Jahn-Teller phonon). Interestingly, such behavior can also be seen in the metallic phase near the Mott insulator but is absent far away from it [\[15,](#page-14-6) [16\]](#page-14-7). This anomalous behavior is called the Jahn-Teller metal where the multiorbital degrees of freedom play an important role. The fullerides are also crystallized on the substrate and the characteristic asymmetry between electron and hole doping is identified [\[17,](#page-14-8) [18\]](#page-14-9). Furthermore, a possible superconducting state has been discussed under the excitation by light above the transition temperature [\[19,](#page-14-10) [20\]](#page-14-11). Thus, the fulleride materials have been providing the intriguing phenomena up until recently.

The alkali-doped fullerides are the systems with triply degenerate t1<sup>u</sup> molecular orbitals which resembles atomic p-electrons in nature. There, the Hund's coupling, which is usually acting ferromagnetically on the electrons located at the different orbitals, is effectively antiferromagnetic due to the coupling to the anisotropic molecular vibrations [\[10,](#page-14-12) [21,](#page-14-13) [22\]](#page-14-14) and is crucial for the low-temperature physics. The multiorbital Hubbard model with the antiferromagnetic Hund's coupling has been studied theoretically, and the various phase diagrams are clarified using the dynamical mean-field theory suitable for the description of the electronically ordered states [\[21](#page-14-13)[–29\]](#page-14-15). The Jahn-Teller metal has been interpreted as the spontaneous orbital selective Mott state [\[26,](#page-14-16) [30\]](#page-14-17) which is an unconventional type of orbital order. The orbital asymmetric feature has also been reported in two-dimensional fullerides by using the manyvariable variational Monte Carlo method [\[31\]](#page-14-18).

With the antiferromagnetic Hund's coupling, one of the intra-molecular interaction, pair hopping, plays an important role: it activates the dynamics of the double occupancy in an orbital (doublon). In order to clarify the characters of the existing fulleride materials in detail, we focus our attention on the Mott insulating phase, where the doublon physics can be tackled with reasonable computational cost even in the realistic situation. As is well known, for a single-orbital case, the electronic behaviors in the strong coupling regime are determined by the Heisenberg model of localized electrons. The extension of the Heisenberg model to the multiorbital system is known as the Kugel-Khomskii model which has been derived for the ferromagnetic Hund's coupling [\[32,](#page-14-19) [33\]](#page-14-20) and describes the degrees of freedom of the spin and orbital. The spin-orbital models have been applied to the e<sup>g</sup> or t2<sup>g</sup> orbital system [\[34](#page-14-21)[–37\]](#page-14-22). On the other hand, the fullerides have antiferromagnetic Hund's coupling, so that their strongly correlated effective model differs from the usual Kugel-Khomskii model. While the localized model with antiferromagnetic Hund's coupling have been constructed for a density-density type interaction [\[28\]](#page-14-23), here we deal with more complicated but realistic situations.

In this paper, we develop the localized spin-orbital

model for the system with antiferromagnetic Hund's coupling. We analyze both the symmetric model and the realistic model for fullerides, the former of which is easier to interpret the results and is useful as a reference. By using the mean field theory, for the spherical model on a bipartite lattice, we obtain the staggered magnetic ordered state, and also the uniform orbital ordered state at lower temperature regime. This orbital ordered state is not characterised by the ordinary orbital moment but by the doublon's orbital moment. In the A15 fulleride effective model, which is bipartite lattice, we reveal that there are two kinds of orbital ordered states below the antiferromagnetic transition temperature. The obtained orbital ordered states are interpreted as related to an effective recovery of the four-fold symmetry at low temperatures in the T<sup>h</sup> point group. We also analyze the geometrically frustrated fcc fulleride model seeking for a spatially uniform ordered state. We reveal that the fcc model has the time-reversal symmetry broken orbital ordered state, where the spin ordered state is absent since the spin-orbit coupling on the fullerene molecule is effectively zero.

This paper is organized as follows. We discuss the construction of strongly correlated effective models and the theoretical method in Sec. [II.](#page-1-0) In Sec. [III,](#page-4-0) we show numerical results for the model with isotropic hopping (spherical model introduced in Sec. [III A\)](#page-4-1). Section [IV](#page-10-0) provides numerical results for the spin-orbital model combined with A15 and fcc fulleride band structure. We summarize the results in Sec. [V.](#page-13-1)

# <span id="page-1-0"></span>II. CONSTRUCTION OF MODELS

# <span id="page-1-5"></span>A. Three orbital Hubbard model in strong-coupling limit

Let us begin with the three-orbital Hubbard model

$$
\mathcal{H} = \mathcal{H}\_t + \mathcal{H}\_U,\tag{1}
$$

$$\mathcal{H}\_t = -\sum\_{i \neq j, \gamma, \gamma', \sigma} t\_{ij}^{\gamma \gamma'} c\_{i, \gamma, \sigma}^\dagger c\_{j, \gamma', \sigma}, \tag{2}$$

$$\begin{split} \mathcal{H}\_{U} &= \frac{U}{2} \sum\_{i,\gamma,\sigma,\sigma'} c\_{i,\gamma,\sigma}^{\dagger} c\_{i,\gamma,\sigma'}^{\dagger} c\_{i,\gamma,\sigma'} c\_{i,\gamma,\sigma'} c\_{i,\gamma,\sigma} \\ &+ \frac{U'}{2} \sum\_{i,\gamma \neq \gamma',\sigma,\sigma'} c\_{i,\gamma,\sigma}^{\dagger} c\_{i,\gamma',\sigma'}^{\dagger} c\_{i,\gamma',\sigma'} c\_{i,\gamma',\sigma'} \alpha \\ &+ \frac{J}{2} \sum\_{i,\gamma \neq \gamma',\sigma,\sigma'} \left( c\_{i,\gamma,\sigma}^{\dagger} c\_{i,\gamma',\sigma'}^{\dagger} c\_{i,\gamma,\sigma'} c\_{i,\gamma',\sigma'} \right) \\ &+ c\_{i,\gamma,\sigma}^{\dagger} c\_{i,\gamma,\sigma'}^{\dagger} c\_{i,\gamma',\sigma'} c\_{i,\gamma',\sigma'} \alpha \gamma \end{split} (3)$$

where ci,γ,σ (c † i,γ,σ) is an annihilation (creation) operator at site i of fullerenes with the t1<sup>u</sup> molecular orbital index γ = x, y, z and spin σ =↑, ↓. We deal with the Hilbert space with a fixed number of electrons. We assume the condition U <sup>0</sup> <sup>=</sup> <sup>U</sup> <sup>−</sup> <sup>2</sup><sup>J</sup> for the local interaction part in the following discussion, which is valid for the

![](_page_1_Figure_10.jpeg)

**Caption:** Figure 8: Displaying the temperature dependencies of order parameters and thermodynamic properties for different models, panel (a) shows the single-sublattice n = 3 model at J/U = -0.1. Panels (b) and (c) compare entropy and specific heat trends under uniform conditions and single-sublattice scenarios with coupling ratio r = -0.4, highlighting the spontaneous spin-orbit coupling reflected in the ground state properties and the relationship between coupling constants and emergent orders .


<span id="page-1-1"></span>FIG. 1. Schematic pictures for the ground state wave functions |γ, σ =↑i<sup>i</sup> of the local Hamiltonian for n = 3 and n = 1.

spherical limit. In this paper, we consider a strong coupling regime (H<sup>U</sup> Ht). When we develop the effective model in this limit, the presence of the Hund's coupling J makes theoretical treatment complicated since it realizes quantum-mechanically superposed local wave functions. Especially for the negative (antiferromagnetic) J relevant to fullerides, the pair hopping plays an important role which creates the dynamics of doubly occupied electrons at an orbital (doublon). As shown in the following, in order to diminish the difficulty, we use a symbolic expression without elaborating each intermediate process explicitly.

In order to apply the perturbation theory from the strong coupling limit, we first consider the ground state of the unperturbed Hamiltonian H<sup>U</sup> . Alkali-doped fullerides with half-filled situation (three electrons per t1<sup>u</sup> orbital) have six-fold degenerate ground states written as

$$\left|\gamma,\sigma\right\rangle\_i = \frac{1}{\sqrt{2}} c\_{i,\gamma,\sigma}^\dagger \sum\_{\gamma' \neq \gamma} b\_{i,\gamma'}^\dagger \left|0\right\rangle,\tag{4}$$

<span id="page-1-4"></span>where we have defined an orbital-dependent doubloncreation operator as

<span id="page-1-2"></span>
$$b\_{i,\gamma}^{\dagger} = c\_{i,\gamma,\downarrow}^{\dagger} c\_{i,\gamma,\uparrow}^{\dagger}.\tag{5}$$

The vacuum has been expressed as |0i. These states are uniquely characterized by the spin and orbital of the electron at the singly occupied orbital, which is called 'singlon' to make contrast against doublons. The schematic picture of the three-electron state |γ, σ =↑i<sup>i</sup> is illustrated in Fig. [1.](#page-1-1)

Using the above Hamiltonian, the second-order effective Hamiltonian is written as

<span id="page-1-3"></span>
$$\mathcal{H}\_{\text{eff}} = \mathcal{P}\mathcal{H}\_t \frac{1}{-\mathcal{H}\_U} \mathcal{Q}\mathcal{H}\_t \mathcal{P},\tag{6}$$

where P is a projection operator to a model space described by Eq. [\(4\)](#page-1-2) as

$$\mathcal{P} = \sum\_{i,\gamma,\sigma} |\gamma,\sigma\rangle\_i \, {}\_i\langle \gamma,\sigma|,\tag{7}$$

and Q = 1 − P. We have used [P, H<sup>U</sup> ] = 0. The energy is measured from the ground state of H<sup>U</sup> . The size of our model space is 6<sup>N</sup> where N = P i 1 is the number of lattice sites.

The strategy for obtaining the concrete form of the effective Hamiltonian is to consider the two-site problem. We first prepare the 2<sup>12</sup> <sup>×</sup> <sup>2</sup> <sup>12</sup> matrix expressions for the annihilation and creation operators for two-site problem (12 = P i,γ,σ 1), and then define all of the matrix expressions given in Eq. [\(6\)](#page-1-3). Performing multiplications of such matrices, we obtain the two-site effective Hamiltonian in the form of the 6<sup>2</sup> <sup>×</sup> <sup>6</sup> <sup>2</sup> matrix. We expand the above effective hamiltonian by following local operators O ηµ i defined as

$$O\_i^{\eta\mu} = \sum\_{\gamma,\gamma'} \sum\_{\sigma,\sigma'} |\gamma,\sigma\rangle\_i \lambda\_{\gamma\gamma'}^{\eta} \sigma\_{\sigma\sigma'}^{\mu}{}\_i \langle \gamma',\sigma'|,\tag{8}$$

in the model Hilbert space. σ <sup>µ</sup>=0,x,y,z is Pauli matrix

$$\begin{aligned} \sigma^0 &= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, & \sigma^x &= \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \\ \sigma^y &= \begin{pmatrix} 0 & -\mathbf{i} \\ \mathbf{i} & 0 \end{pmatrix}, & \sigma^z &= \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, & \end{aligned} \tag{9}$$

which represents the degrees of freedom of the spin. Another matrix λ η=0,··· ,8 is given by

$$
\begin{aligned}
\lambda^0 &= \sqrt{\frac{2}{3}} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \qquad \lambda^1 = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \\
\lambda^2 &= \begin{pmatrix} 0 & -1 & 0 \\ \mathbf{i} & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \qquad \lambda^3 = \begin{pmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \\
\lambda^4 &= \begin{pmatrix} 0 & 0 & -1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \qquad \lambda^5 = \begin{pmatrix} 0 & 0 & \mathbf{i} \\ 0 & 0 & 0 \\ -\mathbf{i} & 0 & 0 \end{pmatrix}, \\
\lambda^6 &= \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & -1 & 0 \end{pmatrix}, \qquad \lambda^7 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & \mathbf{i} & 0 \end{pmatrix}, \\
\lambda^8 &= \sqrt{\frac{1}{3}} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -2 \end{pmatrix}, \end{aligned} \tag{10}$$

where these matrices are slightly different from ordinary definition of the Gell-Mann matrices to make them suitable for p-electron systems. We note that the above local operators satisfy the orthonormal relation

$$\operatorname{Tr}\left[O\_i^{\eta\mu}O\_j^{\eta'\mu'}\right] = 4\delta\_{ij}\delta^{\eta\eta'}\delta^{\mu\mu'}.\tag{11}$$

Thus, the set of operators O ηµ i is regarded as a basis set of the extended Hilbert space (Liouville space). In contrast, the states |γ, σi<sup>i</sup> are the basis in the six-component model Hilbert space. Extending the two-site problem to the full lattice, we obtain the effective Hamiltonian in the strong coupling limit

$$\mathcal{H}\_{\text{eff}} = \sum\_{i,j} \sum\_{\eta,\eta'} \sum\_{\mu,\mu'} I\_{ij}^{\eta\mu;\eta'\mu'} O\_i^{\eta\mu} O\_j^{\eta'\mu'}.\tag{12}$$

This model is to be analyzed in the rest of this paper.

We also comment on the orbital moments in the restricted Hilbert space. In terms of the original Hubbard model, the local orbital moment is defined by

$$\mathcal{L}\_i \equiv \sum\_{\gamma,\gamma',\sigma} c\_{i,\gamma,\sigma}^\dagger \mathcal{E}\_{\gamma\gamma'} c\_{i,\gamma',\sigma},\tag{13}$$

where the 3×3 matrices are given by `<sup>x</sup> = λ 7 , `<sup>y</sup> = λ 5 , and `<sup>z</sup> = λ 2 . This angular momentum operator is, however, zero for the restricted Hilbert space:

$$\mathcal{P}\mathcal{L}\_i\mathcal{P} = \mathbf{0}.\tag{14}$$

<span id="page-2-1"></span>This anomalous disappearance of the angular momentum is due to the composite nature of the ground state [\[26\]](#page-14-16) and is very different from a singly occupied state. Then the active orbital degrees of freedom are not of the original electrons but of the three-electron composite involving doublons. This feature also affects the spin-orbit coupling which takes the form

$$\mathcal{H}\_{\rm SO} = \frac{1}{2} \lambda\_{\rm SO} \sum\_{i} \sum\_{\gamma,\gamma'} \sum\_{\sigma,\sigma'} c\_{i,\gamma,\sigma}^{\dagger} \ell\_{\gamma\gamma'} \cdot \sigma\_{\sigma\sigma'} c\_{i,\gamma',\sigma'}, \quad (15)$$

in the language of the original multiorbital Hubbard model. The spin-orbit coupling for 2p-electron in carbon atom is nearly 2meV, and because of the extended nature of the fullerene molecular the spin-orbit coupling λSO for t1<sup>u</sup> orbitals is one-hundred times smaller than the atomic value (λSO ∼ 20µeV) [\[38\]](#page-14-24). Furthermore, for the restricted Hilbert space of n = 3 states, the effect of the spin-orbit coupling enters only through the second-order perturbation contribution as

$$\mathcal{H}\_{\rm SO}^{(2)} = \mathcal{P}\mathcal{H}\_{\rm SO} \frac{1}{-\mathcal{H}\_U} \mathcal{Q}\mathcal{H}\_{\rm SO}\mathcal{P} \tag{16}$$

$$\Gamma = \frac{1}{2} \Lambda\_{\rm SO} \sum\_{i} \sum\_{\gamma,\gamma'} \sum\_{\sigma,\sigma'} |\gamma,\sigma\rangle\_i \ell\_{\gamma\gamma'} \cdot \sigma\_{\sigma\sigma'} \, \_i\langle \gamma',\sigma'|, \,\,(17)$$

where ΛSO = 11λ 2 SO 20J for J < 0. Using the values for the antiferromagnetic coupling J ∼ −0.03eV for fullerdies [\[39\]](#page-14-25), we obtain ΛSO ∼ 1neV which is so tiny. Hence we can safely neglect the spin-orbit coupling in fullerides.

It is convenient to recognize that the above three electron state is similar to the singly occupied state

<span id="page-2-0"></span>
$$\left|n=1,\gamma,\sigma\right\rangle\_i = c\_{i,\gamma,\sigma}^\dagger \left|0\right\rangle,\tag{18}$$

which is the eigenstate with n<sup>i</sup> = P γ,σ c † i,γ,σci,γ,σ = 1 regardless of the sign of J (see the right column of Fig. [1\)](#page-1-1). In our paper, the number of electrons is fixed at each site and n<sup>i</sup> is sometimes simply written as n. In Eq. [\(18\)](#page-2-0), we explicitly write 'n = 1', and if it is dropped, the state represents n = 3 state defined in Eq. [\(4\)](#page-1-2). The ground state for n = 3 is obtained by filling the empty orbital in n = 1 state by the doublons as in Eq. [\(4\)](#page-1-2).

We will consider the n = 1 case for reference to illuminate the characteristics of n = 3 relevant to fullerides. When we deal with the second-order effective Hamiltonian for the n = 1 states, we just replace |γ, σi i by |n = 1, γ, σi i defined in Eq. [\(18\)](#page-2-0). We note that, in this case, the angular momentum does not vanish as distinct from the n = 3 multiplet. For the usual ferromagnetic Hund's coupling (J > 0), the system corresponds to the spin-orbital model considered for the t2<sup>g</sup> orbitals [\[37\]](#page-14-22).

# B. Mean field approximations

In this paper, we utilize the mean field approximation (MFA) for the obtained effective Hamiltonian. We apply the external field for convenience and the full Hamiltonian is written as

$$\mathcal{H}\_{\text{eff}} = \frac{1}{2} \sum\_{i,j} \vec{O}\_i^{\text{T}} \hat{I}\_{ij} \vec{O}\_j - \sum\_i \vec{H}\_i^{\text{T}} \vec{O}\_i \tag{19}$$

$$\approx -\sum\_{i,j} \left[ \vec{H}\_i^{\rm T} \delta\_{ij} \hat{1} - \frac{1}{2} \vec{\mathcal{M}}\_i^{\rm T} \left( \hat{I}\_{ij} + \hat{I}\_{ji}^{\rm T} \right) \right] \vec{O}\_j$$

$$-\frac{1}{2} \sum\_{i,j} \vec{\mathcal{M}}\_i^{\rm T} \hat{I}\_{ij} \vec{\mathcal{M}}\_j \equiv \mathcal{H}^{\rm MF},\tag{20}$$

where the hat and arrow symbols represent the matrix and vector, respectively, with respect to the intra-site degrees of freedom (η, µ). The vector O~ i is the operator for the order parameter at site i, whose matrix representation is given in Eq. [\(8\)](#page-2-1). Namely, it is a column vector having 35 components, each of which is a 6×6 matrix where the identity is eliminated. The statistical average <sup>M</sup><sup>~</sup> <sup>i</sup> <sup>=</sup> <sup>h</sup>O~ <sup>i</sup>i is the order parameter. In this paper, the coupling constant ˆIij connects only nearest-neighbor (NN) sites for the spherical model (Sec. [III\)](#page-4-0), and NN and next-nearest-neighbor (NNN) site for A15 and fcc fulleride model (Sec. [IV\)](#page-10-0). In the following of this section, we concentrate on the bipartite lattice such as A15 structure. Then we introduce two kinds of AB-sublattice to describe staggered orders. For non-bipartite lattice (i.e. fcc), on the other hand, we consider only the uniform solution and the similar formula can easily be obtained by regarding the two sublattices as identical.

The mean-field Hamiltonian is then rewritten as

$$\begin{split} \mathcal{H}^{\text{MF}} &= -\sum\_{\alpha} \left[ \vec{H}\_{\alpha}^{\text{T}} - \frac{1}{2} \sum\_{\delta \in \text{NN}} \vec{\mathcal{M}}\_{\alpha}^{\text{T}} \left( \hat{I}\_{\delta,0} + \hat{I}\_{0,\delta}^{\text{T}} \right) \right. \\ & \left. - \frac{1}{2} \sum\_{\delta \in \text{NNN}} \vec{\mathcal{M}}\_{\alpha}^{\text{T}} \left( \hat{I}\_{\delta,0} + \hat{I}\_{0,\delta}^{\text{T}} \right) \right] \sum\_{i \in \alpha}^{N/2} \vec{O}\_{i} \\ & - \frac{1}{2} \frac{N}{2} \sum\_{\alpha} \left[ \sum\_{\delta \in \text{NN}} \vec{\mathcal{M}}\_{\alpha}^{\text{T}} \hat{I}\_{\delta,0} \vec{\mathcal{M}}\_{\alpha} + \sum\_{\delta \in \text{NNN}} \vec{\mathcal{M}}\_{\alpha}^{\text{T}} \hat{I}\_{\delta,0} \vec{\mathcal{M}}\_{\alpha} \right], \end{split} \tag{21}$$

where α = A, B is the sub-lattice index and ¯α is a complementary component of α, i.e., A = B and ¯ B = A. ¯ N is the number of site. The number of δ ∈ NN is z, 8 or 12 respectively for the spherical, A15 or fcc model. As for δ ∈ NNN, both the A15 case (and fcc) has six sites. We have used the fact that NN-connected sites belong to the different sub-lattices and the NNN-connected sites belongs to the same sub-lattice. Since the coupling constants are dependent only on the direction of the vector connecting two sites, we write the interaction parameter as ˆIδ,0, where the index 0 represents the site which we focus on.

For the bipartite lattice, we introduce the uniform and staggered moments as

$$
\begin{pmatrix} \vec{\mathcal{M}}\_{\rm u} \\ \vec{\mathcal{M}}\_{\rm s} \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} \hat{1} & \hat{1} \\ \hat{1} & -\hat{1} \end{pmatrix} \begin{pmatrix} \vec{\mathcal{M}}\_{\rm A} \\ \vec{\mathcal{M}}\_{\rm B} \end{pmatrix}. \tag{22}
$$

This expression is useful in analyzing the mean-field solutions shown later.

Now we explain the method of numerical calculation. The solutions are obtained by renewing the order parameters iteratively using the self-consistent equation. The free energy and the self-consistent equation are given by

$$\mathcal{F} = -T \ln Z,\tag{23}$$

$$
\vec{\mathcal{M}}\_{\alpha} = -\frac{\partial \mathcal{F}}{\partial \vec{H}\_{\alpha}},\tag{24}
$$

where Z = Tr e<sup>−</sup>βHMF is the partition function made of the mean-field Hamltonian. For the derivation of the selfconsistent equation, the parameters H~ and <sup>M</sup><sup>~</sup> must be regarded as independent variables.

The system with the present effective Hamiltonian has 35 kinds of order parameters per site, and there may exist several solutions which take the same free energy as they are connected by symmetries. In the next sections, we show the simplest form of the order parameters among those energetically degenerate solutions.

# C. Response functions

In this subsection, we consider the response function to the weak static field. We expand the mean-field Hamiltonian up to first order of the field

$$\mathcal{H}^{\rm MF} = \mathcal{H}^{(0)} + \mathcal{H}^{(1)} + \mathcal{O}\left(H^2\right),\tag{25}$$

$$\mathcal{H}^{(0)} = \sum\_{i,j} \left(\vec{\mathcal{M}}\_i^{(0)}\right)^T \hat{I}\_{ij} \vec{O}\_j,\tag{26}$$

$$\mathcal{H}^{(1)} = -\sum\_{i,j} \left[ \vec{H}\_i^T \delta\_{ij} - \left( \vec{\mathcal{M}}\_i^{(1)} \right)^T \hat{I}\_{ij} \right] \vec{O}\_j,\qquad(27)$$

where the superscript represents the perturbative order of the field and we have neglected the constant term. When we define the effective field as H ~˜ <sup>i</sup> = H~ <sup>i</sup> <sup>−</sup> <sup>P</sup> j ˆI T jiM<sup>~</sup> (1) j and treat <sup>H</sup>(1) as perturbation, we obtain the following linear response relation

$$
\vec{\mathcal{M}}\_i^{(1)} = \sum\_j \hat{\chi}\_{ij}^{(0)} \vec{\tilde{H}}\_j = \sum\_j \hat{\chi}\_{ij} \vec{H}\_j,\tag{28}
$$

where ˆχ is the full susceptibility for the bare external field H~ i . According to linear response theory, the zeroth-order susceptibility is obtained by

$$\hat{\chi}\_{ij}^{(0)} = \int\_0^{1/T} \mathrm{d}\tau \left[ \left< T\_\tau \vec{O}\_i \vec{O}\_j^T(\tau) \right>\_0 - \vec{\mathcal{M}}\_i^{(0)} \left( \vec{\mathcal{M}}\_j^{(0)} \right)^T \right],\tag{29}$$

where τ is an imaginary time and T<sup>τ</sup> is imaginary time ordering operator. The Heisenberg picture in an imaginary time is expressed as

$$
\vec{O}\_i(\tau) = \mathbf{e}^{\tau \mathcal{H}^{(0)}} \vec{O}\_i \mathbf{e}^{-\tau \mathcal{H}^{(0)}}.\tag{30}
$$

h· · ·i<sup>0</sup> represents the statistical average with <sup>H</sup>(0). The susceptibility matrix ˆχ (0) ij has only intra-site component since each site is independent under MFA. Substituting the concrete expression to the effective field in Eq. [\(28\)](#page-4-2), we obtain

$$\sum\_{j} \left[ \delta\_{ij} \hat{1} + \sum\_{k} \hat{\chi}\_{ik}^{(0)} \hat{I}\_{kj}^{\mathrm{T}} \right] \vec{\mathcal{M}}\_{j}^{(1)} = \sum\_{j} \hat{\chi}\_{ij}^{(0)} \vec{H}\_{j}. \tag{31}$$

Then, taking matrix inverse of the left hand side and combining it with Eq. [\(28\)](#page-4-2), we obtain the susceptibility matrix ˆχij . For a bipartite lattice, we introduce the uniform and staggered susceptibilities by

$$
\hat{\chi}\_{\mathbf{u}} = \frac{1}{N} \sum\_{i,j} \hat{\chi}\_{ij},\tag{32}
$$

$$
\hat{\chi}\_s = \frac{1}{N} \sum\_{i,j} s\_i s\_j \hat{\chi}\_{ij},\tag{33}
$$

where s<sup>i</sup> = +1 for i ∈ A and s<sup>i</sup> = −1 for i ∈ B. This quantity will be shown in the next section. Although we focus on the static response functions in this paper, the above argument can easily be generalized for the dynamical susceptibility which captures the magnetic and electric dynamics of the localized model.

From the view point of Landau theory, we can also discuss the stability of the solution based on the susceptibilities. We write down the Landau free energy with an order parameter up to second order as

$$\mathcal{F}\_{\rm L} = \frac{1}{2} \sum\_{i,j} \vec{\mathcal{M}}\_i^{\rm T} \hat{a}\_{ij} \vec{\mathcal{M}}\_j - \sum\_i \vec{H}\_i^{\rm T} \vec{\mathcal{M}}\_i,\tag{34}$$

where ˆaij is a coefficient of the quadratic term. Note that, here, <sup>M</sup><sup>~</sup> is defined as the deviation from its equilibrium point. Then we obtain the following equation of states:

$$\sum\_{j} \hat{a}\_{ij} \vec{\mathcal{M}}\_{j} = \vec{H}\_{i}. \tag{35}$$

<span id="page-4-2"></span>Comparing the linear response function, we find that the Hessian matrix is identical to the inverse susceptibility:

$$\frac{\partial^2 \mathcal{F}\_{\rm L}}{\partial \vec{\mathcal{M}}\_i \partial \vec{\mathcal{M}}\_j} = \hat{a}\_{ij} = (\hat{\chi}^{-1})\_{ij}. \tag{36}$$

We can consider the necessary and sufficient condition for the stable solution. Let ε<sup>n</sup> be the n-th eigenvalue of the matrix ˆaij . Each energy corresponds to the eigenenergy of the excitation modes. We must have the condition

<span id="page-4-3"></span>
$$
\varepsilon\_n \ge 0,\tag{37}
$$

for all n, if the system is thermodynamically stable. If ε<sup>n</sup> = 0 is obtained, it indicates the presence of the Nambu-Goldstone mode. With use of Eq. [\(36\)](#page-4-3), in the actual calculations, we obtain ε<sup>n</sup> by diagonalizing the inverse susceptibility matrix.

# <span id="page-4-0"></span>III. NUMERICAL RESULTS FOR SPHERICAL MODELS

In the following of this paper, we will encounter the successive phase transitions with decreasing temperature. There, we denote each transition temperature as Tc<sup>1</sup> > Tc<sup>2</sup> > · · · . If there is only one transition temperature is identified, we use T<sup>c</sup> to denote it. Note that we use the same symbol for the transition temperatures in different models.

# <span id="page-4-1"></span>A. Spherical spin-orbital model

<span id="page-4-5"></span><span id="page-4-4"></span>First we consider the model in the spherical limit. Namely, we assume the hopping matrix given in Eq. [\(2\)](#page-1-4) as

$$
\hat{t}\_{ij} = \begin{pmatrix} t & 0 & 0 \\ 0 & t & 0 \\ 0 & 0 & t \end{pmatrix}, \tag{38}
$$

for a bipartite lattice with the coordination number z. Using the spin-orbital operator O ηµ i defined in the previous section, we obtain the spherical model as

$$\begin{aligned} \mathcal{H}\_{\text{eff}} &= -\sum\_{\langle ij\rangle} \left[ I\_S \mathbf{S}\_i \cdot \mathbf{S}\_j + I\_L \mathbf{L}\_i \cdot \mathbf{L}\_j + I\_Q \sum\_{\eta} Q\_i^{\eta} Q\_j^{\eta} \right] \\ &+ I\_R \sum\_{\mu} \sum\_{\nu} R\_i^{\nu,\mu} R\_j^{\nu,\mu} + I\_T \sum\_{\mu} \sum\_{\eta} T\_i^{\eta,\mu} T\_j^{\eta,\mu} + I\_0 \right], \end{aligned} \tag{39}$$

where the sum with hiji is taken over the pairs of the NN sites. The superscript µ, ν (= x, y, z) and η (= x <sup>2</sup> <sup>−</sup> <sup>y</sup> 2 , z<sup>2</sup> , xy, yz, zx) are the indices for the polynomials, which represents the component of the spin, rank 1 orbital and rank 2 orbital, respectively. We have rewritten the operators in accordance with their symmetries as

$$S\_i^{\mu} = \frac{1}{2} O\_i^{0\mu},\tag{40}$$

$$L\_i^x = \frac{1}{2} O\_i^{70},\ L\_i^y = \frac{1}{2} O\_i^{50},\ L\_i^z = \frac{1}{2} O\_i^{20},\tag{41}$$

$$\begin{aligned} Q\_i^{x^2 - y^2} &= \frac{1}{2} O\_i^{30}, \; Q\_i^{z^2} = \frac{1}{2} O\_i^{80},\\ Q\_i^{xy} &= \frac{1}{2} O\_i^{10}, \; Q\_i^{yz} = \frac{1}{2} O\_i^{60}, \; Q\_i^{zx} = \frac{1}{2} O\_i^{40}, \end{aligned} \tag{42}$$

$$R\_i^{x, \mu} = \frac{1}{2} O\_i^{7\mu}, \ R\_i^{y, \mu} = \frac{1}{2} O\_i^{5\mu}, \ R\_i^{z, \mu} = \frac{1}{2} O\_i^{2\mu}, \tag{43}$$

$$\begin{aligned} T\_i^{x^2 - y^2, \mu} &= \frac{1}{2} O\_i^{3\mu}, \ T\_i^{z^2, \mu} = \frac{1}{2} O\_i^{8\mu}, \\ T\_i^{xy, \mu} &= \frac{1}{2} O\_i^{1\mu}, \ T\_i^{yz, \mu} = \frac{1}{2} O\_i^{6\mu}, \ T\_i^{zx, \mu} = \frac{1}{2} O\_i^{4\mu}. \end{aligned} \quad (44)$$

The physical meaning of each order parameter now becomes clearer with this notation. We call S µ i a magnetic spin (MS or S), L µ i a magnetic orbital (MO or L), Q µ i a electric orbital (EO or Q), R ν,µ i a electric spin-orbital (ESO or R) and T η,µ i a magnetic spin-orbital (MSO or T) moments. I<sup>0</sup> represents energy gain by the second order perturbation process. Obviously, Eq. [\(39\)](#page-5-0) satisfies SU(2)×SO(3) symmetry in spin-orbital space.

We will show the numerical results of the n = 1 and n = 3 spherical models under MFA, both of which have the six states per site in the model space as discussed in Sec. [II A.](#page-1-5) We beforehand introduce the following notation with regard to the coupling constants defined in Eq. [\(39\)](#page-5-0) as

$$I\_{\xi} = \sum\_{n} A\_{\xi n} \frac{t^2}{\Delta E\_n},\tag{45}$$

for ξ = S, L, Q, R, T, 0, where ∆E<sup>n</sup> represents all possible excitation energies. Its energy corresponds to the denominator of Eq. [\(6\)](#page-1-3). The coefficient A is summarized in the tables in the following subsections (see Sec. [III B](#page-5-1) or Sec. [III C\)](#page-8-0).

Before we show the mean-field results, we discuss the ground state wave function for the two-site problem. Using the single site state defined in Eq. [\(4\)](#page-1-2) or [\(18\)](#page-2-0), we

TABLE I. Coefficients A defined in Eq. [\(45\)](#page-5-2) for n = 1 spherical model. The ground state energy is zero. We add the details for the intermediate state in the main text.

<span id="page-5-3"></span>

| ∆En | U − 3J | U − J | U + 2J |
|-----|--------|-------|--------|
| IS  | −2     | 10/3  | 2/3    |
| IL  | 3      | −5/3  | 2/3    |
| IQ  | 3      | −1/3  | −2/3   |
| IR  | 1      | 5/3   | −2/3   |
| IT  | 1      | 1/3   | 2/3    |
| I0  | −6     | −10/3 | −2/3   |

<span id="page-5-0"></span>obtain the two-site (i.e., sites at i and j) ground state as

$$\langle \text{gs} \rangle = \sum\_{\gamma\_i, \sigma\_i} \sum\_{\gamma\_j, \sigma\_j} C\_{\gamma\_i \sigma\_i, \gamma\_j \sigma\_j} \left| \gamma\_i, \sigma\_i \right\rangle\_i \left| \gamma\_j, \sigma\_j \right\rangle\_j \,. \tag{46}$$

<span id="page-5-4"></span>The explicit form of the matrix C is written as

$$
\hat{C} = \lambda^0 \otimes (-\mathbf{i}\sigma^y). \tag{47}
$$

This shows that the ground-state wave function is spinsinglet and symmetric on the orbital. This is valid for all the spherical cases considered in this section. For an infinite lattice, as in the single-orbital Hubbard model, the inter-site spin-singlet state may favor the antiferromagnetic state in the ground state for a bipartite lattice.

# <span id="page-5-1"></span>B. n = 1 model

<span id="page-5-5"></span>First of all, we consider the results for the n = 1 model. Although the results are not relevant to the alkalidoped fullerides, the knowledge is useful in interpreting the more complicated model for the spherical n = 3 model (Sec. [III C\)](#page-8-0), the realistic A15- (Sec. [IV A\)](#page-10-1) and fccstructure fullerides (Sec. [IV B\)](#page-12-0).

# 1. Coupling constant

<span id="page-5-2"></span>We begin with the analysis of the intermediate states relevant to the second-order perturbation theory. We show the coefficients A defined in Eq. [\(45\)](#page-5-2) in Table [I.](#page-5-3) We have the three kinds of excited states, whose energy is determined by the local Coulomb interaction. For ∆E<sup>n</sup> = U − 3J, the intermediate states are nine-fold degenerate spin-triplet states, as expressed, e.g., by c † i,y,↑ c † i,x,↑ |0i and <sup>√</sup> 1 2 c † i,y,↓ c † i,x,<sup>↑</sup> + c † i,y,↑ c † i,x,↓ |0i. For ∆E<sup>n</sup> = U − J, the intermediate states are the inter-orbital spin-singlet states such as <sup>√</sup> 1 2 c † i,y,↓ c † i,x,<sup>↑</sup> − c † i,y,↑ c † i,x,↓ |0i, and the intra-orbital spin-singlet states with anti-bonding orbitals written as √ 2 3 2b † i,z − b † i,x − b † i,y |0i. These two kinds of states take the same energy since there is the spherically symmetric condition U <sup>0</sup> <sup>=</sup> <sup>U</sup> <sup>−</sup> <sup>2</sup>J. For ∆E<sup>n</sup> = U + 2J, there is only one intermediate state,

![](_page_6_Figure_1.jpeg)

**Caption:** Figure 2: This figure illustrates the dependence of coupling constants on the ratio of Hund's coupling (J/U) for the n = 1 spherical model. The vertical axis is normalized by E⁰ = t²/U. It shows that for J > 0, the system prefers orbital orders (inter-orbital spin triplet), while for J < 0, magnetic orders (intra-orbital spin singlet) are preferred. This reflects the influence of Hund's coupling on the symmetry and energy hierarchy in the system, highlighting the differential favorability of antiferromagnetic vs. ferromagnetic configurations, critical for understanding spin-orbit coupling in fullerides .


<span id="page-6-0"></span>FIG. 2. Hund's coupling ratio J/U dependence of the coupling constants for n = 1 spherical model. The vertical axis is normalized by E<sup>0</sup> = t 2 /U.

which is intra-orbital spin singlet and bonding state written as <sup>√</sup> 1 3 b † i,x + b † i,y + b † i,z |0i.

We show the Hund's coupling dependence of the coupling constants in Fig. [2.](#page-6-0) The perturbation theory is justified for −1/2 < J/U < 1/3 where the ground states are written in the form of Eq. [\(18\)](#page-2-0). Taking J = 0, the coupling constants become identical. This reflects that the system has SU(6) symmetry and the degrees of freedom of the spin and orbital are equivalent in the absence of Hund's coupling. The largest coupling constant is I<sup>S</sup> for the antiferromagnetic case (J < 0) and I<sup>Q</sup> for the ferromagnetic Hund's coupling (J > 0). This shows that the system tends to be antiferromagnetic (AFM) or antiferroorbital (AFO) order depending on the sign of the Hund's coupling. This is understood from the intermediate state.

In the case of J > 0, which is relevant to the usual t2g-orbital d-electron systems with n = 1 per atom, the energetically favorable intermediate two-electron state is inter-orbital spin triplet. To realize this intermediate state, the initial state needs to occupy parallel spin configuration with different orbitals such as c † i,x,↑ c † j,y,↑ |0i. Therefore, the orbital order should be dominant for J > 0 as a leading-order ordering instability. If we take J/U & 0.2, I<sup>S</sup> takes a ferromagnetic coupling constant, which favors parallel spins at two sites.

As for J < 0, on the other hand, the intermediate state tends to be intra-orbital spin singlet and bonding state. The corresponding initial state must be antiparallel spin with the same orbital such as c † i,x,↑ c † j,x,↓ |0i. Thus, the magnetic order should be dominant for J < 0.

![](_page_6_Figure_7.jpeg)

**Caption:** Figure 3: Represents the temperature variation of order parameters, internal energies, entropy, and specific heat for an n = 1 bipartite spherical model with J/U = -0.1, showcasing dynamics under antiferromagnetic Hund's coupling. The entropy release and internal energy trends reinforce the complex interdependencies between spin and orbital orders and signal energy state transitions at reduced temperatures .


<span id="page-6-1"></span>FIG. 3. Temperature dependence of (a) the order parameter, (b) the decomposed internal energy and total free energy density, (c) entropy and (d) specific heat for n = 1, J/U = −0.1 bipartite spherical model. The inset in (c) is enlarged plot around T<sup>c</sup>2. The energy unit of these plots are E<sup>0</sup> = t 2 /U.

# 2. Mean-field solutions for antiferromagnetic Hund's coupling (J < 0)

Let us turn our attention to the numerical results using MFA in the spherical model. We take the NN coordination number z = 6 in the numerical calculation by assuming a simple cubic lattice in three dimensions. Figure [3](#page-6-1) shows the temperature dependence of the physical quantities in the bipartite lattice model at J/U = −0.1 (antiferromagnetic Hund's coupling). We take E<sup>0</sup> ≡ t <sup>2</sup>/U as the unit of energy. The uniform and staggered order parameters are shown in Fig. [3\(](#page-6-1)a), where the antiferromagnetic spin (AF-S) order appears first with decreasing temperature from the high-temperature limit. This corresponds to the largest coupling constant I<sup>S</sup> in Fig. [2.](#page-6-0) At lower temperatures, the ferro (F)-orbital Q moment of z 2 type appears together with the AF-T (MSO) moments. In order to clarify which is the primary order parameter of the second phase transition at Tc2, we show in Fig. [3\(](#page-6-1)b) the internal energy and free energy per site, where the internal energy is decomposed into each contribution as

$$\mathcal{U}\_{\rm S} = I\_{\rm S} \langle \mathbf{S}\_{\rm A} \rangle \cdot \langle \mathbf{S}\_{\rm B} \rangle,\tag{48}$$

$$\mathcal{U}\_{L} = I\_{L} \langle \mathbf{L}\_{\rm A} \rangle \cdot \langle \mathbf{L}\_{\rm B} \rangle,\tag{49}$$

$$\mathcal{U}\_{Q} = I\_{Q} \sum\_{\eta} \langle Q\_{\mathcal{A}}^{\eta} \rangle \langle Q\_{\mathcal{B}}^{\eta} \rangle,\tag{50}$$

$$\mathcal{U}\_{\rm R} = I\_{\rm R} \sum\_{\mu} \sum\_{\nu} \langle R\_{\rm A}^{\nu,\mu} \rangle \langle R\_{\rm B}^{\nu,\mu} \rangle,\tag{51}$$

$$\mathcal{U}\_{\rm T} = I\_T \sum\_{\mu} \sum\_{\eta} \langle T\_{\rm A}^{\eta,\mu} \rangle \langle T\_{\rm B}^{\eta,\mu} \rangle. \tag{52}$$

The total internal energy is given by U = P <sup>ξ</sup> U<sup>ξ</sup> for ξ = S, L, Q, R, T, where the energy is measured from I0. We see from Fig. [3\(](#page-6-1)b) that the energy U<sup>T</sup> is gained below Tc<sup>2</sup> but U<sup>Q</sup> is not. Hence, the AF-T should be the

![](_page_7_Figure_1.jpeg)

**Caption:** Figure 4: This figure presents the temperature dependence of the inverse of uniform and staggered components of the diagonal susceptibilities for a model system with energy unit E⁰ = t²/U. Panel (a) depicts the inverse uniform susceptibility while panel (b) shows the staggered component. All susceptibilities are positive, indicating stable solutions, and the divergence of the antiferromagnetic (AF-S) susceptibility at T/E⁰ ≈ 2.3 marks the onset of antiferromagnetic order, highlighting the stability and the phase transition dynamics within the system .


<span id="page-7-0"></span>FIG. 4. Temperature dependence of the inverse of (a) uniform and (b) staggered component of the diagonal susceptibilities. The energy unit is E<sup>0</sup> = t 2 /U.

primary order parameter and F-Q is just induced by the combination of AF-S plus AF-T moments. The results are consistent with the magnitude relation I<sup>T</sup> > I<sup>Q</sup> seen in Fig. [2,](#page-6-0) where the larger energy gain is obtained from T-moment than the energy loss from Q.

Figure [3\(](#page-6-1)c) shows the temperature dependence of the entropy, where all the entropy is released in the ground state. With increasing temperature, the entropy shows a kink at T /E<sup>0</sup> ' 0.84, at which the value of the entropy is close to ln 3 meaning that the orbital degeneracy is lifted below this transition temperature. The inset of (c) shows the magnified picture of the entropy near Tc2, indicating the first-order transition. The specific heat C = ∂U/∂T is also shown in Fig. [3\(](#page-6-1)d). There are two discontinuity corresponding to the spin and orbital orders.

Next we show in Fig. [4](#page-7-0) the inverse of the diagonal susceptibilities χ ηµ;ηµ u (uniform) and χ ηµ;ηµ s (staggered) which are defined in Eqs. [\(32\)](#page-4-4) and [\(33\)](#page-4-5). First, we observe that the susceptibilities shown here are all positive, indicating a stable solution. The AF-S susceptibility of x, y, z type diverges at T /E<sup>0</sup> ' 2.3 signaling the onset of the antiferromagnetic order. Below this transition temperature, the longitudinal z component is decreased while the perpendicular x, y components remain divergent. This behavior indicates the presence of the Goldstone mode, where the excitations are induced by rotating the z component into xy-plane, as in the standard Heisenberg model. Inside this magnetic phase, the orbital (F-Q) and spin-orbital (AF-T) susceptibility, which are z 2 type in orbital part, continue to grow and tend to diverge at lower transition point (Tc2). As shown in Fig. [4\(](#page-7-0)a), the 'perpendicular' components, i.e. F-Qyz, F-Qzx, remain divergent below Tc2, indicating the presence of the Goldstone mode even for the orbital order in the spherical model. Namely, because of the symmetry of the spin-orbital space, the energetically equivalent solutions exist and are obtained by rotating the order parameters.

Next we discuss the ground state wave function, which includes the information of order parameter at zero temperature limit. As is evident from the zero entropy at T = 0, we have the non-degenerate ground state. In the present case, the ground state wave function is very

![](_page_7_Figure_7.jpeg)

**Caption:** Figure 5: This figure showcases a multi-panel visualization of temperature-dependent order parameters, internal energy, entropy, and specific heat (panels a to d). Focused on a bipartite spherical model with n = 1, J/U = 0.1, it depicts the properties' variation with temperature, evidencing phase transitions and suggesting the stability of the ground state due to non-negative Hessian eigenvalues. This reinforces the understanding of phase transition behaviors in systems with ferromagnetic Hund's coupling .


<span id="page-7-1"></span>FIG. 5. Temperature dependence of (a) the order parameter, (b) the decomposed internal energy and total free energy density, (c, left axis) the single site entropy, (c, right axis) the specific heat and (d) the eigenvalues of the Hessian matrix ˆa for bipartite spherical model with n = 1, J/U = 0.1.

simple and is given using Eq. [\(4\)](#page-1-2) by

$$\left|\psi\_{\mathbf{A}}\right\rangle = \left|n=1, z, \downarrow\right\rangle\_{\mathbf{A}}\,,\tag{53}$$

$$|\psi\_{\mathbf{B}}\rangle = |n=1, z, \uparrow\rangle\_{\mathbf{B}}\,,\tag{54}$$

for each sublattice. This corresponds to the staggered spin ordered and uniform orbital ordered state, as is consistent with Fig. [3\(](#page-6-1)a). More specifically, we can construct the order parameters from the direct product of the wave functions. In the present case, we obtain at sublattice α as

$$\left|\psi\_{\alpha}\right\rangle\left\langle\psi\_{\alpha}\right| = \mp \frac{1}{\sqrt{6}} S\_{\alpha}^{z} - \frac{1}{\sqrt{3}} Q\_{\alpha}^{z^2} \pm \frac{1}{\sqrt{3}} T\_{\alpha}^{z^2, z} + \frac{1}{6}, \quad (55)$$

where the operators are defined in Eqs. [\(40\)](#page-5-4)–[\(44\)](#page-5-5). The upper (lower) sign is chosen for α = A (α = B). The quantities that appear in the right-hand side are identical to the order parameters shown in Fig. [3\(](#page-6-1)a).

# 3. Mean-field solutions for ferromagnetic Hund's coupling (J > 0)

We show the results for the J/U = 0.1 case, where the model is now relevant to materials with d-electrons, to make contrast with behaviors of the systems with antiferromagnetic Hund's coupling. Figure [5\(](#page-7-1)a) shows the temperature evolution of the order parameters. As seen in Fig. [2,](#page-6-0) the largest coupling constant is I<sup>Q</sup> which is antiferro (I<sup>Q</sup> < 0), and therefore the AF-Q order of z 2 -type appears at the highest transition temperature (Tc1). The F-Q order of the same z 2 -type is simultaneously induced. The rise of the order parameters near the transition temperature behaves as ∼ √ Tc<sup>1</sup> − T for AF-Q and ∼ Tc1−T for F-Q. Hence the AF-Q is the primary order. From the symmetry argument, it can be shown that the F-Q order
arises from AF-Q order since the coupling term in the Landau free energy has the form Q<sup>z</sup> 2 u (Q<sup>z</sup> 2 s ) 2 . The existence of such third-order term can be understood if one considers the symmetry in the plane of Q<sup>z</sup> 2 -Q<sup>x</sup> <sup>2</sup>−y 2 [\[26\]](#page-14-0). At lower temperatures, the magnetic F-S order appears, where T-moments of T z 2 ,z-type are also finite. From the internal-energy analysis shown in Fig. [5\(](#page-7-0)b), the relevant ordering at Tc<sup>2</sup> is induced from the interaction I<sup>T</sup> while I<sup>S</sup> is energetically unfavorable. Thus, comparing with the J/U = −0.1 case, the roles of magnetic order and electric (orbital) order are switched. This switching of the magnetic and orbital ordered states depending on the sign of J has also been reported in the two orbital model [\[25\]](#page-14-1).

We next show the temperature dependence of the entropy and specific heat in Fig. [5\(](#page-7-0)c), where we have defined the sublattice-dependent entropy (Shannon entropy) by

$$\mathcal{S}\_{\alpha} = -\sum\_{n} p\_n^{\alpha} \ln p\_n^{\alpha},\tag{56}$$

where p α n is the probability for the n-th state as calculated from the local partition function Z<sup>α</sup> = P n exp(−βE<sup>α</sup> <sup>n</sup> <sup>P</sup> ) = n p α <sup>n</sup>Zα. Since the entropy at zero temperature is zero at A sublattice and is finite at B sublattice, the two sublattices are inequivalent and are not simply connected by symmetry operations. This is due to the presence of the both uniform and staggered orbital order parameters in Fig. [5\(](#page-7-0)a). Indeed, the wave function in the ground state is written for each sublattice as

$$|\psi\_{\mathbf{A}}\rangle = |n=1, z, \downarrow\rangle\_{\mathbf{A}}\,,\tag{57}$$

$$\left| \vec{\psi\_{\mathcal{B}}} \right\rangle = \begin{pmatrix} |n=1, x, \downarrow\rangle\_{\mathcal{B}} \\ |n=1, y, \downarrow\rangle\_{\mathcal{B}} \end{pmatrix}. \tag{58}$$

The remaining degeneracy at B sublattice is because the x- and y-orbital components are equivalent. Namely, the triply degenerate state at each sublattice splits depending on the sublattice: z orbital becomes energetically higher at A sublattice and lower at B sublattice. Thus the antiferro order of this type cannot lift the degeneracy completely.

Usually, the degeneracy is lifted by the interaction effects and the unique ground state is expected. Then, one may suspect that the remaining degeneracy might indicate the instability of the solutions. In order to show that our degenerate ground states are really stable, we show the energy spectra of the Hessian matrix discussed in Sec. [II C.](#page-3-0) As shown in Fig. [5\(](#page-7-0)d), the excitation energies in terms of Landau theory are all positive or zero, and the system is thus stable. The degeneracy at T = 0 is due to the absence of the relevant interactions, and will be resolved once the other types of the interaction are included in the more realistic situations.

We comment on the case where we allow only for the uniform solutions, by having the geometrical frustration effect in mind which does not favor a simple staggered orders. Actually, the n = 1 uniform spherical model around

<span id="page-8-0"></span>TABLE II. Coefficients A in Eq. [\(45\)](#page-5-0) for n = 3 spherical model. The ground state is written as |γi, σii i |γ<sup>j</sup> , σ<sup>j</sup> i j and its energy is 2(3U − 4J). We add the details for the intermediate state in the main text.

| ∆En |      |      | U − 8J U − 6J U − 4J U − 3J U − J U + 2J |      |       |      |
|-----|------|------|------------------------------------------|------|-------|------|
| IS  | 1/2  | −5/3 | 25/18                                    | −4/3 | 20/9  | 8/9  |
| IL  | 9/8  | −5/4 | 25/72                                    | 2    | −10/9 | 8/9  |
| IQ  | −9/8 | 1/4  | −1/72                                    | 2    | −2/9  | −8/9 |
| IR  | −1/8 |      | −5/12 −25/72                             | 2/3  | 10/9  | −8/9 |
| IT  | 1/8  | 1/12 | 1/72                                     | 2/3  | 2/9   | 8/9  |
| I0  | −9/2 | −5   | −25/18                                   | −4   | −20/9 | −8/9 |

J = 0 has no solution at any temperature because all of the coupling constants are negative (antiferromagnetic) in the spherical model (see Fig. [2\)](#page-6-0). On the other hand, for relatively large |J| region the uniform solutions can exist. However, since the typical value of Hund's coupling is |J|/U ∼ 0.1 or less, we do not enter the regime with larger |J| in this paper.

### C. n = 3 model

Here we consider the model with three electrons per molecule and with the antiferromagnetic Hund's coupling (J < 0). This model is more relevant to the existing fullerides with half-filled t1<sup>u</sup> molecular orbitals.

### 1. Coupling constants

We show the coefficients A, which is defined by Eq. [\(45\)](#page-5-0), in Table [II.](#page-8-0) Since we consider the half-filled model, the initial and intermediate states for the twosite problem at the sites i and j relevant to Iij are (n<sup>i</sup> , n<sup>j</sup> ) = (3, 3) and (n<sup>i</sup> , n<sup>j</sup> ) = (2, 4), respectively. Here, n<sup>i</sup> = 2 and n<sup>i</sup> = 4 states are connected with each other by the particle-hole (PH) transformation. The explicit form for n<sup>i</sup> = 2 state is same as those given in Sec. [III B,](#page-5-1) and thereby the n = 4 can also be constructed from n = 2 accordingly. Below, we list the types of the intermediate states and their energies, specifically focusing on the n<sup>j</sup> = 4 state.

The intermediate states with the excited energy∆E<sup>n</sup> = U − 8J are nine kinds of inter-orbital spin triplet state for n<sup>i</sup> = 2 and the PH transformed states for n<sup>j</sup> = 4 such as b † j,zc † j,y,↑ c † j,x,↑ |0i. For ∆E<sup>n</sup> = U − 6J, the intermediate states are the inter-orbital spin triplet states for n<sup>i</sup> = 2 and the PH transformed states which have inter-orbital spin singlet states such as <sup>√</sup> 1 2 b † j,z c † j,y,↓ c † j,x,<sup>↑</sup> − c † j,y,↑ c † j,x,↓ |0i or intra-orbital spin singlet with anti-bonding such as √ 2 3 2b † j,zb † j,y − b † j,zb † j,x − b † j,yb † j,x |0i. For ∆E<sup>n</sup> = U −4J, the intermediate states are the inter-orbital spin singlet or intra-orbital spin singlet with anti-bonding states for

![](0__page_9_Figure_0.jpeg)

**Caption:** Figure 6: This figure illustrates the dependence of coupling constants on the Hund's coupling ratio J/U for the n = 3 spherical model. The perturbation theory is valid for -0.5 < J/U < 0, where no level crossing occurs. It is noted that for J > 0, the ground state converges to a total spin S = 3/2 configuration, contrasting with J < 0 configurations. This demonstrates the influence of Hund's coupling on the spin state hierarchy and the coupling constants' role in stabilizing different spin-orbit coupling behaviors as a function of J/U .


<span id="page-9-0"></span>FIG. 6. Hund's coupling ratio J/U dependence of the coupling constants for n = 3 spherical model.

n<sup>i</sup> = 2, and their PH transformed versions for the j site. For ∆E<sup>n</sup> = U −3J, the intermediate states are the intraorbital spin singlet and bonding states for n<sup>i</sup> = 2, and the states which have inter-orbital spin triplet for n<sup>j</sup> = 4. For ∆E<sup>n</sup> = U − J, the intermediate states are the interorbital spin singlet or intra-orbital spin singlet with antibonding states (n<sup>i</sup> = 2), and intra-orbital spin singlet and bonding state such as <sup>√</sup> 1 3 b † j,zb † j,y + b † j,zb † j,x + b † j,yb † j,x |0i for n<sup>j</sup> = 4. Finally, for ∆E<sup>n</sup> = U + 2J, which is the lowest among the excited states for J < 0, the intermediate state is non-degenerate and is written as the intra-orbital spin singlet with bonding state for n<sup>i</sup> = 2 and its PH transformed states for n<sup>j</sup> = 4.

Figure [6](#page-9-0) shows the Hund's coupling dependence of the coupling constants. The perturbation theory is justified for −1/2 < J/U < 0 where any level cross for the unperturbed Hamiltonian does not occur. If we consider J > 0, the ground state is a total spin S = 3/2 state (e.g., c † i,z,↑ c † i,y,↑ c † i,x,↑ |0ii) and is different from J < 0. This point is in contrast with n = 1 case where the ground state of the local Hamiltonian is not dependent on the sign of J as shown in Fig. [2.](#page-6-0) It is notable that the coupling constants for n = 3 case are similar to those of the n = 1 spherical model in the region near J/U = −0.5, where the same physical behavior is expected.

### 2. Mean-field solutions for bipartite lattice

We show in Fig. [7\(](#page-10-0)a) the order parameters for the bipartite lattice model with n = 3 and J/U = −0.1. At Tc<sup>1</sup> ' 2.3E0, the system shows the antiferromagnetic order, which is consistent with the largest coupling constant shown in Fig. [6.](#page-9-0) With decreasing temperature, the second order at Tc<sup>2</sup> appears, where the F-Q<sup>z</sup> 2 and AF-T z 2 ,z order parameters are additionally induced. We emphasize that this orbital order is not of the ordinary

orbital moment of electrons, but of the doublons relevant to the antiferromagnetic Hund's coupling as discussed in Sec. [II A.](#page-1-0)

Figure [7\(](#page-10-0)b) shows the temperature dependences of the internal energies and free energy. We show the orderparameter-resolved energies and all the components decrease upon entering the ordered phase. While this is in contrast to n = 1 cases shown in the previous subsections, the largest energy gain arises from the AF-T order.

We show in Fig. [7\(](#page-10-0)c) the entropy and specific heat. The clear jump in the specific heat at Tc<sup>1</sup> indicates the secondorder phase transition, and the jump in the entropy at Tc<sup>2</sup> is the fingerprint of the first-order phase transition. The wave function in the ground state is

$$|\psi\_{\mathbf{A}}\rangle = |z,\downarrow\rangle\_{\mathbf{A}}\,,\tag{59}$$

$$|\psi\_{\mathbf{B}}\rangle = |z,\uparrow\rangle\_{\mathbf{B}}\,. \tag{60}$$

The ground state is thus non-degenerate as is consistent with the zero entropy at T = 0.

### <span id="page-9-1"></span>3. Single-sublattice solution

Having the geometrically frustrated lattice in mind, we assume that the spatially modulated solutions are not realized. Then we seek for the spatially uniform solutions (single-sublattice) only.

Figure [8\(](#page-10-1)a) shows the order parameter for the singlesublattice model with n = 3, J/U = −0.1. The system shows the Q<sup>z</sup> 2 order at Tc/E<sup>0</sup> ' 0.28, which is consistent with the magnitude of the coupling constant shown in Fig. [6.](#page-9-0) The entropy and specific heat are shown in Fig. [8\(](#page-10-1)b) with left and right axis, respectively. The residual entropy S = ln 2 remains, which is in accordance with the degeneracy of spin in the absence of the sublattice degrees of freedom. Namely, the wave function of the ground state is degenerated and is written as

$$\left| \vec{\psi} \right\rangle = \begin{pmatrix} |z, \uparrow\rangle \\ |z, \downarrow\rangle \end{pmatrix}. \tag{61}$$

We have confirmed that the eigenvalues of ˆa in Eq. [\(36\)](#page-4-0) are all non-negative (not shown) and thus the ordered state is stable.

We also point out the other interesting possibilities. The above orbital order is induced by the coupling constant I<sup>Q</sup> > 0 in Fig. [6.](#page-9-0) In this figure, it is notable that the values of I<sup>Q</sup> and I<sup>R</sup> are very close with each other. Then we try to search for another solutions by introducing the modified coupling constants defined as

$$
\tilde{I}\_Q = (1+r)I\_Q,\tag{62}
$$

$$
\tilde{I}\_R = (1 - r)I\_R,\tag{63}
$$

where the original spherical model corresponds to r = 0.

![](0__page_10_Figure_1.jpeg)

**Caption:** Figure 7: This diagram analyzes the temperature evolution of order parameters, internal energies, entropy, and specific heat for an n = 3 bipartite spherical model with J/U = -0.1. Noteworthy is the phase transition marked by sharp jumps in specific heat and entropy, corresponding to changes in order parameters. The results reveal the hierarchy of spin and orbital orders and suggest an entropic gain linked to AF-T order at lower temperatures .


<span id="page-10-0"></span>FIG. 7. Temperature dependence of (a) the order parameter, (b) the decomposed internal energy and total free energy density, (c, left axis) entropy and (c, right axis) the specific heat for n = 3 bipartite spherical model with J/U = −0.1. The horizontal axis are normalized by E<sup>0</sup> = t 2 /U.

![](0__page_10_Figure_3.jpeg)

**Caption:** Figure 8: Presents temperature dependence for the n = 3 uniform spherical model with J/U = -0.1. The order parameter evolution, entropic characteristics, and specific heat analysis demonstrate the role of coupling constant ratios in determining the system's phase behavior, highlighting the competitive nature of orbital and spin orders and the stability of solutions due to the symmetry in the order-parameter space .


<span id="page-10-1"></span>FIG. 8. Temperature dependence of (a) the order parameter, (b, left axis) the entropy and (b, right axis) the specific heat for n = 3 uniform spherical model with J/U = −0.1. (c) Similar order-parameter plots for the n = 3 single-sublattice model with coupling constant ratio r = −0.4. The energy unit is E<sup>0</sup> = t 2 /U.

We show the order parameters for n = 3, J/U = −0.1 uniform model with the coupling constant ratio r = −0.4 in Fig. [8\(](#page-10-1)c). Since the magnitude of the modified coupling constants satisfies ˜I<sup>R</sup> > ˜I<sup>Q</sup> in the present condition, we obtain the solution for Rµ,µ moments. Recalling the definition of the R moment, we may rewrite the order parameter as <sup>R</sup>µ,µ <sup>∼</sup> <sup>L</sup> <sup>µ</sup>S <sup>µ</sup> symbolically. Therefore, it is interpreted that the system has the effective spin-orbit coupling spontaneously. The wave function is written as

$$\left| \vec{\psi} \right> = \frac{1}{\sqrt{3}} \begin{pmatrix} |x,\uparrow\rangle - \mathrm{i} \, |y,\uparrow\rangle - |z,\downarrow\rangle \\ - |x,\downarrow\rangle - \mathrm{i} \, |y,\downarrow\rangle - |z,\uparrow\rangle \end{pmatrix},\tag{64}$$

which indicates that the ground state is entangled with respect to spin and orbital. These doubly degenerate ground states are connected with each other by the timereversal symmetry.

This "spontaneous spin-orbit coupling" splits the sixfold degeneracy into two-fold and four-fold multiplets, and which is realized in the ground state is dependent on the sign of the order parameters. Our solutions show that the ground state is always doubly degenerate, and this should be related to the minimization of the entropy at low temperatures.

Thus, although the system at the original parameter shows the doublon-orbital ordering (Q), the system is located near the parameter range where the intriguing R order occurs. As discussed in Sec. [II A](#page-1-0) the original spinorbit coupling ΛSO is tiny, but it might enter through the R-type ordering. Such situation is realized only for n = 3 model with the antiferromagnetic Hund's coupling.

![](0__page_10_Figure_10.jpeg)

**Caption:** Figure 9: This figure shows (a) the temperature dependence of the order parameter and (b) the eigenvalues of the Hessian matrix for the A15 fulleride model in the strong coupling regime. Panel (a) illustrates the appearance of antiferromagnetic moment at Tc1 around 80 K, followed by consecutive transitions with orbital Q and spin-orbital T symmetry at Tc2 and Tc3. Panel (b) shows eigenvalues that remain non-negative, affirming the stability of the solution; failure to reach the stable ground state without ordering at Tc3 is represented by negative eigenvalues associated with open red symbols. The figure is crucial for understanding phase behaviors in fullerides and their orbital and spin ordering processes .


<span id="page-10-2"></span>FIG. 9. Temperature dependence of (a) the order parameter and (b) the eigenvalues of the matrix ˆa for A15 fulleride model. The blue filled symbol in (b) corresponds to the solution given in (a). The red circle represents the solution without the phase transition at T<sup>c</sup>3.

### IV. NUMERICAL RESULTS FOR FULLERIDES

We show the numerical results for the fulleride in the strong coupling regime by using the hopping parameters obtained by the first principles calculation [\[40\]](#page-14-2). We take the intra-orbital Coulomb interaction U = 1eV and the Hund's coupling J/U = −0.1 in the following.

# A. A15 structure

First of all we show in Fig. [9\(](#page-10-2)a) the temperature dependence of order parameters for the strong-coupling limit model of the realistic fulleride material with the A15 structure. The hopping parameters for Cs3C<sup>60</sup> is chosen (A15-Cs(V opt−P SC ) in Ref. [\[40\]](#page-14-2)). The lattice structure is a bipartite lattice, and A and B sublattices are connected with each other by screw transformation (i.e., translation plus four-fold rotation). As shown in the figure, at Tc<sup>1</sup> ' 80K, the antiferromagnetic moment (AF-S) appears by the second-order phase transition. At lower temperatures, we identify the two successive phase transitions (Tc2,3) with orbital moment Q and spin-orbital moment T. These two Q, T moments share the same symmetry under the presence of AF-S <sup>z</sup> order. We cannot simply conclude which one is the primary order parameter, because the interaction has complicated form for the realistic model and cannot be decomposed to each contribution as in the spherical model. We also note that our choice of parameter is not fine-tuned to reproduce correctly the transition temperature in the actual materials, although our results can be compared with the experiments semi-quantitatively.

We show in Fig. [9\(](#page-10-2)b) the eigenvalues (filled blue symbols) of the Hessian matrix defined in Eq. [\(36\)](#page-4-0). All the values are non-negative, and therefore the system is stable. On the other hand, we can also calculate the low-temperature solutions by suppressing the ordering at Tc3. The results are plotted as the open red symbols in Fig. [9\(](#page-10-2)b). In this case, the eigenvalues become partially negative and hence the system is not stable although the entropy goes to zero even in this case. Thus, the emergence of the order at Tc<sup>3</sup> is essential in order to reach the stable ground state.

We discuss the origin of the second orbital order at Tc<sup>3</sup> in more detail. Below, we concentrate on the properties of Q moments to make the discussion simple, since the symmetry of Q is same as that of T below the transition temperature Tc1. Figure [10\(](#page-11-0)a) shows the orbital order parameters for sublattice A (left panel) and B (right panel) slightly below the transition temperature Tc<sup>2</sup> (but above Tc3). The three patterns are obtained depending on the initial condition and hence are degenerate solutions. It is seen from Fig. [10\(](#page-11-0)a) that the plane of X<sup>α</sup> = Q<sup>z</sup> 2 <sup>α</sup> and Y<sup>α</sup> = Q<sup>x</sup> <sup>2</sup>−y 2 <sup>α</sup> has a three-fold rotational symmetry and the equilateral triangle points, where the free energy minima are located, are tilted from the X axis. This tilt angle remains finite at low temperatures below Tc3.

This result can be understood from the Landau theory: we can show that, without four-fold rotational symmetry as in T<sup>h</sup> point group symmetry in fulleride materials, the Landau free energy is written in the restricted orderparameter space as

$$\mathcal{F}\_{\mathcal{L}} = \sum\_{\alpha=\mathcal{A},\mathcal{B}} \left[ c\_1 X\_{\alpha} (X\_{\alpha}^2 - 3Y\_{\alpha}^2) + c\_2 s\_{\alpha} Y\_{\alpha} (3X\_{\alpha}^2 - Y\_{\alpha}^2) \right],\tag{65}$$

where sα=A = +1 and sα=B = −1. We have considered only the third-order term for our purpose. This is consistent with the numerical results and the tilt of the angle is due to the presence of c<sup>2</sup> term. The tilt angle is estimated with the polar coordinates X = r cos θ and

![](0__page_11_Figure_7.jpeg)

**Caption:** Figure 10: This figure shows sublattice-dependent order parameters on the Qᶻ² and Qˣ²⁻ʸ² plane for different sublattices at specific thermal conditions. It highlights the order parameter patterns induced by rotational symmetries at T < Tc² and zero temperature, elucidating the role of symmetry breaking in order parameter modulation and systemic stability across temperature gradients .


<span id="page-11-0"></span>FIG. 10. (a) Sublattice-dependent order parameters in the plane of Q z 2 -Q x <sup>2</sup>−y 2 for A (left) an B (right) sublattices at T = 40.4K (< T<sup>c</sup>2). The similar plots at low-temperature limit without the transition at T<sup>c</sup><sup>3</sup> is shown in (b). The dashed circles in (a,b) correspond to the solutions in the system with four-fold symmetries. Each color shows different kind of solutions, which share the same free energy. The gray arrows with Q x 2 or Q y 2 are the guide for taking the other quantization axis. Specifically, the solution given in Fig. [9\(](#page-10-2)a) corresponds to the blue circle in the present figure (a). The angle φ in the left panel of (a) is the deviation from the horizontal axis.

Y = r sin θ, leading to another expression of the free energy <sup>F</sup><sup>L</sup> <sup>∝</sup> cos(3<sup>θ</sup> <sup>+</sup> <sup>φ</sup>) with <sup>φ</sup> = tan<sup>−</sup><sup>1</sup> c2/c<sup>1</sup> being the tilt angle. For example, one can estimate this angle from Fig. [10\(](#page-11-0)a) as φ = 6.76◦ . The A15 structure has the screw symmetry, i.e., the combination of the translation along [111] and four-fold rotation around x, y, z axes, which relates the order parameters at A and B sublattices. Indeed, the above Landau free energy is invariant under the three-fold rotation and screw transformations.

If the four-fold symmetry is present, the condition c<sup>2</sup> = 0 or φ = 0 is required. In Fig. [10\(](#page-11-0)b), we show the order parameters at T → 0 without the second orbital ordering below Tc3, where the four-fold symmetry seems to be effectively recovered since the tilt angle goes to zero when T → 0. Hence, the origin of the second orbital order in Fig. [9\(](#page-10-2)a) below Tc<sup>3</sup> is interpreted as induced from this emergent symmetry at low temperatures which provides an additional free energy gain.

![](0__page_12_Figure_1.jpeg)

**Caption:** Figure 11: Examining the temperature evolution of order parameters and inverse diagonal susceptibility for an fcc fulleride model with J/U = -0.1, it exposes the hierarchy of spin and orbital orders emerging as temperature declines. Timereversal symmetry breaking is indicated by Lᶻ moments arising simultaneously with Qᶻ² orders, aligning with theoretical expectations of coupled Landau free energy terms .


<span id="page-12-0"></span>FIG. 11. Temperature dependence of (a) the order parameter and (b) the inverse of the diagonal susceptibility for uniform fcc fulleride model with J/U = −0.1.

# B. fcc structure

Finally, we consider the fulleride material with the fcc structure. The spin-orbital model in the strong-coupling limit is obtained by using the hopping parameters for Rb3C<sup>60</sup> in Ref. [\[40\]](#page-14-2). Because of the geometrically frustrated nature of the fcc lattice, we here seek for only the spatially uniform ordered states.

Figure [11\(](#page-12-0)a) shows the temperature evolution of the order parameters. Here the primary order parameter is the uniform L <sup>z</sup> moment which breaks the time-reversal symmetry. The L z -order arises as (T<sup>c</sup> − T) 1/2 , and the Q<sup>z</sup> 2 is also induced simultaneously with the linear temperature dependence ∝ (Tc−T). The latter Q moment is induced from the coupling term with the form (L z ) <sup>2</sup>Q<sup>z</sup> 2 in the Landau free energy. We note that L<sup>z</sup> is not induced when Q<sup>z</sup> 2 is a primary order parameter from that coupling, since L<sup>z</sup> and Q<sup>z</sup> 2 have different time-reversal symmetry [(L z ) <sup>2</sup> and Q<sup>z</sup> 2 are same]. The ground-state wave function is written in a simple form as

$$\left| \vec{\psi} \right> = \frac{1}{\sqrt{2}} \begin{pmatrix} |x,\uparrow\rangle - \mathrm{i} \, |y,\uparrow\rangle \\ |x,\downarrow\rangle - \mathrm{i} \, |y,\downarrow\rangle \end{pmatrix},\tag{66}$$

where the complex wave function clearly shows the timereversal symmetry breaking. We note that this orbital moment is not a simple orbital motion around the fullerene molecule, but a complex motion of the three electron state given in Eq. [\(4\)](#page-1-1). In our calculations the spin S order does not occur and the ground state is doubly degenerate at each cite. The stability of the solution is checked by the non-negative eigenvalues of the Hessian matrix.

The results found here is different from the spherical case discussed in Sec. [III C 3.](#page-9-1) The difference is due to the specific form of the tight-binding hopping parameters. We show one of the coupling constant I<sup>L</sup> for the nearest neighbour sites as

$$\begin{pmatrix} I\_1 & I\_2 & 0 \\ I\_2 & -I\_3 & 0 \\ 0 & 0 & I\_4 \end{pmatrix}, \quad \mathcal{R} = \left(\frac{a}{2}, \frac{a}{2}, 0\right), \tag{67}$$

where R is the direction of the NN molecules and a is the lattice constant for the fcc fulleride. The information for the other NN pairs is constructed from the symmetry operations. The values of the matrix element are I<sup>1</sup> = 12.5, I<sup>2</sup> = 9.77, I<sup>3</sup> = 0.511 and I<sup>4</sup> = 20.1 in units of K in the present models. The coupling constant has the same symmetry as the hopping parameters in Ref. [\[40\]](#page-14-2) as required by the space group symmetry. The nearest neighbor coupling constant is largest and is positive, which favors the uniform magnetic orbital moment L. As for the next nearest neighbour site, the coupling constant matrices are diagonal and every component of them is smaller than nearest neighbour ones.

Since the spin S moment has the same symmetry as L, it can in general be simultaneously induced under the small but finite spin-orbit coupling. However, as discussed in Sec. [II A,](#page-1-0) the magnitude of the effective spinorbit coupling for the doublon orbital is ΛSO <sup>∼</sup> <sup>10</sup><sup>−</sup><sup>9</sup> eV, which can be regarded as zero in practice. Hence, the spin order can occur independently at low temperatures. The absence of the spin S order is interpreted from the point of view of the coupling constant. Figure [11\(](#page-12-0)b) shows that the temperature dependence of the inverse of the diagonal susceptibilities. The blue lines represents the magnetic susceptibility (S), which indicates that the coupling constants of S are antiferromagnetic owing to the negative Curie-Weiss temperature. In this case, the transition temperature should be very low due to the geometrical frustration of fcc lattice, but finally the system should show some magnetic ordering [\[41\]](#page-14-3).

### C. Discussion

The models in this section are based on the bandstructure calculation results. Furthermore, the fulleride materials can be located in the Mott insulator regime depending on the pressure. Hence, our results are potentially applied to the real materials. In fulleride materials, the antiferromagnetic orders is experimentally identified at low temperatures, while the orbital orders are not yet reported. Based on our results, we propose that at low temperatures the orbital ordered moments Q are induced with two successive transitions for A15 structures, and L moments may appear for fcc structures. Such fingerprints of the orbital orders may be found in thermodynamic quantities in principle. Here the orbital moment is not for a usual electron but for the doublons specific to the systems with antiferromagnetic Hund's coupling as emphasized in the present paper. On the other hand, since the real compounds are polycrystals and the disorder effects are also present, the orbital orders might be

smeared out in realistic situations. In this context, the effect of disorders on our spin-orbital model is interesting future issues which make it more direct to compare the theoretical results with experimental observations. Moreover, the antiferromagnetic Hund's coupling originates from the electron-phonon coupling. The resultant retardation effects are also the parts not included in this paper and an important issue for the more realistic arguments.

# V. SUMMARY AND OUTLOOK

In order to clarify the properties of strongly correlated electrons in fulleride superconductors, we have constructed the spin-orbital model in the strong coupling limit. We begin with the three-orbital Hubbard model with the antiferromagnetic Hund's coupling which is realized by the coupling between the electronic degrees of freedom and anisotropic Jahn-Teller molecular vibrations. In this case, the pair hopping effect among the different orbitals becomes relevant in strong contrast to the multiorbital d-electron systems with the ferromagnetic Hund's coupling. We have mainly considered the half-filled n = 3 case relevant to real materials, where it is composed of the singly-occupied (singlon) plus doubly occupied orbitals (doublon) as illustrated in Fig. [1.](#page-1-2) The correlated ground state for an isolated fullerene molecule is six-fold degenerate and is characterized by the spin and orbital indices. This is the situation similar to the n = 1 ground states and the analogy between n = 3 and n = 1 helps us for interpreting the results. The usual orbital moment, which is present for the n = 1 case, is absent for n = 3 because of the correlated nature of the wave function, and instead the active orbital moment characteristic for doublons exists. As the result, the spin-orbit coupling, which is the order of 1meV for pelectrons, becomes 1neV because of the extended nature of the molecular orbitals and the correlation effects.

We have applied the second-order perturbation theory with respect to the inter-molecule hopping, and have obtained the localized spin-orbital model specific to the fullerides. The obtained spin-orbital model is analyzed by employing the mean-field approximation. For reference, we have first solved the spherical n = 1 model for both ferromagnetic and antiferromagnetic Hund's couplings with a spherical limit for the bipartite lattice. We then apply our method to the n = 3 model where the magnetic order is found at relatively high temperatures and the orbital order also occurs at lower temperatures.

The temperature dependences of the physical quantities such as order parameters, internal and free energies, specific heat, entropy, and susceptibilities are investigated in detail. The thermodynamic stability is also studied based on the Hessian matrix derived from the inverse susceptibilities, and are checked by confirming that all the eigenvalues are non-negative.

We have also considered the realistic situation in alkalidoped fullerides, by using the tight-binding parameters derived from the first principles calculations. For the choice of the lattice structure, we have taken both the bipartite A15 and fcc structures, whose hopping parameters have been derived in Ref. [\[40\]](#page-14-2). For the A15 structure, the antiferromagnetic order occurs at high temperatures, and the electric orbital orders arise at lower temperatures with two successive transitions. The first orbital order is already captured in the spherical model, but the second orbital order is characteristic for the T<sup>h</sup> symmetry in fulleride materials where only the three-fold rotation symmetry exists. This point has been discussed in detail based on the Landau theory. For the fcc model, we have concentrated on the spatially uniform solutions due to the geometrically frustrated nature of the lattice. We have found that the magnetic orbital order occurs. Although this orbital moment has the same symmetry as the electronic spin, the spin moment is not induced simultaneously in fulleride since the spin-orbit coupling is tiny as mentioned above. Thus the spin-moment can order independently, and is expected to be antiferromagnetically ordered in the ground state where the transition temperature is expected to be low owing to the geometrical frustration of the fcc lattice.

Our formalism itself is constructed in a very general way, and can be applied to any systems in the strong coupling limit with integer fillings per atom or molecule. In this context, it would be desirable to develop the general framework for the strong-coupling-limit spin-orbital model with the combination of the hopping parameters in the Wannier functions obtained from the band-structure calculations. This application is of interest specifically in studying the ordered state of the multiorbital electronic systems including transition metals and organic materials. This point remains to be explored and is an intriguing issue in the future.

### ACKNOWLEDGEMENT

This work was supported by JSPS KAKENHI Grants No. JP18K13490 and No. JP19H01842.

- [1] A. F. Hebard, M. J. Rosseinsky, R. C. Haddon, D. W. Murphy, S. H. Glarum, T. T. M. Palstra, A. P. Ramirez, and A. R. Kortan, [Nature \(London\)](https://doi.org/10.1038/350600a0) 350, 600 (1991).
- [2] M. J. Rosseinsky, A. P. Ramirez, S. H. Glarum, D. W. Murphy, R. C. Haddon, A. F. Hebard, T. T. M. Palstra, A. R. Kortan, S. M. Zahurak, and A. V. Makhija, [Phys.](https://doi.org/10.1103/PhysRevLett.66.2830) Rev. Lett. 66[, 2830 \(1991\).](https://doi.org/10.1103/PhysRevLett.66.2830)
- [3] K. Holczer, O. Klein, S. mei Huang, R. B. Kaner, K. jian Fu, R. L. Whetten, and F. Diederich, [Science](https://doi.org/10.1126/science.252.5009.1154) 252, 1154 [\(1991\).](https://doi.org/10.1126/science.252.5009.1154)
- [4] K. Tanigaki, T. W. Ebbesen, S. Saito, J. Mizuki, J. S. Tsai, Y. Kubo, and S. Kuroshima, [Nature \(London\)](https://doi.org/10.1038/352222a0) 352, [222 \(1991\).](https://doi.org/10.1038/352222a0)
- [5] R. M. Fleming, A. P. Ramirez, M. J. Rosseinsky, D. W. Murphy, R. C. Haddon, S. M. Zahurak, and A. V. Makhija, [Nature \(London\)](https://doi.org/10.1038/352787a0) 352, 787 (1991).
- [6] A. Y. Ganin, Y. Takabayashi, Y. Z. Khimyak, S. Margadonna, A. Tamai, M. J. Rosseinsky, and K. Prassides, Nat. Mater. 7[, 367 \(2008\).](https://doi.org/10.1038/nmat2179)
- [7] Y. Takabayashi, A. Y. Ganin, P. Jegliˇc, D. Arˇcon, T. Takano, Y. Iwasa, Y. Ohishi, M. Takata, N. Takeshita, K. Prassides, and M. J. Rosseinsky, [Science](https://doi.org/10.1126/science.1169163) 323, 1585 [\(2009\).](https://doi.org/10.1126/science.1169163)
- [8] A. Y. Ganin, Y. Takabayashi, P. Jegliˇc, D. Arˇcon, A. Potoˇcnik, P. J. Baker, Y. Ohishi, M. T. McDonald, M. D. Tzirakis, A. McLennan, G. R. Darling, M. Takata, M. J. Rosseinsky, and K. Prassides, [Nature \(London\)](https://doi.org/10.1038/nature09120) 466[, 221 \(2010\).](https://doi.org/10.1038/nature09120)
- [9] O. Gunnarsson, [Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.69.575) 69, 575 (1997).
- [10] M. Fabrizio and E. Tosatti, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.55.13465) 55, 13465 [\(1997\).](https://doi.org/10.1103/PhysRevB.55.13465)
- [11] M. Capone, M. Fabrizio, C. Castellani, and E. Tosatti, Science 296[, 2364 \(2002\).](https://doi.org/10.1126/science.1071122)
- [12] M. Capone, M. Fabrizio, C. Castellani, and E. Tosatti, [Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.81.943) 81, 943 (2009).
- [13] Y. Nomura, S. Sakai, M. Capone, and R. Arita, [J. Phys.](https://doi.org/10.1088/0953-8984/28/15/153001) [Condens. Matter](https://doi.org/10.1088/0953-8984/28/15/153001) 28, 153001 (2016).
- [14] Y. Takabayashi and K. Prassides, [Phil. Trans. R. Soc. A](https://doi.org/10.1098/rsta.2015.0320) 374[, 20150320 \(2016\).](https://doi.org/10.1098/rsta.2015.0320)
- [15] R. H. Zadik, Y. Takabayashi, G. Klupp, R. H. Colman, A. Y. Ganin, A. Potoˇcnik, P. Jegliˇc, D. Arˇcon, P. Matus, K. Kamar´as, Y. Kasahara, Y. Iwasa, A. N. Fitch, Y. Ohishi, G. Garbarino, K. Kato, M. J. Rosseinsky, and K. Prassides, Sci. Adv. 1[, e1500059 \(2015\).](https://doi.org/10.1126/sciadv.1500059)
- [16] Y. Kasahara, Y. Takeuchi, R. H. Zadik, Y. Takabayashi, R. H. Colman, R. D. McDonald, M. J. Rosseinsky, K. Prassides, and Y. Iwasa, [Nat. Commun.](https://doi.org/10.1038/ncomms14467) 8, 14467 [\(2017\).](https://doi.org/10.1038/ncomms14467)
- [17] S. Han, M.-X. Guan, C.-L. Song, Y.-L. Wang, M.-Q. Ren, S. Meng, X.-C. Ma, and Q.-K. Xue, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.101.085413) 101, [85413 \(2020\).](https://doi.org/10.1103/PhysRevB.101.085413)
- [18] M.-Q. Ren, S. Han, S.-Z. Wang, J.-Q. Fan, C.-L. Song, X.-C. Ma, and Q.-K. Xue, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.124.187001) 124, 187001 [\(2020\).](https://doi.org/10.1103/PhysRevLett.124.187001)
- [19] M. Mitrano, A. Cantaluppi, D. Nicoletti, S. Kaiser, A. Perucchi, S. Lupi, P. D. Pietro, D. Pontiroli, M. Ricc`o,

S. R. Clark, D. Jaksch, and A. Cavalleri, [Nature \(Lon](https://doi.org/10.1038/nature16522)don) 530[, 461 \(2016\).](https://doi.org/10.1038/nature16522)

- [20] A. Cantaluppi, M. Buzzi, G. Jotzu, D. Nicoletti, M. Mitrano, D. Pontiroli, M. Ricc`o, A. Perucchi, P. D. Pietro, and A. Cavalleri, Nat. Phys. 14[, 837 \(2018\).](https://doi.org/10.1038/s41567-018-0134-8)
- [21] M. Capone, M. Fabrizio, P. Giannozzi, and E. Tosatti, Phys. Rev. B 62[, 7619 \(2000\).](https://doi.org/10.1103/PhysRevB.62.7619)
- [22] Y. Nomura, S. Sakai, M. Capone, and R. Arita, [Sci. Adv.](https://doi.org/10.1126/sciadv.1500568) 1[, e1500568 \(2015\).](https://doi.org/10.1126/sciadv.1500568)
- [23] A. Koga and P. Werner, Phys. Rev. B 91[, 085108 \(2015\).](https://doi.org/10.1103/PhysRevB.91.085108)
- [24] S. Hoshino and P. Werner, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.93.155161) 93, 155161 [\(2016\).](https://doi.org/10.1103/PhysRevB.93.155161)
- <span id="page-14-1"></span>[25] K. Steiner, S. Hoshino, Y. Nomura, and P. Werner, [Phys.](https://doi.org/10.1103/PhysRevB.94.075107) Rev. B 94[, 075107 \(2016\).](https://doi.org/10.1103/PhysRevB.94.075107)
- <span id="page-14-0"></span>[26] S. Hoshino and P. Werner, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.118.177002) 118, 177002 [\(2017\).](https://doi.org/10.1103/PhysRevLett.118.177002)
- [27] K. Ishigaki, J. Nasu, A. Koga, S. Hoshino, and P. Werner, Phys. Rev. B 98[, 235120 \(2018\).](https://doi.org/10.1103/PhysRevB.98.235120)
- [28] K. Ishigaki, J. Nasu, A. Koga, S. Hoshino, and P. Werner, Phys. Rev. B 99[, 085131 \(2019\).](https://doi.org/10.1103/PhysRevB.99.085131)
- [29] C. Yue, S. Hoshino, and P. Werner, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.102.195103) 102, [195103 \(2020\).](https://doi.org/10.1103/PhysRevB.102.195103)
- [30] S. Hoshino, P. Werner, and R. Arita, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.99.235133) 99, [235133 \(2019\).](https://doi.org/10.1103/PhysRevB.99.235133)
- [31] T. Misawa and M. Imada, [arXiv:1711.10205 \(2017\).](https://arxiv.org/abs/1711.10205)
- [32] K. I. Kugel and D. I. Khomskii, ZhETF Pis. Red. 15, 629 (1972), [Sov. Phys. JETP Lett. 15, 446 (1972)].
- [33] K. I. Kugel and D. I. Khomskii, Zh. Eksp. Teor. Fiz. 64, 1429 (1973), [Sov. Phys. JETP 37, 725 (1973)].
- [34] S. Inagaki, [J. Phys. Soc. Jpn.](https://doi.org/10.1143/JPSJ.39.596) 39, 596 (1975).
- [35] S. Ishihara, J. Inoue, and S. Maekawa, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.55.8280) 55, [8280 \(1997\).](https://doi.org/10.1103/PhysRevB.55.8280)
- [36] L. F. Feiner and A. M. Ole´s, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.59.3295) 59, 3295 [\(1999\).](https://doi.org/10.1103/PhysRevB.59.3295)
- [37] B. Normand and A. M. Ole´s, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.78.094427) 78, 094427 [\(2008\).](https://doi.org/10.1103/PhysRevB.78.094427)
- [38] E. Tosatti, N. Manini, and O. Gunnarsson, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.54.17184) 54[, 17184 \(1996\).](https://doi.org/10.1103/PhysRevB.54.17184)
- [39] Y. Nomura and R. Arita, Phys. Rev. B 92[, 245108 \(2015\).](https://doi.org/10.1103/PhysRevB.92.245108)
- <span id="page-14-2"></span>[40] Y. Nomura, K. Nakamura, and R. Arita, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.85.155452) 85[, 155452 \(2012\).](https://doi.org/10.1103/PhysRevB.85.155452)
- <span id="page-14-3"></span>[41] Y. Kasahara, Y. Takeuchi, T. Itou, R. H. Zadik, Y. Takabayashi, A. Y. Ganin, D. Arˇcon, M. J. Rosseinsky, K. Prassides, and Y. Iwasa, [Phys. Rev. B](https://doi.org/10.1103/PhysRevB.90.014413) 90, 014413 [\(2014\).](https://doi.org/10.1103/PhysRevB.90.014413)