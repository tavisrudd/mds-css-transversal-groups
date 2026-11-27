#!/usr/bin/env python3
"""Additive MDS conductors, label groups, and the pencil Hermitian locus.

Standard-library replay for the extension-field statements of the paper:
``prop:additive-conductor``, ``thm:additive-label-groups`` and
``prop:pencil-hermitian-gale``.

For q = p^e and F_q-linear MDS codes E, F <= F_q^n, the additive conductor
Cond_{F_p}(E, F) is the space of coordinatewise F_p-linear maps sending E into
F.  The generator computes its F_p-dimension from an explicit linear system and
compares it with the Galois-sector count e * #{tau : dim Cond(tau E, F) = 1},
where each F_q-linear conductor dimension is computed independently from a
Schur-product rank.  It also checks full support, the cross-ratio field, and
semilinearity.

For selected codes it enumerates the whole fixed-coordinate additive label
algebra, keeps the elements preserving the trace-symplectic form at every
site, and classifies them as semilinear for one field automorphism or sector
mixing.  On the Hermitian branch it evaluates the explicit map of the proof of
``thm:additive-label-groups`` into 2x2 matrices and checks that it is an
injective homomorphism onto the unitary group of the skew-Hermitian form
[[0,1],[-1,0]], enumerated directly.

At the subfield point t = 2a+1 of F_49 (zero type, cross-ratio field F_7) it
enumerates the label group GL_2(7) by solving for the Z-block forced by each
X-block, and counts its 96 semilinear elements.  The same shortcut is checked
against the full enumeration at the F_25 zero-type control.

The pencil sweep covers every parameter of the regular non-GRS locus U over
F_9, F_25, F_27, F_49, F_81, F_121, F_125, F_169.  At each point it records
the four additive conductor dimensions, the cross-ratio field, and the
Galois sectors of the code-to-dual conductor, and checks that these sectors
are exactly the automorphisms tau with D_tau(t) = 0.

Run from the supplement directory:
    python3 evidence/2026-11-26-additive-mds-conductors.py          # write
    python3 evidence/2026-11-26-additive-mds-conductors.py --check  # compare
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

OUTPUT = Path(__file__).with_suffix(".json")
SCHEMA = "mds-css-additive-conductors-v2"

# Monic moduli, coefficients from degree 0 upward.  Irreducibility is checked.
MODULI = {
    4: (2, [1, 1, 1]),
    8: (2, [1, 1, 0, 1]),
    9: (3, [2, 2, 1]),
    16: (2, [1, 1, 0, 0, 1]),
    25: (5, [2, 4, 1]),
    27: (3, [1, 2, 0, 1]),
    49: (7, [3, 6, 1]),
    81: (3, [2, 0, 0, 2, 1]),
    121: (11, [2, 7, 1]),
    125: (5, [3, 3, 0, 1]),
    169: (13, [2, 12, 1]),
}


# ------------------------------------------------------------------ fields
class Field:
    """F_{p^e} with elements 0..q-1; digit j (base p) is the coefficient of a^j."""

    def __init__(self, q: int) -> None:
        p, mod = MODULI[q]
        self.q, self.p, self.e, self.mod = q, p, len(mod) - 1, mod
        e = self.e
        self.digits = [[(x // p ** j) % p for j in range(e)] for x in range(q)]
        self.add = [[self._from([(u + v) % p for u, v in zip(self.digits[x], self.digits[y])])
                     for y in range(q)] for x in range(q)]
        self.neg = [self._from([(-u) % p for u in self.digits[x]]) for x in range(q)]
        self.mul = [[self._mul(x, y) for y in range(q)] for x in range(q)]
        self.inv = [0] * q
        for x in range(1, q):
            hits = [y for y in range(1, q) if self.mul[x][y] == 1]
            assert len(hits) == 1, ("modulus not irreducible", q)
            self.inv[x] = hits[0]
        assert e > 1, "prime fields are not used"
        self.a = p
        self.frob1 = [self.power(x, p) for x in range(q)]
        self.prim = next(x for x in range(2, q) if self._order(x) == q - 1)
        self.T = [[self.trace(self.mul[self.basis(r)][self.basis(s)]) for s in range(e)] for r in range(e)]

    def _from(self, ds: list[int]) -> int:
        return sum(d * self.p ** j for j, d in enumerate(ds))

    def _mul(self, x: int, y: int) -> int:
        p, e = self.p, self.e
        dx, dy = self.digits[x], self.digits[y]
        prod = [0] * (2 * e - 1)
        for i, u in enumerate(dx):
            if u:
                for j, v in enumerate(dy):
                    prod[i + j] = (prod[i + j] + u * v) % p
        for deg in range(2 * e - 2, e - 1, -1):
            c = prod[deg]
            if c:
                for j in range(e + 1):
                    prod[deg - e + j] = (prod[deg - e + j] - c * self.mod[j]) % p
        return self._from(prod[:e])

    def _order(self, x: int) -> int:
        k, y = 1, x
        while y != 1:
            y = self.mul[y][x]
            k += 1
        return k

    def subfield_generator(self, f: int) -> int:
        return self.power(self.prim, (self.q - 1) // (self.p ** f - 1))

    def basis(self, j: int) -> int:
        return self.p ** j

    def power(self, x: int, k: int) -> int:
        out = 1
        for _ in range(k):
            out = self.mul[out][x]
        return out

    def frob(self, x: int, r: int) -> int:
        for _ in range(r % self.e):
            x = self.frob1[x]
        return x

    def trace(self, x: int) -> int:
        s = 0
        y = x
        for _ in range(self.e):
            s = self.add[s][y]
            y = self.frob1[y]
        assert s < self.p, "trace outside F_p"
        return s

    def sub(self, x: int, y: int) -> int:
        return self.add[x][self.neg[y]]

    def from_int(self, k: int) -> int:
        return k % self.p

    def name(self, x: int) -> str:
        terms = []
        for j in range(self.e - 1, -1, -1):
            c = self.digits[x][j]
            if c == 0:
                continue
            mono = "" if j == 0 else ("a" if j == 1 else "a^%d" % j)
            terms.append(str(c) if j == 0 else (mono if c == 1 else "%d%s" % (c, mono)))
        return "+".join(terms) if terms else "0"

    def degree(self, elems: list[int]) -> int:
        for f in range(1, self.e + 1):
            if self.e % f == 0 and all(self.frob(x, f) == x for x in elems):
                return f
        raise AssertionError

    def mulmat(self, h: int) -> list[list[int]]:
        """Matrix of x -> h x on F_p^e (columns: images of the power basis)."""
        cols = [self.digits[self.mul[h][self.basis(j)]] for j in range(self.e)]
        return [[cols[j][i] for j in range(self.e)] for i in range(self.e)]

    def frobmat(self, r: int) -> list[list[int]]:
        cols = [self.digits[self.frob(self.basis(j), r)] for j in range(self.e)]
        return [[cols[j][i] for j in range(self.e)] for i in range(self.e)]


# ------------------------------------------------------------ linear algebra
def rref(F: Field, rows: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    rows = [list(r) for r in rows]
    piv = []
    r = 0
    ncols = len(rows[0]) if rows else 0
    for c in range(ncols):
        hit = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if hit is None:
            continue
        rows[r], rows[hit] = rows[hit], rows[r]
        iv = F.inv[rows[r][c]]
        rows[r] = [F.mul[iv][x] for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [F.sub(x, F.mul[f][y]) for x, y in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    return rows[:r], piv


def kernel(F: Field, rows: list[list[int]], ncols: int) -> list[list[int]]:
    red, piv = rref(F, rows) if rows else ([], [])
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for fc in free:
        v = [0] * ncols
        v[fc] = 1
        for i, pc in enumerate(piv):
            v[pc] = F.neg[red[i][fc]]
        out.append(v)
    return out


def rank(F: Field, rows: list[list[int]]) -> int:
    return len(rref(F, rows)[0]) if rows else 0


def modp_rref(p: int, rows: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    rows = [[x % p for x in r] for r in rows]
    piv = []
    r = 0
    ncols = len(rows[0]) if rows else 0
    for c in range(ncols):
        hit = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if hit is None:
            continue
        rows[r], rows[hit] = rows[hit], rows[r]
        iv = pow(rows[r][c], p - 2, p)
        rows[r] = [(x * iv) % p for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [(x - f * y) % p for x, y in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    return rows[:r], piv


def modp_kernel(p: int, rows: list[list[int]], ncols: int) -> list[list[int]]:
    red, piv = modp_rref(p, rows)
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for fc in free:
        v = [0] * ncols
        v[fc] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-red[i][fc]) % p
        out.append(v)
    return out


def matmul(p: int, A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) % p for j in range(len(B[0]))] for i in range(len(A))]


def transpose(A: list[list[int]]) -> list[list[int]]:
    return [list(r) for r in zip(*A)]


# --------------------------------------------------------------------- codes
def dual(F: Field, G: list[list[int]]) -> list[list[int]]:
    return kernel(F, G, len(G[0]))


def is_mds(F: Field, G: list[list[int]]) -> bool:
    k, n = len(G), len(G[0])
    for S in itertools.combinations(range(n), k):
        if rank(F, [[row[j] for j in S] for row in G]) < k:
            return False
    return True


def conj(F: Field, G: list[list[int]], r: int) -> list[list[int]]:
    return [[F.frob(x, r) for x in row] for row in G]


def linear_conductor_dim(F: Field, GE: list[list[int]], GF_: list[list[int]]) -> int:
    """dim_{F_q} Cond(E, F) = n - rank(E * F^perp)."""
    HF = dual(F, GF_)
    n = len(GE[0])
    prods = [[F.mul[x][y] for x, y in zip(g, h)] for g in GE for h in HF]
    return n - rank(F, prods)


def normalized(F: Field, G: list[list[int]], info: tuple[int, ...]) -> list[list[int]]:
    k, n = len(G), len(G[0])
    rest = [j for j in range(n) if j not in info]
    red, piv = rref(F, [[row[j] for j in info] + [row[j] for j in rest] for row in G])
    assert piv == list(range(k))
    A = [r[k:] for r in red]
    return [[F.mul[F.mul[A[i][j]][A[0][0]]][F.inv[F.mul[A[i][0]][A[0][j]]]]
             for j in range(n - k)] for i in range(k)]


def cross_ratio_degree(F: Field, G: list[list[int]]) -> int:
    k, n = len(G), len(G[0])
    if k < 2 or n - k < 2:
        return 1
    degs = {F.degree([x for row in normalized(F, G, I) for x in row])
            for I in itertools.combinations(range(n), k)}
    assert len(degs) == 1, "cross-ratio field depends on the information set"
    return degs.pop()


def additive_conductor(F: Field, GE: list[list[int]], GF_: list[list[int]]) -> list[list[list[list[int]]]]:
    """Basis of Cond_{F_p}(E, F); each element is a list of n e-by-e matrices."""
    p, e, n = F.p, F.e, len(GE[0])
    HF = dual(F, GF_)
    nv = n * e * e
    rows = []
    for g in GE:
        for j in range(e):
            vx = [F.digits[F.mul[F.basis(j)][x]] for x in g]
            for h in HF:
                mh = [F.mulmat(x) for x in h]
                for t in range(e):
                    row = [0] * nv
                    for i in range(n):
                        for r in range(e):
                            c = mh[i][t][r]
                            if c:
                                for s in range(e):
                                    if vx[i][s]:
                                        row[(i * e + r) * e + s] += c * vx[i][s]
                    rows.append(row)
    ker = modp_kernel(p, rows, nv)
    return [[[v[(i * e + r) * e: (i * e + r) * e + e] for r in range(e)] for i in range(n)] for v in ker]


def all_combinations(p: int, basis: list, limit: int):
    D = len(basis)
    if D == 0 or p ** D > limit:
        return None
    out = []
    for c in itertools.product(range(p), repeat=D):
        if any(c):
            out.append(c)
    return out


def combine(p: int, basis: list, coeffs) -> list[list[list[int]]]:
    n, e = len(basis[0]), len(basis[0][0])
    out = []
    for i in range(n):
        out.append([[sum(c * b[i][r][s] for c, b in zip(coeffs, basis)) % p for s in range(e)] for r in range(e)])
    return out


def semilinear_power(F: Field, alpha: list[list[list[int]]], f: int) -> list[int]:
    """All r in [0, f) with alpha_i(x u) = x^{p^r} alpha_i(u) for x generating F_{p^f}."""
    x = F.subfield_generator(f)
    mx = F.mulmat(x)
    out = []
    for r in range(f):
        mr = F.mulmat(F.frob(x, r))
        if all(matmul(F.p, A, mx) == matmul(F.p, mr, A) for A in alpha):
            out.append(r)
    return out


def analyse_pair(F: Field, name: str, GE, GF_) -> dict:
    e = F.e
    assert is_mds(F, GE) and is_mds(F, GF_)
    basis = additive_conductor(F, GE, GF_)
    sectors = [r for r in range(e) if linear_conductor_dim(F, conj(F, GE, r), GF_) == 1]
    assert all(linear_conductor_dim(F, conj(F, GE, r), GF_) <= 1 for r in range(e))
    fE, fF = cross_ratio_degree(F, GE), cross_ratio_degree(F, GF_)
    rec = {"name": name, "q": F.q, "n": len(GE[0]), "k": len(GE), "cross_ratio_degree_E": fE,
           "cross_ratio_degree_F": fF, "dim_Fp": len(basis), "sectors_frob_powers": sectors}
    assert len(basis) == e * len(sectors), rec
    if sectors:
        assert fE == fF and len(sectors) == e // fE, rec
        assert all((r - sectors[0]) % fE == 0 for r in sectors), rec
        combos = all_combinations(F.p, basis, 4096)
        elems = [combine(F.p, basis, c) for c in combos] if combos is not None else []
        if combos is None:
            elems = [b for b in basis]
        assert all(all(any(any(row) for row in A) for A in al) for al in elems), "support"
        rec["full_support_checked"] = len(elems)
        rec["full_support_exhaustive"] = combos is not None
        sig = {tuple(semilinear_power(F, al, fE)) for al in basis}
        assert len(sig) == 1 and len(next(iter(sig))) == 1, rec
        rec["sigma_frob_power_on_K"] = next(iter(sig))[0]
        assert rec["sigma_frob_power_on_K"] == sectors[0] % fE
    return rec


# ------------------------------------------------------------- label groups
def label_group(F: Field, G: list[list[int]]) -> dict:
    p, e, n = F.p, F.e, len(G[0])
    Gd = dual(F, G)
    blocks = [(additive_conductor(F, G, G), 0, 0), (additive_conductor(F, Gd, G), 0, 1),
              (additive_conductor(F, G, Gd), 1, 0), (additive_conductor(F, Gd, Gd), 1, 1)]
    dims = [len(b) for b, _, _ in blocks]
    basis = []
    for bl, r0, c0 in blocks:
        for b in bl:
            arr = []
            for i in range(n):
                M = [[0] * (2 * e) for _ in range(2 * e)]
                for r in range(e):
                    for s in range(e):
                        M[r0 * e + r][c0 * e + s] = b[i][r][s]
                arr.append(M)
            basis.append(arr)
    D = len(basis)
    Om = [[0] * (2 * e) for _ in range(2 * e)]
    for r in range(e):
        for s in range(e):
            Om[r][e + s] = F.T[r][s]
            Om[e + r][s] = (-F.T[r][s]) % p
    mx = F.mulmat(F.a)

    def blockdiag(m):
        Z = [[0] * (2 * e) for _ in range(2 * e)]
        for i in range(e):
            for j in range(e):
                Z[i][j] = m[i][j]
                Z[e + i][e + j] = m[i][j]
        return Z
    Mx = blockdiag(mx)
    Mt = [blockdiag(F.mulmat(F.frob(F.a, r))) for r in range(e)]
    group = []
    by_tau = [0] * e
    mixed = 0
    for c in itertools.product(range(p), repeat=D):
        g = [[[sum(ci * b[i][r][s] for ci, b in zip(c, basis)) % p for s in range(2 * e)]
              for r in range(2 * e)] for i in range(n)]
        if all(matmul(p, matmul(p, transpose(gi), Om), gi) == Om for gi in g):
            group.append(g)
            hit = False
            for r in range(e):
                if all(matmul(p, gi, Mx) == matmul(p, Mt[r], gi) for gi in g):
                    by_tau[r] += 1
                    hit = True
            mixed += 0 if hit else 1
    site0 = {tuple(map(tuple, g[0])) for g in group}
    assert len(site0) == len(group), "projection to one site is not injective"
    return {"block_dims_Fp": dims, "order": len(group), "semilinear_by_frob_power": by_tau,
            "sector_mixing": mixed}, group


def modp_inverse(p: int, A: list[list[int]]) -> list[list[int]] | None:
    m = len(A)
    red, piv = modp_rref(p, [list(r) + [int(i == j) for j in range(m)] for i, r in enumerate(A)])
    if piv[:m] != list(range(m)):
        return None
    return [r[m:] for r in red]


def zero_type_label_group(F: Field, G: list[list[int]]) -> dict:
    """Zero type: both off-diagonal blocks vanish, so a label map is a pair of
    diagonal blocks (M_i, M'_i) with M_i^T T M'_i = T at every site.  For each
    X-block the Z-block is forced; it is looked up among all Z-blocks of the
    algebra.  Avoids enumerating the product of the two block spaces."""
    p, e, n = F.p, F.e, len(G[0])
    Gd = dual(F, G)
    assert not additive_conductor(F, Gd, G) and not additive_conductor(F, G, Gd), "not zero type"
    Xb, Zb = additive_conductor(F, G, G), additive_conductor(F, Gd, Gd)
    key = lambda blk: tuple(tuple(map(tuple, M)) for M in blk)
    zs = {}
    for c in itertools.product(range(p), repeat=len(Zb)):
        blk = combine(p, Zb, c)
        zs[key(blk)] = blk
    mx = F.mulmat(F.a)
    mt = [F.mulmat(F.frob(F.a, r)) for r in range(e)]
    Om = [[0] * (2 * e) for _ in range(2 * e)]
    for r in range(e):
        for s in range(e):
            Om[r][e + s] = F.T[r][s]
            Om[e + r][s] = (-F.T[r][s]) % p
    order, mixed, by_tau, site0 = 0, 0, [0] * e, set()
    for c in itertools.product(range(p), repeat=len(Xb)):
        xb = combine(p, Xb, c)
        need = []
        for M in xb:
            inv = modp_inverse(p, matmul(p, transpose(M), F.T))
            if inv is None:
                break
            need.append(matmul(p, inv, F.T))
        if len(need) != n or key(need) not in zs:
            continue
        zb = zs[key(need)]
        for M, Mp in zip(xb, zb):  # direct check of the trace-symplectic condition
            g = [[0] * (2 * e) for _ in range(2 * e)]
            for r in range(e):
                for s in range(e):
                    g[r][s], g[e + r][e + s] = M[r][s], Mp[r][s]
            assert matmul(p, matmul(p, transpose(g), Om), g) == Om
        order += 1
        site0.add((key(xb[:1]), key(zb[:1])))
        hits = [r for r in range(e)
                if all(matmul(p, M, mx) == matmul(p, mt[r], M) for M in xb + zb)]
        xhits = [r for r in range(e)
                 if all(matmul(p, M, mx) == matmul(p, mt[r], M) for M in xb)]
        assert hits == xhits, "Z-block semilinearity is not forced by the X-block"
        for r in hits:
            by_tau[r] += 1
        mixed += 0 if hits else 1
    assert len(site0) == order, "projection to one site is not injective"
    return {"block_dims_Fp": [len(Xb), 0, 0, len(Zb)], "order": order,
            "semilinear_by_frob_power": by_tau, "sector_mixing": mixed}


def unitary_model(F: Field, G: list[list[int]], group: list) -> dict:
    """Hermitian branch with cross-ratio field F_q: check the explicit
    isomorphism g -> [[a, b], [s(c), s(d)]] onto U([[0,1],[-1,0]])."""
    p, e, q = F.p, F.e, F.q
    assert e % 2 == 0
    h = e // 2
    sig = lambda x: F.frob(x, h)
    S = F.frobmat(h)

    def val(A):  # F_p matrix applied to the element 1
        return F._from([A[i][0] for i in range(e)])

    def blk(M, r0, c0):
        return [[M[r0 * e + i][c0 * e + j] for j in range(e)] for i in range(e)]

    images = {}
    for g in group:
        M = g[0]
        al, be, ga, de = blk(M, 0, 0), blk(M, 0, 1), blk(M, 1, 0), blk(M, 1, 1)
        a, d = val(al), val(de)
        b = F._from([matmul(p, be, S)[i][0] for i in range(e)])  # beta = m_b sigma, sigma(1)=1
        c = F._from([matmul(p, ga, S)[i][0] for i in range(e)])
        assert al == F.mulmat(a) and de == F.mulmat(d), "diagonal block not F_q-linear"
        assert be == matmul(p, F.mulmat(b), S) and ga == matmul(p, F.mulmat(c), S), "off-diagonal not sigma-semilinear"
        images[tuple(map(tuple, M))] = (a, b, sig(c), sig(d))
    assert len(set(images.values())) == len(group), "model map not injective"

    def mm(X, Y):
        A, B, C, D = X
        A2, B2, C2, D2 = Y
        ad, mu = F.add, F.mul
        return (ad[mu[A][A2]][mu[B][C2]], ad[mu[A][B2]][mu[B][D2]],
                ad[mu[C][A2]][mu[D][C2]], ad[mu[C][B2]][mu[D][D2]])

    keys = list(images)
    for X in keys:
        for Y in keys:
            prod = tuple(map(tuple, matmul(p, [list(r) for r in X], [list(r) for r in Y])))
            assert prod in images, "group not closed"
            assert images[prod] == mm(images[X], images[Y]), "model map not a homomorphism"
    unitary = set()
    for A, B, C, D in itertools.product(range(q), repeat=4):
        mu, ad = F.mul, F.add
        if (mu[A][sig(C)] == mu[C][sig(A)] and mu[B][sig(D)] == mu[D][sig(B)]
                and F.sub(mu[A][sig(D)], mu[C][sig(B)]) == 1
                and F.sub(mu[B][sig(C)], mu[D][sig(A)]) == F.neg[1]):
            unitary.add((A, B, C, D))
    r = int(round(q ** 0.5))
    assert r * r == q
    gu_order = r * (r * r - 1) * (r + 1)
    assert len(unitary) == gu_order
    assert set(images.values()) == unitary, "image is not the unitary group"
    return {"unitary_group_order": len(unitary), "GU2_order_formula": gu_order,
            "model_map_isomorphism": True}


# -------------------------------------------------------------------- pencil
def pencil_H(F: Field, t: int) -> list[list[int]]:
    one = 1
    u = F.sub(one, t)
    mu = F.neg[u]
    cols = [(0, 1, u), (0, 1, mu), (1, u, 0), (1, mu, 0), (1, 0, F.neg[t]), (1, 0, t)]
    return [[c[i] for c in cols] for i in range(3)]


def pencil_poly(F: Field, t: int, coeffs: list[int]) -> int:
    out = 0
    for c in reversed(coeffs):
        out = F.add[F.mul[out][t]][F.from_int(c)]
    return out


def in_U(F: Field, t: int) -> bool:
    vals = [t, F.sub(t, 1), pencil_poly(F, t, [1, -1, 1]), pencil_poly(F, t, [1, -3, 1]),
            pencil_poly(F, t, [1, -4, 7, -4, 1])]
    return all(v != 0 for v in vals)


def D_tau(F: Field, t: int, r: int) -> int:
    s = F.frob(t, r)
    hh = F.mul[F.sub(1, s)][F.sub(1, t)]
    return F.add[F.mul[hh][hh]][F.mul[s][t]]


def y_of(F: Field, t: int) -> int:
    u = F.sub(t, 1)
    return F.mul[F.mul[u][u]][F.inv[t]]


def pencil_sweep(q: int) -> dict:
    F = Field(q)
    e = F.e
    points = []
    hermitian = []
    for t in range(q):
        if not in_U(F, t):
            continue
        H = pencil_H(F, t)
        Cp = rref(F, H)[0]
        C = dual(F, H)
        dims = [len(additive_conductor(F, A, B)) for A, B in ((C, C), (Cp, C), (C, Cp), (Cp, Cp))]
        fK = cross_ratio_degree(F, C)
        y = y_of(F, t)
        fy = F.degree([y])
        yp, ym = F.add[y][1], F.sub(y, 1)
        two = F.from_int(2)
        expected_N = [[1, 1, 1],
                      [1, F.neg[F.mul[yp][F.inv[ym]]], F.neg[F.mul[ym][F.inv[yp]]]],
                      [1, F.neg[F.mul[two][F.inv[ym]]], F.mul[two][F.inv[yp]]]]
        assert normalized(F, C, (0, 1, 2)) == expected_N, (q, F.name(t))
        sectors = [r for r in range(e) if linear_conductor_dim(F, conj(F, C, r), Cp) == 1]
        droots = [r for r in range(e) if D_tau(F, t, r) == 0]
        assert fK == fy, (q, t)
        assert dims[0] == dims[3] == e * (e // fK), (q, t, dims)
        assert dims[1] == dims[2] == e * len(sectors), (q, t, dims)
        assert sectors == droots, (q, F.name(t), sectors, droots)
        assert 0 not in droots
        assert all((2 * r) % fK == 0 for r in droots)
        points.append(1)
        if sectors:
            hermitian.append({"t": F.name(t), "cross_ratio_degree": fK, "sectors_frob_powers": sectors})
    return {"q": q, "modulus_low_to_high": MODULI[q][1], "regular_points": len(points),
            "hermitian_gale_points": len(hermitian),
            "hermitian_gale_points_with_K_full": sum(1 for h in hermitian if h["cross_ratio_degree"] == e),
            "hermitian_gale": hermitian}


# -------------------------------------------------------------------- driver
def lcg_mds(F: Field, k: int, n: int, seed: int) -> list[list[int]]:
    state = seed
    while True:
        rows = []
        for i in range(k):
            row = []
            for j in range(n - k):
                state = (1103515245 * state + 12345) % (1 << 31)
                row.append((state >> 16) % F.q)
            rows.append([1 if c == i else 0 for c in range(k)] + row)
        if is_mds(F, rows):
            return rows


def scale(F: Field, G, d):
    return [[F.mul[x][y] for x, y in zip(row, d)] for row in G]


def hexacode(F: Field) -> list[list[int]]:
    w = F.subfield_generator(2)
    return [[1, 0, 0, 1, w, w], [0, 1, 0, w, 1, w], [0, 0, 1, w, w, 1]]


def code_pairs() -> list[dict]:
    out = []
    F4 = Field(4)
    hx = hexacode(F4)
    out += [analyse_pair(F4, "hexacode C->C", hx, hx), analyse_pair(F4, "hexacode C->C^perp", hx, dual(F4, hx)),
            analyse_pair(F4, "hexacode C^perp->C", dual(F4, hx), hx)]
    E = lcg_mds(F4, 1, 4, 3)
    out.append(analyse_pair(F4, "[4,1]_4 E->E", E, E))
    E = lcg_mds(F4, 3, 4, 4)
    out.append(analyse_pair(F4, "[4,3]_4 E->E", E, E))
    F8 = Field(8)
    E = lcg_mds(F8, 3, 6, 5)
    D8 = [1, 2, 3, 4, 5, 6]
    out += [analyse_pair(F8, "[6,3]_8 C->C", E, E), analyse_pair(F8, "[6,3]_8 C->C^perp", E, dual(F8, E)),
            analyse_pair(F8, "[6,3]_8 E->D*Frob(E)", E, scale(F8, conj(F8, E, 1), D8))]
    F9 = Field(9)
    tet = scale(F9, [[1, 0, 1, 1], [0, 1, 1, 2]], [1, 3, 5, 7])
    out += [analyse_pair(F9, "D*tetracode in F_9 C->C", tet, tet),
            analyse_pair(F9, "D*tetracode in F_9 C->C^perp", tet, dual(F9, tet))]
    E = lcg_mds(F9, 3, 6, 6)
    out += [analyse_pair(F9, "[6,3]_9 C->C", E, E), analyse_pair(F9, "[6,3]_9 C->C^perp", E, dual(F9, E))]
    F16 = Field(16)
    hx16 = scale(F16, hexacode(F16), [1, 2, 3, 5, 7, 11])
    R16 = lcg_mds(F16, 3, 6, 7)
    out += [analyse_pair(F16, "D*hexacode in F_16 C->C", hx16, hx16),
            analyse_pair(F16, "D*hexacode in F_16 C->C^perp", hx16, dual(F16, hx16)),
            analyse_pair(F16, "hexacode in F_16 -> [6,3]_16", hx16, R16)]
    F27 = Field(27)
    tet27 = scale(F27, [[1, 0, 1, 1], [0, 1, 1, 2]], [1, 4, 10, 13])
    out.append(analyse_pair(F27, "D*tetracode in F_27 C->C", tet27, tet27))
    return out


def label_groups() -> list[dict]:
    out = []
    F4 = Field(4)
    hx = hexacode(F4)
    rec, grp = label_group(F4, hx)
    rec.update(name="hexacode [6,3,4]_4", branch="Hermitian", q=4)
    rec.update(unitary_model(F4, hx, grp))
    out.append(rec)
    F9 = Field(9)
    for t in range(9):
        if not in_U(F9, t) or D_tau(F9, t, 1) != 0:
            continue
        C = dual(F9, pencil_H(F9, t))
        rec, grp = label_group(F9, C)
        rec.update(name="pencil t=%s" % F9.name(t), branch="Hermitian", q=9)
        rec.update(unitary_model(F9, C, grp))
        out.append(rec)
    grs9 = scale(F9, [[1, 1, 1, 0], [0, 1, F9.a, 1]], [1, 2, F9.a, 5])
    assert cross_ratio_degree(F9, grs9) == 2
    rec, _ = label_group(F9, grs9)
    rec.update(name="D*GRS[4,2]_9", branch="Euclidean", q=9)
    assert rec["order"] == 720
    out.append(rec)
    F25 = Field(25)
    t = F25.mul[2][F25.a]
    C = dual(F25, pencil_H(F25, t))
    rec, _ = label_group(F25, C)
    assert zero_type_label_group(F25, C) == rec  # cross-check of the zero-type shortcut
    rec.update(name="pencil t=%s" % F25.name(t), branch="zero", q=25)
    assert rec["order"] == 24
    out.append(rec)
    for r in out:
        if r["branch"] == "Hermitian":
            assert r["order"] - r["sector_mixing"] == 2 * (r["q"] - 1)
    return out


def subfield_point() -> dict:
    """t = 2a+1 in F_49: t generates F_49, but E_Frob(t) = 0 and K_{C_t} = F_7."""
    F = Field(49)
    t = F.add[F.mul[2][F.a]][1]
    assert in_U(F, t)
    C = dual(F, pencil_H(F, t))
    Cp = rref(F, pencil_H(F, t))[0]
    s = F.frob(t, 1)
    E = F.mul[F.sub(s, t)][F.sub(1, F.mul[s][t])]
    rec = {"q": 49, "t": F.name(t), "degree_t": F.degree([t]), "E_frob_zero": E == 0,
           "cross_ratio_degree": cross_ratio_degree(F, C),
           "D_zero_frob_powers": [r for r in range(2) if D_tau(F, t, r) == 0],
           "dim_Fp_C_to_C": len(additive_conductor(F, C, C)),
           "dim_Fp_C_to_Cperp": len(additive_conductor(F, C, Cp))}
    assert rec["degree_t"] == 2 and rec["E_frob_zero"] and rec["cross_ratio_degree"] == 1
    assert rec["D_zero_frob_powers"] == [] and rec["dim_Fp_C_to_C"] == 4 and rec["dim_Fp_C_to_Cperp"] == 0
    # Zero type with K = F_7, d = 2: the label group is GL_2(7), and its semilinear
    # elements are those whose X-block is c*tau, c in F_49^x, tau in {1, Frob}.
    grp = zero_type_label_group(F, C)
    assert grp["order"] == (49 - 1) * (49 - 7)
    assert grp["semilinear_by_frob_power"] == [48, 48] and grp["sector_mixing"] == 2016 - 96
    rec["label_group"] = grp
    return rec


def build_certificate() -> dict:
    return {
        "schema": SCHEMA,
        "conventions": {
            "fields": "F_{p^e} = F_p[a]/(modulus), element digits base p are coefficients of a^j",
            "conductor": "coordinatewise F_p-linear maps sending E into F; dim over F_p",
            "sectors": "frob powers r with dim_{F_q} Cond(Frob^r E, F) = 1 (Schur-product rank)",
            "label_group": "sitewise F_p-linear maps preserving C_X + C^perp_Z and tr(xz'-x'z) at every site",
            "D_tau": "((1-tau t)(1-t))^2 + t tau(t)",
        },
        "code_pairs": code_pairs(),
        "label_groups": label_groups(),
        "pencil_sweep": [pencil_sweep(q) for q in (9, 25, 27, 49, 81, 121, 125, 169)],
        "pencil_subfield_point": subfield_point(),
    }


def canonical_bytes(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    certificate = build_certificate()
    generated = canonical_bytes(certificate)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"missing certificate: {OUTPUT}")
        if OUTPUT.read_bytes() != generated:
            with tempfile.NamedTemporaryFile(prefix="additive-", suffix=".json", delete=False) as handle:
                handle.write(generated)
                temporary = handle.name
            raise SystemExit(f"certificate mismatch; regenerated copy: {temporary}")
        print(f"PASS additive conductors; {len(generated)} bytes")
    else:
        OUTPUT.write_bytes(generated)
        print(f"WROTE {OUTPUT} ({len(generated)} bytes)")


if __name__ == "__main__":
    main()
