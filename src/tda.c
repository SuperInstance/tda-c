#include "tda.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

/* ========== Point Cloud ========== */

TDAPointCloud *tda_pointcloud_create(size_t n_points, size_t dim) {
    TDAPointCloud *pc = malloc(sizeof(TDAPointCloud));
    if (!pc) return NULL;
    pc->n_points = n_points;
    pc->dim = dim;
    pc->coords = calloc(n_points * dim, sizeof(double));
    if (!pc->coords) { free(pc); return NULL; }
    return pc;
}

void tda_pointcloud_set(TDAPointCloud *pc, size_t idx, const double *coords) {
    if (idx >= pc->n_points) return;
    memcpy(pc->coords + idx * pc->dim, coords, pc->dim * sizeof(double));
}

double tda_pointcloud_get(const TDAPointCloud *pc, size_t idx, size_t axis) {
    if (idx >= pc->n_points || axis >= pc->dim) return 0.0;
    return pc->coords[idx * pc->dim + axis];
}

void tda_pointcloud_free(TDAPointCloud *pc) {
    if (!pc) return;
    free(pc->coords);
    free(pc);
}

/* ========== Distance Matrix ========== */

TDADistanceMatrix *tda_distance_matrix_create(size_t n) {
    TDADistanceMatrix *dm = malloc(sizeof(TDADistanceMatrix));
    if (!dm) return NULL;
    dm->n = n;
    dm->data = calloc(n * n, sizeof(double));
    if (!dm->data) { free(dm); return NULL; }
    return dm;
}

double tda_distance_matrix_get(const TDADistanceMatrix *dm, size_t i, size_t j) {
    return dm->data[i * dm->n + j];
}

void tda_distance_matrix_set(TDADistanceMatrix *dm, size_t i, size_t j, double val) {
    dm->data[i * dm->n + j] = val;
}

void tda_distance_matrix_free(TDADistanceMatrix *dm) {
    if (!dm) return;
    free(dm->data);
    free(dm);
}

TDADistanceMatrix *tda_compute_euclidean_distances(const TDAPointCloud *pc) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(pc->n_points);
    if (!dm) return NULL;

    for (size_t i = 0; i < pc->n_points; i++) {
        tda_distance_matrix_set(dm, i, i, 0.0);
        for (size_t j = i + 1; j < pc->n_points; j++) {
            double sum = 0.0;
            for (size_t d = 0; d < pc->dim; d++) {
                double diff = tda_pointcloud_get(pc, i, d) - tda_pointcloud_get(pc, j, d);
                sum += diff * diff;
            }
            double dist = sqrt(sum);
            tda_distance_matrix_set(dm, i, j, dist);
            tda_distance_matrix_set(dm, j, i, dist);
        }
    }
    return dm;
}

/* ========== Internal: Simplex helpers ========== */

static int simplex_cmp_dim_then_birth(const void *a, const void *b) {
    const TDASimplex *sa = (const TDASimplex *)a;
    const TDASimplex *sb = (const TDASimplex *)b;
    /* Sort by (birth_time, dimension) */
    if (sa->birth < sb->birth) return -1;
    if (sa->birth > sb->birth) return 1;
    if (sa->n_vertices < sb->n_vertices) return -1;
    if (sa->n_vertices > sb->n_vertices) return 1;
    return 0;
}

/* Compute filtration value of a simplex (max pairwise distance among vertices) */
static double simplex_filtration(const TDADistanceMatrix *dm,
                                  const uint32_t *vertices, size_t n_vertices) {
    double max_dist = 0.0;
    for (size_t i = 0; i < n_vertices; i++) {
        for (size_t j = i + 1; j < n_vertices; j++) {
            double d = tda_distance_matrix_get(dm, vertices[i], vertices[j]);
            if (d > max_dist) max_dist = d;
        }
    }
    return max_dist;
}

/* ========== Vietoris-Rips Complex ========== */

/* Internal builder with dynamic array */
typedef struct {
    TDASimplex *items;
    size_t len;
    size_t cap;
} SimplexVec;

static void simplex_vec_init(SimplexVec *v) {
    v->cap = 256;
    v->len = 0;
    v->items = malloc(v->cap * sizeof(TDASimplex));
    if (!v->items) { v->cap = 0; }
}

static int simplex_vec_push(SimplexVec *v, const TDASimplex *s) {
    if (v->len >= v->cap) {
        v->cap *= 2;
        TDASimplex *new_items = realloc(v->items, v->cap * sizeof(TDASimplex));
        if (!new_items) return -1;
        v->items = new_items;
    }
    v->items[v->len++] = *s;
    return 0;
}

/* Recursive co-face enumeration */
static void enumerate_simplices(const TDADistanceMatrix *dm,
                                 uint32_t *current, size_t current_len,
                                 size_t max_dim, double epsilon,
                                 uint32_t start, SimplexVec *out) {
    if (current_len > max_dim) return;

    size_t n = dm->n;
    for (uint32_t v = start; v < n; v++) {
        current[current_len] = v;
        double filtr = simplex_filtration(dm, current, current_len + 1);
        if (filtr <= epsilon + 1e-12) {
            TDASimplex s;
            s.n_vertices = current_len + 1;
            s.vertices = malloc((current_len + 1) * sizeof(uint32_t));
            memcpy(s.vertices, current, (current_len + 1) * sizeof(uint32_t));
            s.birth = filtr;
            s.death = TDA_INFINITY;
            simplex_vec_push(out, &s);

            enumerate_simplices(dm, current, current_len + 1, max_dim, epsilon,
                                v + 1, out);
        }
    }
}

TDAVietorisRips *tda_rips_build(const TDADistanceMatrix *dm, double epsilon,
                                 size_t max_dim) {
    SimplexVec vec;
    simplex_vec_init(&vec);

    uint32_t *current = malloc((max_dim + 2) * sizeof(uint32_t));

    /* Add empty simplex (vertex set of size 0) — skip, start from vertices */

    /* Add vertices (0-simplices) */
    for (size_t i = 0; i < dm->n; i++) {
        TDASimplex s;
        s.n_vertices = 1;
        s.vertices = malloc(sizeof(uint32_t));
        s.vertices[0] = (uint32_t)i;
        s.birth = 0.0;
        s.death = TDA_INFINITY;
        simplex_vec_push(&vec, &s);
    }

    /* Build higher simplices starting from edges */
    for (uint32_t i = 0; i < dm->n; i++) {
        for (uint32_t j = i + 1; j < dm->n; j++) {
            double d = tda_distance_matrix_get(dm, i, j);
            if (d <= epsilon + 1e-12) {
                /* Add edge (i, j) */
                TDASimplex s;
                s.n_vertices = 2;
                s.vertices = malloc(2 * sizeof(uint32_t));
                s.vertices[0] = i;
                s.vertices[1] = j;
                s.birth = d;
                s.death = TDA_INFINITY;
                simplex_vec_push(&vec, &s);

                /* Extend to higher dimensions */
                current[0] = i;
                current[1] = j;
                if (max_dim >= 2) {
                    enumerate_simplices(dm, current, 2, max_dim + 1, epsilon,
                                        j + 1, &vec);
                }
            }
        }
    }
    free(current);

    /* Sort by filtration value then dimension */
    qsort(vec.items, vec.len, sizeof(TDASimplex), simplex_cmp_dim_then_birth);

    TDAVietorisRips *vr = malloc(sizeof(TDAVietorisRips));
    vr->simplices = vec.items;
    vr->n_simplices = vec.len;
    vr->capacity = vec.cap;
    return vr;
}

void tda_rips_free(TDAVietorisRips *vr) {
    if (!vr) return;
    for (size_t i = 0; i < vr->n_simplices; i++) {
        free(vr->simplices[i].vertices);
    }
    free(vr->simplices);
    free(vr);
}

/* ========== Persistence: Union-Find based matrix reduction ========== */

/* Simple Union-Find */
typedef struct {
    int *parent;
    int *rank;
    size_t n;
} UnionFind;

static UnionFind *uf_create(size_t n) {
    UnionFind *uf = malloc(sizeof(UnionFind));
    uf->parent = malloc(n * sizeof(int));
    uf->rank = calloc(n, sizeof(int));
    uf->n = n;
    for (size_t i = 0; i < n; i++) uf->parent[i] = (int)i;
    return uf;
}

static int uf_find(UnionFind *uf, int x) {
    while (uf->parent[x] != x) {
        uf->parent[x] = uf->parent[uf->parent[x]]; /* path splitting */
        x = uf->parent[x];
    }
    return x;
}

static int uf_union(UnionFind *uf, int a, int b) {
    a = uf_find(uf, a);
    b = uf_find(uf, b);
    if (a == b) return 0; /* already same set */
    if (uf->rank[a] < uf->rank[b]) { int t = a; a = b; b = t; }
    uf->parent[b] = a;
    if (uf->rank[a] == uf->rank[b]) uf->rank[a]++;
    return 1;
}

static void uf_free(UnionFind *uf) {
    free(uf->parent);
    free(uf->rank);
    free(uf);
}

/* Edge for sorting */
typedef struct {
    uint32_t i, j;
    double dist;
} Edge;

static int edge_cmp(const void *a, const void *b) {
    double da = ((const Edge *)a)->dist;
    double db = ((const Edge *)b)->dist;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Internal: persistence diagram builder */
typedef struct {
    TDAPDPoint *items;
    size_t len;
    size_t cap;
} PDVec;

static void pd_vec_init(PDVec *v) {
    v->cap = 64;
    v->len = 0;
    v->items = malloc(v->cap * sizeof(TDAPDPoint));
}

static void pd_vec_push(PDVec *v, double birth, double death, int dim) {
    if (v->len >= v->cap) {
        v->cap *= 2;
        TDAPDPoint *new_items = realloc(v->items, v->cap * sizeof(TDAPDPoint));
        if (!new_items) return;
        v->items = new_items;
    }
    v->items[v->len].birth = birth;
    v->items[v->len].death = death;
    v->items[v->len].dimension = dim;
    v->len++;
}

/* Compute connected components (H0) and basic H1 persistence using
 * edge filtration. For H0 we use Union-Find. For H1+, we detect cycles. */

TDAPersistenceDiagram *tda_compute_persistence(const TDADistanceMatrix *dm,
                                                 size_t max_hom_dim) {
    if (!dm) {
        TDAPersistenceDiagram *pd = calloc(1, sizeof(TDAPersistenceDiagram));
        return pd;
    }
    size_t n = dm->n;
    if (n == 0) {
        TDAPersistenceDiagram *pd = calloc(1, sizeof(TDAPersistenceDiagram));
        return pd;
    }

    PDVec vec;
    pd_vec_init(&vec);

    /* Collect all edges */
    size_t edge_cap = n * (n - 1) / 2;
    Edge *edges = malloc(edge_cap * sizeof(Edge));
    size_t n_edges = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            edges[n_edges].i = (uint32_t)i;
            edges[n_edges].j = (uint32_t)j;
            edges[n_edges].dist = tda_distance_matrix_get(dm, i, j);
            n_edges++;
        }
    }
    qsort(edges, n_edges, sizeof(Edge), edge_cmp);

    /* ---- H0: connected components ---- */
    UnionFind *uf = uf_create(n);
    int *root_alive = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) root_alive[i] = 1;

    for (size_t e = 0; e < n_edges; e++) {
        int ri = uf_find(uf, (int)edges[e].i);
        int rj = uf_find(uf, (int)edges[e].j);
        if (ri != rj) {
            /* Perform union first, then determine which root survived */
            uf_union(uf, ri, rj);
            int new_root = uf_find(uf, ri);
            int dead = (new_root == ri) ? rj : ri;

            if (root_alive[dead]) {
                pd_vec_push(&vec, 0.0, edges[e].dist, 0);
                root_alive[dead] = 0;
            }
            /* Ensure the surviving root is marked alive */
            root_alive[new_root] = 1;
        }
    }

    /* Remaining alive components live forever */
    for (size_t i = 0; i < n; i++) {
        int ri = uf_find(uf, (int)i);
        if (root_alive[ri]) {
            pd_vec_push(&vec, 0.0, TDA_INFINITY, 0);
            root_alive[ri] = 0;
        }
    }

    /* ---- H1 and higher: cycle detection via Union-Find ---- */
    /* For H1: re-scan edges; edges that connect already-connected vertices create 1-cycles.
     * The birth time is the edge weight. Death time requires finding when the cycle is filled.
     * For a proper implementation we need the full matrix reduction.
     * Here we use a simplified approach: for H1, each such edge births a cycle,
     * and death is estimated as 2x the birth (approximation) or infinity for
     * the most persistent ones.
     *
     * Actually, let's do a proper boundary matrix reduction for small inputs.
     */

    if (max_hom_dim >= 1 && n <= 30) {
        /* Build simplices and do full reduction */
        /* Collect all simplices up to dimension max_hom_dim + 1 */
        /* For H1 we need edges and triangles.
         * An edge (i,j) appears at filtration dm[i][j].
         * A triangle (i,j,k) appears at max(dm[i][j], dm[i][k], dm[j][k]).
         */

        /* Enumerate triangles */
        typedef struct { uint32_t a, b, c; double filt; } Tri;
        Tri *tris = NULL;
        size_t n_tris = 0;
        size_t tri_cap = 0;

        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                for (size_t k = j + 1; k < n; k++) {
                    double d01 = tda_distance_matrix_get(dm, i, j);
                    double d02 = tda_distance_matrix_get(dm, i, k);
                    double d12 = tda_distance_matrix_get(dm, j, k);
                    double filt = d01;
                    if (d02 > filt) filt = d02;
                    if (d12 > filt) filt = d12;
                    if (n_tris >= tri_cap) {
                        tri_cap = tri_cap ? tri_cap * 2 : 256;
                        Tri *new_tris = realloc(tris, tri_cap * sizeof(Tri));
                        if (!new_tris) { free(tris); free(edges); free(root_alive); uf_free(uf); free(vec.items); return NULL; }
                        tris = new_tris;
                    }
                    tris[n_tris].a = (uint32_t)i;
                    tris[n_tris].b = (uint32_t)j;
                    tris[n_tris].c = (uint32_t)k;
                    tris[n_tris].filt = filt;
                    n_tris++;
                }
            }
        }

        /* Boundary matrix reduction for H1.
         * Columns = edges (in filtration order), rows = triangles (in filtration order).
         * Actually, standard reduction: columns are simplices sorted by filtration.
         * For H1, the 1-simplices (edges) are boundaries of 2-simplices (triangles).
         *
         * We'll do the standard algorithm:
         * 1. List all simplices (vertices, edges, triangles) sorted by filtration
         * 2. Build boundary matrix
         * 3. Reduce from left to right
         */

        /* Total simplices: n vertices + n_edges edges + n_tris triangles */
        size_t total = n + n_edges + n_tris;

        typedef struct { size_t dim; double filt; size_t idx; uint32_t verts[3]; } SimplexInfo;
        SimplexInfo *all_sx = malloc(total * sizeof(SimplexInfo));
        size_t sx_count = 0;

        /* Vertices */
        for (size_t i = 0; i < n; i++) {
            all_sx[sx_count].dim = 0;
            all_sx[sx_count].filt = 0.0;
            all_sx[sx_count].idx = sx_count;
            all_sx[sx_count].verts[0] = (uint32_t)i;
            sx_count++;
        }
        /* Edges */
        for (size_t e = 0; e < n_edges; e++) {
            all_sx[sx_count].dim = 1;
            all_sx[sx_count].filt = edges[e].dist;
            all_sx[sx_count].idx = sx_count;
            all_sx[sx_count].verts[0] = edges[e].i;
            all_sx[sx_count].verts[1] = edges[e].j;
            sx_count++;
        }
        /* Triangles */
        for (size_t t = 0; t < n_tris; t++) {
            all_sx[sx_count].dim = 2;
            all_sx[sx_count].filt = tris[t].filt;
            all_sx[sx_count].idx = sx_count;
            all_sx[sx_count].verts[0] = tris[t].a;
            all_sx[sx_count].verts[1] = tris[t].b;
            all_sx[sx_count].verts[2] = tris[t].c;
            sx_count++;
        }

        free(tris);

        int simplex_info_cmp(const void *a, const void *b) {
            const SimplexInfo *sa = (const SimplexInfo *)a;
            const SimplexInfo *sb = (const SimplexInfo *)b;
            if (sa->filt < sb->filt) return -1;
            if (sa->filt > sb->filt) return 1;
            if (sa->dim < sb->dim) return -1;
            if (sa->dim > sb->dim) return 1;
            return 0;
        }

        qsort(all_sx, sx_count, sizeof(SimplexInfo), simplex_info_cmp);

        /* Map: for each simplex, what's its column index in the boundary matrix? */
        /* Build a lookup: vertex/edge/triangle -> column index */
        /* We need to map simplex (verts, dim) -> index in all_sx */

        /* Build boundary matrix (bitfield: for n<=30, total<=30+435+4060 < 5000)
         * Use char matrix for simplicity */
        unsigned char *boundary = calloc(sx_count * sx_count, sizeof(unsigned char));

        /* Fill boundary: for simplex j of dimension d, its boundary is
         * the (d-1)-simplices obtained by removing each vertex */
        /* First build a map from (dim, sorted vertices) -> index */
        /* For small n, just search */

        for (size_t j = 0; j < sx_count; j++) {
            if (all_sx[j].dim == 0) continue;

            /* Boundary of simplex j: remove each vertex one at a time */
            int n_v = (int)all_sx[j].dim + 1;
            for (int skip = 0; skip < n_v; skip++) {
                /* Build the face without vertex skip */
                uint32_t face_verts[3];
                int fv = 0;
                for (int vi = 0; vi < n_v; vi++) {
                    if (vi == skip) continue;
                    face_verts[fv++] = all_sx[j].verts[vi];
                }
                /* Sort */
                for (int a = 0; a < fv - 1; a++)
                    for (int b = a + 1; b < fv; b++)
                        if (face_verts[a] > face_verts[b]) {
                            uint32_t t = face_verts[a];
                            face_verts[a] = face_verts[b];
                            face_verts[b] = t;
                        }

                /* Find this face in all_sx */
                int face_dim = n_v - 2;
                for (size_t k = 0; k < sx_count; k++) {
                    if ((int)all_sx[k].dim != face_dim) continue;
                    int match = 1;
                    for (int vi = 0; vi < fv; vi++) {
                        if (all_sx[k].verts[vi] != face_verts[vi]) { match = 0; break; }
                    }
                    if (match) {
                        boundary[k * sx_count + j] ^= 1;
                        break;
                    }
                }
            }
        }

        /* Reduce: left-to-right column reduction */
        int *lowest_one = malloc(sx_count * sizeof(int));
        for (size_t i = 0; i < sx_count; i++) lowest_one[i] = -1;

        for (size_t j = 0; j < sx_count; j++) {
            /* Find lowest 1 in column j */
            int low = -1;
            for (int i = (int)sx_count - 1; i >= 0; i--) {
                if (boundary[i * sx_count + j]) { low = i; break; }
            }
            while (low != -1 && lowest_one[low] != -1) {
                /* Add column lowest_one[low] to column j */
                size_t k = (size_t)lowest_one[low];
                for (size_t i = 0; i < sx_count; i++) {
                    boundary[i * sx_count + j] ^= boundary[i * sx_count + k];
                }
                low = -1;
                for (int i = (int)sx_count - 1; i >= 0; i--) {
                    if (boundary[i * sx_count + j]) { low = i; break; }
                }
            }
            if (low != -1) {
                lowest_one[low] = (int)j;
            }
        }

        /* Read off persistence pairs:
         * If column j has lowest_one[j] = k, then simplex k is paired with simplex j:
         *   - birth = filtration[k], death = filtration[j]
         *   - dimension = dim(k) = dim(j) - 1
         * Unpaired columns represent births at their filtration with death = infinity.
         */
        int *paired = calloc(sx_count, sizeof(int));

        for (size_t j = 0; j < sx_count; j++) {
            if (lowest_one[j] != -1) {
                size_t k = (size_t)lowest_one[j];
                paired[j] = 1;
                paired[k] = 1;
                /* j = row index (cycle that is born), k = column index (boundary that kills it) */
                int dim = (int)all_sx[j].dim;
                if (dim <= (int)max_hom_dim && dim >= 1) {
                    pd_vec_push(&vec, all_sx[j].filt, all_sx[k].filt, dim);
                }
            }
        }

        /* Unpaired simplices that are not vertices (H0 handled above) */
        /* Actually, for H0 we already have all pairs. For H1+ unpaired means infinity death. */
        /* But we already added H0 above. So only add H1+ unpaired */
        for (size_t j = 0; j < sx_count; j++) {
            if (!paired[j] && all_sx[j].dim >= 1 && all_sx[j].dim <= max_hom_dim) {
                pd_vec_push(&vec, all_sx[j].filt, TDA_INFINITY, (int)all_sx[j].dim);
            }
        }

        free(boundary);
        free(lowest_one);
        free(paired);
        free(all_sx);
    } else if (max_hom_dim >= 1) {
        /* For large inputs, use Union-Find based H1 approximation */
        UnionFind *uf2 = uf_create(n);
        for (size_t e = 0; e < n_edges; e++) {
            int ri = uf_find(uf2, (int)edges[e].i);
            int rj = uf_find(uf2, (int)edges[e].j);
            if (ri == rj) {
                /* This edge creates a cycle → H1 birth */
                pd_vec_push(&vec, edges[e].dist, TDA_INFINITY, 1);
            } else {
                uf_union(uf2, (int)edges[e].i, (int)edges[e].j);
            }
        }
        uf_free(uf2);
    }

    free(edges);
    free(root_alive);
    uf_free(uf);

    TDAPersistenceDiagram *pd = malloc(sizeof(TDAPersistenceDiagram));
    pd->points = vec.items;
    pd->n_points = vec.len;
    pd->capacity = vec.cap;
    return pd;
}

/* ========== Betti Numbers ========== */

size_t *tda_betti_numbers(const TDADistanceMatrix *dm, double epsilon,
                           size_t max_dim) {
    size_t *betti = calloc(max_dim + 1, sizeof(size_t));
    if (!betti) return NULL;

    if (dm->n == 0) return betti;

    /* Compute persistence diagram */
    TDAPersistenceDiagram *pd = tda_compute_persistence(dm, max_dim);
    if (!pd) return betti;

    /* Count features alive at epsilon:
     * birth <= epsilon AND (death > epsilon OR death = infinity) */
    for (size_t i = 0; i < pd->n_points; i++) {
        int dim = pd->points[i].dimension;
        if (dim < 0 || (size_t)dim > max_dim) continue;
        if (pd->points[i].birth <= epsilon + 1e-12 &&
            (pd->points[i].death > epsilon + 1e-12 || isinf(pd->points[i].death))) {
            betti[dim]++;
        }
    }

    tda_persistence_diagram_free(pd);
    return betti;
}

/* ========== Bottleneck Distance ========== */

/* Compute cost of matching under diagonal using Hungarian-like approach.
 * For simplicity, use a greedy algorithm with the proper definition:
 * bottleneck = max over all matched pairs of the L-infinity distance,
 *              considering unmatched points matched to the diagonal.
 */

static double point_cost(double b1, double d1, double b2, double d2) {
    double cb1 = (b1 + d1) / 2.0;
    double cd1 = (d1 - b1) / 2.0;
    double cb2 = (b2 + d2) / 2.0;
    double cd2 = (d2 - b2) / 2.0;
    double dx = cb1 - cb2;
    double dy = cd1 - cd2;
    return sqrt(dx * dx + dy * dy);
}

static double diag_cost(double b, double d) {
    /* Distance to diagonal: |d - b| / 2 */
    return fabs(d - b) / 2.0;
}

double tda_bottleneck_distance(const TDAPersistenceDiagram *pd1,
                                const TDAPersistenceDiagram *pd2,
                                int dimension) {
    /* Filter points by dimension */
    size_t n1 = 0, n2 = 0;
    for (size_t i = 0; i < pd1->n_points; i++)
        if (dimension < 0 || pd1->points[i].dimension == dimension)
            if (!isinf(pd1->points[i].death) || pd1->points[i].birth < pd1->points[i].death)
                n1++;
    for (size_t i = 0; i < pd2->n_points; i++)
        if (dimension < 0 || pd2->points[i].dimension == dimension)
            if (!isinf(pd2->points[i].death) || pd2->points[i].birth < pd2->points[i].death)
                n2++;

    /* Collect filtered points */
    typedef struct { double b, d; } BDPoint;
    BDPoint *p1 = malloc((n1 + 1) * sizeof(BDPoint));
    BDPoint *p2 = malloc((n2 + 1) * sizeof(BDPoint));
    size_t i1 = 0, i2 = 0;

    for (size_t i = 0; i < pd1->n_points; i++) {
        if (dimension < 0 || pd1->points[i].dimension == dimension) {
            if (isinf(pd1->points[i].death) && pd1->points[i].birth >= pd1->points[i].death) continue;
            p1[i1].b = pd1->points[i].birth;
            p1[i1].d = isinf(pd1->points[i].death) ? pd1->points[i].birth + 1e10 : pd1->points[i].death;
            i1++;
        }
    }
    for (size_t i = 0; i < pd2->n_points; i++) {
        if (dimension < 0 || pd2->points[i].dimension == dimension) {
            if (isinf(pd2->points[i].death) && pd2->points[i].birth >= pd2->points[i].death) continue;
            p2[i2].b = pd2->points[i].birth;
            p2[i2].d = isinf(pd2->points[i].death) ? pd2->points[i].birth + 1e10 : pd2->points[i].death;
            i2++;
        }
    }

    if (n1 == 0 && n2 == 0) { free(p1); free(p2); return 0.0; }

    /* Build the complete bipartite cost matrix:
     * Size = (n1 + n2) x (n2 + n1)
     * Top-left: point-to-point costs
     * Diagonal extensions: unmatched points go to diagonal
     */
    size_t m = n1 + n2;
    double *cost = malloc(m * m * sizeof(double));

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            if (i < n1 && j < n2) {
                cost[i * m + j] = point_cost(p1[i].b, p1[i].d, p2[j].b, p2[j].d);
            } else if (i < n1 && j >= n2) {
                /* p1[i] matched to diagonal */
                cost[i * m + j] = (j - n2 == i) ? diag_cost(p1[i].b, p1[i].d) : 1e18;
            } else if (i >= n1 && j < n2) {
                /* p2[j] matched to diagonal */
                cost[i * m + j] = (i - n1 == j) ? diag_cost(p2[j].b, p2[j].d) : 1e18;
            } else {
                cost[i * m + j] = ((i - n1) == (j - n2)) ? 0.0 : 1e18;
            }
        }
    }

    /* Simple greedy matching (for correctness, we should use Hungarian, but
     * for the bottleneck case we can binary-search + check augmenting paths).
     * Use the simpler approach: collect all costs, sort, binary search for
     * feasibility with augmenting path check.
     */

    /* Collect unique costs for binary search */
    double *all_costs = malloc(m * m * sizeof(double));
    size_t n_costs = 0;
    for (size_t i = 0; i < m * m; i++) {
        if (cost[i] < 1e17) all_costs[n_costs++] = cost[i];
    }

    int double_cmp(const void *a, const void *b) {
        double da = *(const double *)a;
        double db = *(const double *)b;
        if (da < db) return -1;
        if (da > db) return 1;
        return 0;
    }

    /* Sort */
    qsort(all_costs, n_costs, sizeof(double), double_cmp);

    /* Remove duplicates */
    size_t unique = 1;
    for (size_t i = 1; i < n_costs; i++) {
        if (all_costs[i] != all_costs[unique - 1])
            all_costs[unique++] = all_costs[i];
    }
    n_costs = unique;

    /* Use iterative matching approach */
    int *match_l = malloc(m * sizeof(int));
    int *match_r = malloc(m * sizeof(int));
    int *visited = malloc(m * sizeof(int));
    int *stk = malloc(m * sizeof(int));
    int *parent = malloc(m * sizeof(int));

    /* Binary search for minimum threshold */
    double lo = 0, hi = n_costs > 0 ? all_costs[n_costs - 1] : 0.0;
    double result = hi;

    while (n_costs > 0 && hi - lo > 1e-10) {
        double mid = (lo + hi) / 2.0;

        /* Check if perfect matching exists with cost <= mid */
        memset(match_l, -1, m * sizeof(int));
        memset(match_r, -1, m * sizeof(int));
        int ok = 1;

        for (size_t u = 0; u < m && ok; u++) {
            memset(visited, 0, m * sizeof(int));
            int found = 0;
            for (size_t v = 0; v < m && !found; v++) {
                if (visited[v]) continue;
                if (cost[u * m + v] > mid) continue;
                visited[v] = 1;
                if (match_r[v] < 0) {
                    match_l[u] = (int)v;
                    match_r[v] = (int)u;
                    found = 1;
                } else {
                    /* Try to augment from match_r[v] via DFS */
                    stk[0] = match_r[v];
                    int sp = 1;
                    parent[v] = (int)u;
                    int aug_found = 0;
                    while (sp > 0 && !aug_found) {
                        int cur = stk[--sp];
                        for (size_t w = 0; w < m; w++) {
                            if (visited[w]) continue;
                            if (cost[cur * m + w] > mid) continue;
                            visited[w] = 1;
                            parent[w] = cur;
                            if (match_r[w] < 0) {
                                /* Found augmenting path, unwind */
                                int pv = (int)w;
                                while (1) {
                                    int pu = parent[pv];
                                    int next_pv = match_l[pu];
                                    match_l[pu] = pv;
                                    match_r[pv] = pu;
                                    if (pu == (int)u) break;
                                    pv = next_pv;
                                }
                                aug_found = 1;
                                found = 1;
                                break;
                            } else {
                                stk[sp++] = match_r[w];
                            }
                        }
                    }
                }
            }
            if (!found) ok = 0;
        }

        if (ok) {
            result = mid;
            hi = mid;
        } else {
            lo = mid;
        }
    }

    free(stk);
    free(parent);
    free(match_l);
    free(match_r);
    free(visited);
    free(cost);
    free(all_costs);
    free(p1);
    free(p2);

    return result;
}

/* ========== Persistence Diagram Cleanup ========== */

void tda_persistence_diagram_free(TDAPersistenceDiagram *pd) {
    if (!pd) return;
    free(pd->points);
    free(pd);
}

/* ========== Barcodes ========== */

TDABarcodes *tda_barcodes_from_persistence(const TDAPersistenceDiagram *pd) {
    TDABarcodes *bc = malloc(sizeof(TDABarcodes));
    bc->n_bars = pd->n_points;
    bc->capacity = pd->n_points;
    bc->bars = malloc(pd->n_points * sizeof(TDABarcode));

    for (size_t i = 0; i < pd->n_points; i++) {
        bc->bars[i].start = pd->points[i].birth;
        bc->bars[i].end = pd->points[i].death;
        bc->bars[i].dimension = pd->points[i].dimension;
    }
    return bc;
}

void tda_barcodes_free(TDABarcodes *bc) {
    if (!bc) return;
    free(bc->bars);
    free(bc);
}
