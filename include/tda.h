#ifndef TDA_H
#define TDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Point Cloud ========== */

typedef struct {
    double *coords;   /* flattened n_points x dim */
    size_t n_points;
    size_t dim;
} TDAPointCloud;

TDAPointCloud *tda_pointcloud_create(size_t n_points, size_t dim);
void tda_pointcloud_set(TDAPointCloud *pc, size_t idx, const double *coords);
double tda_pointcloud_get(const TDAPointCloud *pc, size_t idx, size_t axis);
void tda_pointcloud_free(TDAPointCloud *pc);

/* ========== Distance Matrix ========== */

typedef struct {
    double *data;      /* flattened n x n, row-major */
    size_t n;
} TDADistanceMatrix;

TDADistanceMatrix *tda_distance_matrix_create(size_t n);
double tda_distance_matrix_get(const TDADistanceMatrix *dm, size_t i, size_t j);
void tda_distance_matrix_set(TDADistanceMatrix *dm, size_t i, size_t j, double val);
void tda_distance_matrix_free(TDADistanceMatrix *dm);

/* Compute Euclidean distance matrix from point cloud */
TDADistanceMatrix *tda_compute_euclidean_distances(const TDAPointCloud *pc);

/* ========== Simplex ========== */

typedef struct {
    uint32_t *vertices; /* sorted vertex indices */
    size_t n_vertices;  /* dimension = n_vertices - 1 */
    double birth;       /* filtration value */
    double death;       /* INFINITY if alive */
} TDASimplex;

/* ========== Vietoris-Rips Complex ========== */

typedef struct {
    TDASimplex *simplices;
    size_t n_simplices;
    size_t capacity;
} TDAVietorisRips;

/* Build VR complex up to max_dim at given epsilon threshold.
 * Returns all simplices with birth time <= epsilon. */
TDAVietorisRips *tda_rips_build(const TDADistanceMatrix *dm, double epsilon, size_t max_dim);
void tda_rips_free(TDAVietorisRips *vr);

/* ========== Persistence Diagram ========== */

typedef struct {
    double birth;
    double death;  /* INFINITY = never dies */
    int dimension; /* homology dimension */
} TDAPDPoint;

typedef struct {
    TDAPDPoint *points;
    size_t n_points;
    size_t capacity;
} TDAPersistenceDiagram;

/* Compute persistence diagram from distance matrix using Rips filtration.
 * Uses Union-Find based algorithm. Computes up to max_hom_dim. */
TDAPersistenceDiagram *tda_compute_persistence(const TDADistanceMatrix *dm,
                                                size_t max_hom_dim);
void tda_persistence_diagram_free(TDAPersistenceDiagram *pd);

/* ========== Betti Numbers ========== */

/* Compute Betti numbers b0, b1, ..., b_max_dim at filtration value epsilon */
size_t *tda_betti_numbers(const TDADistanceMatrix *dm, double epsilon,
                          size_t max_dim);

/* ========== Bottleneck Distance ========== */

/* Compute bottleneck distance between two persistence diagrams
 * (considers only points of the specified dimension, or -1 for all) */
double tda_bottleneck_distance(const TDAPersistenceDiagram *pd1,
                                const TDAPersistenceDiagram *pd2,
                                int dimension);

/* ========== Persistence Barcodes ========== */

typedef struct {
    double start;
    double end;     /* INFINITY = alive */
    int dimension;
} TDABarcode;

typedef struct {
    TDABarcode *bars;
    size_t n_bars;
    size_t capacity;
} TDABarcodes;

/* Generate barcodes from a persistence diagram */
TDABarcodes *tda_barcodes_from_persistence(const TDAPersistenceDiagram *pd);
void tda_barcodes_free(TDABarcodes *bc);

/* ========== Utility ========== */

#ifndef INFINITY
#define TDA_INFINITY (1.0 / 0.0)
#else
#define TDA_INFINITY INFINITY
#endif

#define TDA_PI 3.14159265358979323846

#ifdef __cplusplus
}
#endif

#endif /* TDA_H */
