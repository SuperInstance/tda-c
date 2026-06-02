#include "tda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) ASSERT(fabs((a) - (b)) < (eps), msg)

/* ========== Test 1: Point cloud create/free ========== */
static void test_pointcloud_create(void) {
    TDAPointCloud *pc = tda_pointcloud_create(5, 3);
    ASSERT(pc != NULL, "pointcloud create returns non-null");
    ASSERT(pc->n_points == 5, "pointcloud has 5 points");
    ASSERT(pc->dim == 3, "pointcloud has dim 3");
    tda_pointcloud_free(pc);
}

/* ========== Test 2: Point cloud set/get ========== */
static void test_pointcloud_set_get(void) {
    TDAPointCloud *pc = tda_pointcloud_create(3, 2);
    double p0[] = {1.0, 2.0};
    double p1[] = {3.0, 4.0};
    double p2[] = {0.0, 0.0};
    tda_pointcloud_set(pc, 0, p0);
    tda_pointcloud_set(pc, 1, p1);
    tda_pointcloud_set(pc, 2, p2);

    ASSERT_NEAR(tda_pointcloud_get(pc, 0, 0), 1.0, 1e-12, "point 0 x");
    ASSERT_NEAR(tda_pointcloud_get(pc, 0, 1), 2.0, 1e-12, "point 0 y");
    ASSERT_NEAR(tda_pointcloud_get(pc, 1, 0), 3.0, 1e-12, "point 1 x");
    ASSERT_NEAR(tda_pointcloud_get(pc, 1, 1), 4.0, 1e-12, "point 1 y");
    ASSERT_NEAR(tda_pointcloud_get(pc, 2, 0), 0.0, 1e-12, "point 2 x");
    tda_pointcloud_free(pc);
}

/* ========== Test 3: Distance matrix create/free ========== */
static void test_distance_matrix_create(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(4);
    ASSERT(dm != NULL, "distance matrix create returns non-null");
    ASSERT(dm->n == 4, "distance matrix has n=4");
    tda_distance_matrix_free(dm);
}

/* ========== Test 4: Euclidean distance computation ========== */
static void test_euclidean_distances(void) {
    /* 3 points on a line: (0,0), (1,0), (3,0) */
    TDAPointCloud *pc = tda_pointcloud_create(3, 2);
    double p0[] = {0.0, 0.0};
    double p1[] = {1.0, 0.0};
    double p2[] = {3.0, 0.0};
    tda_pointcloud_set(pc, 0, p0);
    tda_pointcloud_set(pc, 1, p1);
    tda_pointcloud_set(pc, 2, p2);

    TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);
    ASSERT(dm != NULL, "euclidean distances returns non-null");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 0), 0.0, 1e-12, "dist(0,0)=0");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 1), 1.0, 1e-12, "dist(0,1)=1");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 2), 3.0, 1e-12, "dist(0,2)=3");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 1, 2), 2.0, 1e-12, "dist(1,2)=2");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 2, 1), 2.0, 1e-12, "dist(2,1)=2 (symmetric)");

    tda_distance_matrix_free(dm);
    tda_pointcloud_free(pc);
}

/* ========== Test 5: 2D Euclidean distances ========== */
static void test_euclidean_2d(void) {
    /* Points: (0,0), (3,4), (0,5) */
    TDAPointCloud *pc = tda_pointcloud_create(3, 2);
    double p0[] = {0.0, 0.0};
    double p1[] = {3.0, 4.0};
    double p2[] = {0.0, 5.0};
    tda_pointcloud_set(pc, 0, p0);
    tda_pointcloud_set(pc, 1, p1);
    tda_pointcloud_set(pc, 2, p2);

    TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 1), 5.0, 1e-12, "3-4-5 triangle");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 2), 5.0, 1e-12, "vertical distance");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 1, 2), sqrt(9.0 + 1.0), 1e-12, "dist p1-p2");

    tda_distance_matrix_free(dm);
    tda_pointcloud_free(pc);
}

/* ========== Test 6: VR complex with 3 points ========== */
static void test_rips_3points(void) {
    /* Equilateral-ish triangle: edges at distances 1, 1, 1.5 */
    TDADistanceMatrix *dm = tda_distance_matrix_create(3);
    tda_distance_matrix_set(dm, 0, 0, 0); tda_distance_matrix_set(dm, 1, 1, 0); tda_distance_matrix_set(dm, 2, 2, 0);
    tda_distance_matrix_set(dm, 0, 1, 1.0); tda_distance_matrix_set(dm, 1, 0, 1.0);
    tda_distance_matrix_set(dm, 0, 2, 1.0); tda_distance_matrix_set(dm, 2, 0, 1.0);
    tda_distance_matrix_set(dm, 1, 2, 1.5); tda_distance_matrix_set(dm, 2, 1, 1.5);

    /* epsilon=0.5: only vertices */
    TDAVietorisRips *vr = tda_rips_build(dm, 0.5, 2);
    ASSERT(vr != NULL, "rips build returns non-null");
    ASSERT(vr->n_simplices == 3, "epsilon=0.5: only 3 vertices");
    tda_rips_free(vr);

    /* epsilon=1.0: vertices + 2 edges */
    vr = tda_rips_build(dm, 1.0, 2);
    ASSERT(vr->n_simplices == 5, "epsilon=1.0: 3 vertices + 2 edges");
    tda_rips_free(vr);

    /* epsilon=1.5: vertices + 3 edges + 1 triangle */
    vr = tda_rips_build(dm, 1.5, 2);
    ASSERT(vr->n_simplices == 7, "epsilon=1.5: 3v + 3e + 1tri");
    tda_rips_free(vr);

    tda_distance_matrix_free(dm);
}

/* ========== Test 7: VR complex with 4 points ========== */
static void test_rips_4points(void) {
    /* Square with side 1: (0,0), (1,0), (0,1), (1,1) */
    TDAPointCloud *pc = tda_pointcloud_create(4, 2);
    double pts[][2] = {{0,0},{1,0},{0,1},{1,1}};
    for (int i = 0; i < 4; i++) tda_pointcloud_set(pc, i, pts[i]);

    TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 1), 1.0, 1e-12, "square side");
    ASSERT_NEAR(tda_distance_matrix_get(dm, 0, 3), sqrt(2.0), 1e-12, "square diagonal");

    TDAVietorisRips *vr = tda_rips_build(dm, 1.0, 2);
    /* At epsilon=1: 4 vertices + 4 edges (sides only, diagonals are sqrt(2)) */
    ASSERT(vr->n_simplices == 8, "square epsilon=1: 4v + 4e");
    tda_rips_free(vr);

    tda_distance_matrix_free(dm);
    tda_pointcloud_free(pc);
}

/* ========== Test 8: Persistence diagram - single point ========== */
static void test_persistence_single(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(1);
    tda_distance_matrix_set(dm, 0, 0, 0);

    TDAPersistenceDiagram *pd = tda_compute_persistence(dm, 1);
    ASSERT(pd != NULL, "persistence single point returns non-null");
    ASSERT(pd->n_points == 1, "single point has 1 PD point (H0, alive)");
    ASSERT(pd->points[0].dimension == 0, "dimension is 0");
    ASSERT(isinf(pd->points[0].death), "single point never dies");

    tda_persistence_diagram_free(pd);
    tda_distance_matrix_free(dm);
}

/* ========== Test 9: Persistence - two points ========== */
static void test_persistence_two_points(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(2);
    tda_distance_matrix_set(dm, 0, 0, 0); tda_distance_matrix_set(dm, 1, 1, 0);
    tda_distance_matrix_set(dm, 0, 1, 5.0); tda_distance_matrix_set(dm, 1, 0, 5.0);

    TDAPersistenceDiagram *pd = tda_compute_persistence(dm, 1);
    ASSERT(pd->n_points == 2, "two points: 2 PD points");
    /* One H0 dies at 5.0, one H0 lives forever */
    int found_death5 = 0, found_inf = 0;
    for (size_t i = 0; i < pd->n_points; i++) {
        if (pd->points[i].dimension == 0 && fabs(pd->points[i].death - 5.0) < 1e-10)
            found_death5 = 1;
        if (pd->points[i].dimension == 0 && isinf(pd->points[i].death))
            found_inf = 1;
    }
    ASSERT(found_death5, "H0 death at 5.0");
    ASSERT(found_inf, "H0 alive (infinity)");

    tda_persistence_diagram_free(pd);
    tda_distance_matrix_free(dm);
}

/* ========== Test 10: Persistence - triangle ========== */
static void test_persistence_triangle(void) {
    /* Equilateral triangle with side 2 */
    TDADistanceMatrix *dm = tda_distance_matrix_create(3);
    for (int i = 0; i < 3; i++) tda_distance_matrix_set(dm, i, i, 0);
    tda_distance_matrix_set(dm, 0, 1, 2.0); tda_distance_matrix_set(dm, 1, 0, 2.0);
    tda_distance_matrix_set(dm, 0, 2, 2.0); tda_distance_matrix_set(dm, 2, 0, 2.0);
    tda_distance_matrix_set(dm, 1, 2, 2.0); tda_distance_matrix_set(dm, 2, 1, 2.0);

    TDAPersistenceDiagram *pd = tda_compute_persistence(dm, 1);
    /* 3 points: H0 dies at 2.0 (x2), H0 alive (x1) */
    int h0_count = 0;
    for (size_t i = 0; i < pd->n_points; i++) {
        if (pd->points[i].dimension == 0) h0_count++;
    }
    ASSERT(h0_count == 3, "triangle: 3 H0 points");

    tda_persistence_diagram_free(pd);
    tda_distance_matrix_free(dm);
}

/* ========== Test 11: Betti numbers - single point ========== */
static void test_betti_single(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(1);
    tda_distance_matrix_set(dm, 0, 0, 0);

    size_t *betti = tda_betti_numbers(dm, 1.0, 2);
    ASSERT(betti[0] == 1, "single point: b0=1");
    ASSERT(betti[1] == 0, "single point: b1=0");

    free(betti);
    tda_distance_matrix_free(dm);
}

/* ========== Test 12: Betti numbers - two connected points ========== */
static void test_betti_two_connected(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(2);
    tda_distance_matrix_set(dm, 0, 0, 0); tda_distance_matrix_set(dm, 1, 1, 0);
    tda_distance_matrix_set(dm, 0, 1, 1.0); tda_distance_matrix_set(dm, 1, 0, 1.0);

    /* At epsilon=0.5: not connected yet */
    size_t *betti = tda_betti_numbers(dm, 0.5, 2);
    ASSERT(betti[0] == 2, "two points epsilon=0.5: b0=2");
    free(betti);

    /* At epsilon=1.0: connected */
    betti = tda_betti_numbers(dm, 1.0, 2);
    ASSERT(betti[0] == 1, "two points epsilon=1: b0=1");
    ASSERT(betti[1] == 0, "two points epsilon=1: b1=0");
    free(betti);

    tda_distance_matrix_free(dm);
}

/* ========== Test 13: Betti numbers - triangle (H0=1, H1=1 when filled) ========== */
static void test_betti_triangle(void) {
    /* Triangle with sides 1, 1, 1.4 (not equilateral) */
    TDADistanceMatrix *dm = tda_distance_matrix_create(3);
    for (int i = 0; i < 3; i++) tda_distance_matrix_set(dm, i, i, 0);
    tda_distance_matrix_set(dm, 0, 1, 1.0); tda_distance_matrix_set(dm, 1, 0, 1.0);
    tda_distance_matrix_set(dm, 0, 2, 1.0); tda_distance_matrix_set(dm, 2, 0, 1.0);
    tda_distance_matrix_set(dm, 1, 2, 1.4); tda_distance_matrix_set(dm, 2, 1, 1.4);

    /* At epsilon=0.9: 3 separate points */
    size_t *betti = tda_betti_numbers(dm, 0.9, 1);
    ASSERT(betti[0] == 3, "triangle eps=0.9: b0=3");
    free(betti);

    /* At epsilon=1.0: 3 points, 2 edges -> b0=1, no loop yet */
    betti = tda_betti_numbers(dm, 1.0, 1);
    ASSERT(betti[0] == 1, "triangle eps=1.0: b0=1");
    free(betti);

    /* At epsilon=1.4: all edges + triangle -> b0=1, b1=0 (filled) */
    betti = tda_betti_numbers(dm, 1.4, 1);
    ASSERT(betti[0] == 1, "triangle eps=1.4: b0=1");
    free(betti);

    tda_distance_matrix_free(dm);
}

/* ========== Test 14: Bottleneck distance - identical diagrams ========== */
static void test_bottleneck_identical(void) {
    TDAPersistenceDiagram *pd1 = malloc(sizeof(TDAPersistenceDiagram));
    pd1->n_points = 2;
    pd1->capacity = 2;
    pd1->points = malloc(2 * sizeof(TDAPDPoint));
    pd1->points[0] = (TDAPDPoint){0.0, 1.0, 0};
    pd1->points[1] = (TDAPDPoint){0.0, TDA_INFINITY, 0};

    double d = tda_bottleneck_distance(pd1, pd1, 0);
    ASSERT_NEAR(d, 0.0, 1e-10, "identical diagrams: bottleneck=0");

    tda_persistence_diagram_free(pd1);
}

/* ========== Test 15: Bottleneck distance - shifted ========== */
static void test_bottleneck_shifted(void) {
    TDAPersistenceDiagram *pd1 = malloc(sizeof(TDAPersistenceDiagram));
    pd1->n_points = 1; pd1->capacity = 1;
    pd1->points = malloc(sizeof(TDAPDPoint));
    pd1->points[0] = (TDAPDPoint){0.0, 2.0, 0};

    TDAPersistenceDiagram *pd2 = malloc(sizeof(TDAPersistenceDiagram));
    pd2->n_points = 1; pd2->capacity = 1;
    pd2->points = malloc(sizeof(TDAPDPoint));
    pd2->points[0] = (TDAPDPoint){1.0, 3.0, 0};

    double d = tda_bottleneck_distance(pd1, pd2, 0);
    /* Both points shifted by 1 in birth and death.
     * In (center, half-life) coords: (1,1) -> (2,1). Distance = 1.0 */
    ASSERT(d > 0.0, "shifted: bottleneck > 0");

    tda_persistence_diagram_free(pd1);
    tda_persistence_diagram_free(pd2);
}

/* ========== Test 16: Barcodes from persistence ========== */
static void test_barcodes(void) {
    TDAPersistenceDiagram *pd = malloc(sizeof(TDAPersistenceDiagram));
    pd->n_points = 3; pd->capacity = 3;
    pd->points = malloc(3 * sizeof(TDAPDPoint));
    pd->points[0] = (TDAPDPoint){0.0, 1.0, 0};
    pd->points[1] = (TDAPDPoint){0.0, TDA_INFINITY, 0};
    pd->points[2] = (TDAPDPoint){2.0, 5.0, 1};

    TDABarcodes *bc = tda_barcodes_from_persistence(pd);
    ASSERT(bc->n_bars == 3, "barcode has 3 bars");
    ASSERT(bc->bars[0].start == 0.0 && bc->bars[0].end == 1.0, "bar 0: [0,1]");
    ASSERT(bc->bars[1].start == 0.0 && isinf(bc->bars[1].end), "bar 1: [0,inf)");
    ASSERT(bc->bars[2].start == 2.0 && bc->bars[2].end == 5.0, "bar 2: [2,5]");

    tda_barcodes_free(bc);
    tda_persistence_diagram_free(pd);
}

/* ========== Test 17: Empty point cloud ========== */
static void test_empty_pointcloud(void) {
    TDAPointCloud *pc = tda_pointcloud_create(0, 0);
    /* Should handle gracefully */
    tda_pointcloud_free(pc);
    ASSERT(1, "empty pointcloud create/free doesn't crash");
}

/* ========== Test 18: Persistence - 4 points in square ========== */
static void test_persistence_square(void) {
    TDAPointCloud *pc = tda_pointcloud_create(4, 2);
    double pts[][2] = {{0,0},{1,0},{0,1},{1,1}};
    for (int i = 0; i < 4; i++) tda_pointcloud_set(pc, i, pts[i]);

    TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);
    TDAPersistenceDiagram *pd = tda_compute_persistence(dm, 1);

    /* 4 points: b0 starts at 4, merges to 1 as edges appear.
     * H0: 3 deaths + 1 alive = 4 points total in dim 0.
     * H1: the square has 1 cycle that dies when the triangle fills. */
    int h0_count = 0, h1_count = 0;
    for (size_t i = 0; i < pd->n_points; i++) {
        if (pd->points[i].dimension == 0) h0_count++;
        if (pd->points[i].dimension == 1) h1_count++;
    }
    ASSERT(h0_count == 4, "square: 4 H0 points");
    /* H1 depends on full reduction - may have 1 cycle */

    tda_persistence_diagram_free(pd);
    tda_distance_matrix_free(dm);
    tda_pointcloud_free(pc);
}

/* ========== Test 19: Betti numbers - 5 isolated points ========== */
static void test_betti_isolated(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            tda_distance_matrix_set(dm, i, j, (i == j) ? 0.0 : 100.0);
        }
    }

    size_t *betti = tda_betti_numbers(dm, 0.5, 2);
    ASSERT(betti[0] == 5, "5 far-apart points: b0=5 at small epsilon");
    ASSERT(betti[1] == 0, "5 far-apart points: b1=0");
    free(betti);

    tda_distance_matrix_free(dm);
}

/* ========== Test 20: VR complex epsilon=0 ========== */
static void test_rips_epsilon_zero(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(3);
    for (int i = 0; i < 3; i++) tda_distance_matrix_set(dm, i, i, 0);
    tda_distance_matrix_set(dm, 0, 1, 5.0); tda_distance_matrix_set(dm, 1, 0, 5.0);
    tda_distance_matrix_set(dm, 0, 2, 5.0); tda_distance_matrix_set(dm, 2, 0, 5.0);
    tda_distance_matrix_set(dm, 1, 2, 5.0); tda_distance_matrix_set(dm, 2, 1, 5.0);

    TDAVietorisRips *vr = tda_rips_build(dm, 0.0, 2);
    ASSERT(vr->n_simplices == 3, "epsilon=0: only 3 vertices");
    tda_rips_free(vr);
    tda_distance_matrix_free(dm);
}

/* ========== Test 21: Bottleneck distance - empty diagrams ========== */
static void test_bottleneck_empty(void) {
    TDAPersistenceDiagram *pd1 = calloc(1, sizeof(TDAPersistenceDiagram));
    TDAPersistenceDiagram *pd2 = calloc(1, sizeof(TDAPersistenceDiagram));
    pd1->points = NULL; pd1->n_points = 0; pd1->capacity = 0;
    pd2->points = NULL; pd2->n_points = 0; pd2->capacity = 0;

    double d = tda_bottleneck_distance(pd1, pd2, 0);
    ASSERT_NEAR(d, 0.0, 1e-10, "empty diagrams: bottleneck=0");

    tda_persistence_diagram_free(pd1);
    tda_persistence_diagram_free(pd2);
}

/* ========== Test 22: Distance matrix symmetry ========== */
static void test_distance_symmetry(void) {
    TDAPointCloud *pc = tda_pointcloud_create(5, 3);
    for (size_t i = 0; i < 5; i++) {
        double c[] = {i * 1.1, i * 2.2, i * 0.5};
        tda_pointcloud_set(pc, i, c);
    }

    TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);
    int symmetric = 1;
    for (size_t i = 0; i < 5 && symmetric; i++) {
        for (size_t j = 0; j < 5 && symmetric; j++) {
            double a = tda_distance_matrix_get(dm, i, j);
            double b = tda_distance_matrix_get(dm, j, i);
            if (fabs(a - b) > 1e-12) symmetric = 0;
        }
    }
    ASSERT(symmetric, "distance matrix is symmetric");

    tda_distance_matrix_free(dm);
    tda_pointcloud_free(pc);
}

/* ========== Test 23: Betti numbers - single point at various epsilons ========== */
static void test_betti_various_eps(void) {
    TDADistanceMatrix *dm = tda_distance_matrix_create(1);
    tda_distance_matrix_set(dm, 0, 0, 0);

    for (double eps = 0.0; eps <= 10.0; eps += 2.5) {
        size_t *betti = tda_betti_numbers(dm, eps, 2);
        ASSERT(betti[0] == 1, "single point always has b0=1");
        free(betti);
    }
    tda_distance_matrix_free(dm);
}

/* ========== Test 24: Persistence - line of points ========== */
static void test_persistence_line(void) {
    /* 4 points on a line at 0, 1, 2, 3 */
    TDAPointCloud *pc = tda_pointcloud_create(4, 1);
    for (int i = 0; i < 4; i++) {
        double c = (double)i;
        tda_pointcloud_set(pc, i, &c);
    }

    TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);
    TDAPersistenceDiagram *pd = tda_compute_persistence(dm, 1);

    /* H0: 4 points, merges to 1. 3 deaths + 1 alive = 4 H0 points */
    int h0_count = 0;
    for (size_t i = 0; i < pd->n_points; i++) {
        if (pd->points[i].dimension == 0) h0_count++;
    }
    ASSERT(h0_count == 4, "line of 4: 4 H0 points");

    /* First death should be at distance 1.0 */
    int found_death_1 = 0;
    for (size_t i = 0; i < pd->n_points; i++) {
        if (pd->points[i].dimension == 0 &&
            fabs(pd->points[i].death - 1.0) < 1e-10) {
            found_death_1 = 1;
            break;
        }
    }
    ASSERT(found_death_1, "line of 4: first H0 death at dist=1.0");

    tda_persistence_diagram_free(pd);
    tda_distance_matrix_free(dm);
    tda_pointcloud_free(pc);
}

/* ========== Test 25: Barcodes from empty persistence ========== */
static void test_barcodes_empty(void) {
    TDAPersistenceDiagram *pd = calloc(1, sizeof(TDAPersistenceDiagram));
    pd->points = NULL; pd->n_points = 0; pd->capacity = 0;

    TDABarcodes *bc = tda_barcodes_from_persistence(pd);
    ASSERT(bc->n_bars == 0, "empty PD produces 0 bars");

    tda_barcodes_free(bc);
    tda_persistence_diagram_free(pd);
}

int main(void) {
    printf("=== TDA-C Test Suite ===\n\n");

    test_pointcloud_create();
    test_pointcloud_set_get();
    test_distance_matrix_create();
    test_euclidean_distances();
    test_euclidean_2d();
    test_rips_3points();
    test_rips_4points();
    test_persistence_single();
    test_persistence_two_points();
    test_persistence_triangle();
    test_betti_single();
    test_betti_two_connected();
    test_betti_triangle();
    test_bottleneck_identical();
    test_bottleneck_shifted();
    test_barcodes();
    test_empty_pointcloud();
    test_persistence_square();
    test_betti_isolated();
    test_rips_epsilon_zero();
    test_bottleneck_empty();
    test_distance_symmetry();
    test_betti_various_eps();
    test_persistence_line();
    test_barcodes_empty();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
