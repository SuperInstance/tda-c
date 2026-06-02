# tda-c — Topological Data Analysis in C

A pure C implementation of topological data analysis (TDA), ported from [tda-rs](https://github.com/turnerlean/tda-rs). No external dependencies beyond the standard C math library.

## Features

- **Point cloud representation** — N-dimensional double arrays
- **Distance matrix computation** — Euclidean pairwise distances
- **Vietoris-Rips complex** — Simplicial complex construction at a given epsilon threshold
- **Persistence diagrams** — Computed via boundary matrix reduction (Rips filtration)
- **Betti numbers** — β₀, β₁, … at any filtration value
- **Bottleneck distance** — Between persistence diagrams with binary-search matching
- **Persistence barcodes** — Barcode representation of persistence diagrams

## Building

```bash
make
```

## Running Tests

```bash
make test
```

## Usage

```c
#include "tda.h"

/* Create a point cloud */
TDAPointCloud *pc = tda_pointcloud_create(5, 2);
double p[] = {1.0, 2.0};
tda_pointcloud_set(pc, 0, p);
/* ... set more points ... */

/* Compute distances */
TDADistanceMatrix *dm = tda_compute_euclidean_distances(pc);

/* Compute persistence */
TDAPersistenceDiagram *pd = tda_compute_persistence(dm, 1);

/* Get Betti numbers at epsilon = 2.0 */
size_t *betti = tda_betti_numbers(dm, 2.0, 2);
printf("b0 = %zu, b1 = %zu\n", betti[0], betti[1]);

/* Clean up */
free(betti);
tda_persistence_diagram_free(pd);
tda_distance_matrix_free(dm);
tda_pointcloud_free(pc);
```

## API Overview

| Function | Description |
|---|---|
| `tda_pointcloud_create` | Create an N-point cloud in D dimensions |
| `tda_compute_euclidean_distances` | Build Euclidean distance matrix |
| `tda_rips_build` | Build Vietoris-Rips complex up to epsilon |
| `tda_compute_persistence` | Compute persistence diagram (Rips filtration) |
| `tda_betti_numbers` | Compute Betti numbers at given epsilon |
| `tda_bottleneck_distance` | Bottleneck distance between two diagrams |
| `tda_barcodes_from_persistence` | Generate barcodes from persistence diagram |

## Compiler Requirements

- C11 compliant compiler (GCC, Clang)
- `-lm` (math library)
- `-Wall -Wextra -std=c11`

## License

MIT
