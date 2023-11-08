
# Optimizing Algorithms in Halide

This repository contains examples on how to use [Halide](https://halide-lang.org/),
a domain language for writing and optimizing array-processing algorithms.

The main idea behind Halide is to divide the implementation and scheduling of an algorithm.
Once the algorithm is implemented, Halide provides an API to set up the way the algorithm should be executed.
This allows for finding the optimal scheduling of algorithms in a fast and easy way.

Halide works only with array data.

## How to run examples

Make sure you have the Halide library installed. Compiline with cmake.

Running the **color-to-gray** conversion on a CPU with 100 repetitions:

```bash
$ halide_experiments -i images/4k_bird.jpg -r 100 -p colortogray -t cpu
```

Running the **non-local-means** filter on a GPU:

```bash
$ halide_experiments -i images/lena_grayscale.jpg -r 1 -p nonlocalmeans -t gpu
```

## Examples

Two examples are provided. A simple **Color-to-Gray Conversion** and relatively complex **Non-Local Means Filter**.


### Color-to-Gray Conversion

Color-to-gray conversion is a simple image processing task. It is implemented by a single function in Halide.

```Halide
Func result;
result(x, y) = cast<uint8_t>(0.299f * input(x, y, 0) +
                             0.587f * input(x, y, 1) +
                             0.114f * input(x, y, 2));
```

#### Optimizing for CPUs

There aren't many options in the case of a simple algorithm as this one. It easy to optimize in general.
Parallelization and vectorization is the best we can do here.

* 4K image (3840 × 2160)
* CPU with 4 logical cores

| **CPU Scheduling**                                                       | **Ex. time [ms]** |
|--------------------------------------------------------------------------|-------------------|
| result.compute_root();                                                   |             12.65 |
| result.compute_root().parallel(y);                                       |              5.86 |
| result.compute_root().vectorize(x);                                      |              4.47 |
| result.compute_root()<br>       .vectorize(x, 4)<br>       .parallel(y); |          **2.03** |

#### Optimizing for GPUs

* 4K image (3840 × 2160)
* Tesla T4

| **GPU Scheduling**                                                       | **Ex. time [ms]** |
|--------------------------------------------------------------------------|-------------------|
| result.compute_root();                                                   |             14.51 |
| result.split(x, xo, xi, 16)<br>             .split(y, yo, yi, 16)<br>             .reorder(xi, yi, xo, yo)<br>             .gpu_blocks(xo, yo)<br>             .gpu_threads(xi, yi);  |          **2.12** |

### Non-Local Means Filter

Non-local means is an algorithm in image processing for image denoising.

A. Buades, B. Coll and J. . -M. Morel, "A non-local algorithm for image denoising," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 60-65 vol. 2, doi: 10.1109/CVPR.2005.38.

The benefit of Halide is best observed on a more complex example, such as the Non-local Means Filter.
The implementaiton consists of many nested or dependent functions and loops. With dependant functions,
we can decide when they should execute, whether they should retain the immediate results in cache, or whether
they should be recomputed each time.

A typical C++ implementation would require many modifications to the implementation itself, which makes
the process of optimization error-prone, slower and more difficult.

The implementation is relatively long. Note that loops are modelled using the `RDom`.

```Halide
    clamped(x, y) = cast<float>(BoundaryConditions::repeat_edge(input)(x, y)) / 255;
    gaussian = createGaussian(patchSize, patchSize, weighingGaussianSigma);

    RDom r_inner(-patchSize / 2, patchSize,
                 -patchSize / 2, patchSize);
    RDom r_outer(-searchWindowSize / 2, searchWindowSize,
                 -searchWindowSize / 2, searchWindowSize);
    Expr half_inner_neighborhood = patchSize / 2;

    weightedPixelDist(x, y, a, b) = pow(absd(clamped(x, y), clamped(a, b)), 2.0f);

    neighborhoodDifference(x, y, a, b) = sum(
            gaussian(r_inner.x + half_inner_neighborhood,
                     r_inner.y + half_inner_neighborhood) *
            weightedPixelDist(x + r_inner.x, y + r_inner.y,
                              a + r_inner.x, b + r_inner.y)
    );

    areDifferentPoints(x, y, a, b) = (x - a != 0) || (y - b != 0);

    neighborhoodWeight(x, y, a, b) = exp(
            -neighborhoodDifference(x, y, a, b) / (h * h)
    ) * areDifferentPoints(x, y, a, b);

    weightsSum(x, y) += neighborhoodWeight(x, y, x + r_outer.x, y + r_outer.y);
    newPixelValues(x, y) +=
            neighborhoodWeight(x, y, x + r_outer.x, y + r_outer.y)
            * clamped(x + r_outer.x, y + r_outer.y);

    newPixelValuesNormalized(x, y) = newPixelValues(x, y) / weightsSum(x, y);
    result(x, y) = cast<uint8_t>(newPixelValuesNormalized(x, y) * 255);
```

#### Optimizing for CPUs

Optimized version for CPU is **151x** faster.

* 625x623 image
* Parameters:
  * searchWindowSize = 13
  * patchSize = 5
* Intel Xeon Gold 5218 2.30GHz (8 logical cores)

| **CPU Scheduling**                                                                                                                                                                                                                                                                                                                                                                                       | **Ex. time [ms]** |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| gaussian.compute_root();  <br> result.compute_root();                                                                                                                                                                                                                                                                                                                                                    |             71640 |
| gaussian.compute_root();  <br> result.compute_root().parallel(y);                                                                                                                                                                                                                                                                                                                                        |              8356 |
| gaussian.compute_root();<br><br> Var xi, yi, xo, yo, tile_index;<br> result.tile(x, y, xo, yo, xi, yi, 4, 4);<br> result.fuse(xo, yo, tile_index);<br> result.parallel(tile_index); <br> <br> Var x_vec, y_vec;  <br> result.split(xi, x_vec, xi, 4);  <br> result.split(yi, y_vec, yi, 4);  <br> result.vectorize(xi);  <br> result.vectorize(yi);  <br><br> neighborhoodWeight.compute_at(result, xi); |           **473** |

#### Optimizing for GPUs

Optimized version is **18x** then the optimized one for the CPU.

* Tesla T4

| **GPU Scheduling**                                                                                                                                                                                                                                                                                                                                                                                                 | **Ex. time [ms]** |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Var xi, yi, xo, yo; result.gpu_tile(x, y, xi, yi, xo, yo, 16, 16);                                                                                                                                                                                                                                                                                                                                                 |             147.8 |
| gaussian.compute_root();<br> <br> areDifferentPoints.compute_inline();<br> weightedPixelDist.compute_inline();<br> <br>  Var xi, yi, xo, yo;<br> result.gpu_tile(x, y, xi, yi, xo, yo, 16, 16);<br> <br>  Var x_vec, y_vec;<br> result.split(xi, x_vec, xi, 4);<br> result.split(yi, y_vec, yi, 4);<br> <br>  result.vectorize(xi);<br> result.vectorize(yi);<br> <br>  neighborhoodWeight.compute_at(result, xo); |          **25.9** |

4K image: 461.6 ms.

## License

The project is released under the MIT license.
