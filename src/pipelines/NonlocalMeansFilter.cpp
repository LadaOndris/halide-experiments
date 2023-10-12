


#include "pipelines/NonlocalMeansFilter.h"
#include "target.h"

NonlocalMeansFilter::NonlocalMeansFilter(
        const Buffer<uint8_t> &input, int patchSize, int searchWindowSize) :
        input(input), patchSize(patchSize), searchWindowSize(searchWindowSize),
        x("x"), y("y"), a("a"), b("b"), i("i"), j("j"),
        clamped("clamped"),
        gaussian("gaussian"),
        weightedPixelDist("weightedPixelDist"),
        neighborhoodDifference("neighborhoodDifference"),
        areDifferentPoints("areDifferentPoints"),
        neighborhoodWeight("neighborhoodWeight"),
        weightsSum("weightsSum"),
        newPixelValues("newPixelValues"),
        newPixelValuesNormalized("newPixelValuesNormalized") {
    implement();
}

void NonlocalMeansFilter::implement() {
    // Makes sure the image
    clamped(x, y) = cast<float>(BoundaryConditions::repeat_edge(input)(x, y)) / 255;

    gaussian = createGaussian(patchSize, patchSize, weighingGaussianSigma);

    RDom r_inner(-patchSize / 2, patchSize,
                 -patchSize / 2, patchSize);
    RDom r_outer(-searchWindowSize / 2, searchWindowSize,
                 -searchWindowSize / 2, searchWindowSize);
    Expr half_inner_neighborhood = patchSize / 2;

    // The difference between individual pixels
    weightedPixelDist(x, y, a, b) = pow(absd(clamped(x, y), clamped(a, b)), 2.0f);

    // The difference between two patches
    neighborhoodDifference(x, y, a, b) = sum(
            gaussian(r_inner.x + half_inner_neighborhood,
                     r_inner.y + half_inner_neighborhood) *
            weightedPixelDist(x + r_inner.x, y + r_inner.y,
                              a + r_inner.x, b + r_inner.y)
    );

    // Returns the value of one if the points differ.
    areDifferentPoints(x, y, a, b) = (x - a != 0) || (y - b != 0);

    // Find weights
    neighborhoodWeight(x, y, a, b) = exp(
            -neighborhoodDifference(x, y, a, b) / (h * h)
    ) * areDifferentPoints(x, y, a, b); // Weight for the pixel itself is 0.

    // Weights sum
    weightsSum(x, y) += neighborhoodWeight(x, y, x + r_outer.x, y + r_outer.y);

    // Weighing the surrounding pixels
    newPixelValues(x, y) +=
            neighborhoodWeight(x, y, x + r_outer.x, y + r_outer.y) * clamped(x + r_outer.x, y + r_outer.y);

    // Normalize by the total sum of weights
    newPixelValuesNormalized(x, y) = newPixelValues(x, y) / weightsSum(x, y);

    result(x, y) = cast<uint8_t>(newPixelValuesNormalized(x, y) * 255);
}

Func NonlocalMeansFilter::createGaussian(int width, int height, float sigma) {
    Var x("x"), y("y");

    Func gauss("gauss");
    gauss(x, y) = exp(
            -((x - (width - 1) / 2) * (x - (width - 1) / 2) + (y - (height - 1) / 2) * (y - (height - 1) / 2)) /
            (2.0f * sigma * sigma));

    RDom r(0, width, 0, height);
    Expr gaussSum = sum(gauss(r.x, r.y));

    Func normalized_gauss("normalized_gauss");
    normalized_gauss(x, y) = gauss(x, y) / gaussSum;

    return normalized_gauss;
}


void NonlocalMeansFilter::scheduleForCPU() {
    // The Gaussian can be precomputed entirely.
    // Otherwise, it will be recomputed for every patch
    // (with a quadratic complexity).
    gaussian.compute_root();

    neighborhoodWeight.compute_root().compute_with(neighborhoodDifference, a);

//    Var x_outer, y_outer, x_inner, y_inner, tile_index;
//    result.tile(x, y, x_outer, y_outer, x_inner, y_inner, 16, 16)
//            .fuse(x_outer, y_outer, tile_index)
//            .parallel(tile_index);


    result.compute_root();//.parallel(y);
}

bool NonlocalMeansFilter::scheduleForGPU() {
    Target target = find_gpu_target();
    if (!target.has_gpu_feature()) {
        return false;
    }

    Var xi, yi, xo, yo;
    result.gpu_tile(x, y, xi, yi, xo, yo, 16, 16);

    printf("Target: %s\n", target.to_string().c_str());
    result.compile_jit(target);

    return true;
}

