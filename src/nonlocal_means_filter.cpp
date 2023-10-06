


#include "nonlocal_means_filter.h"

NonlocalMeansFilter::NonlocalMeansFilter(
        const Buffer<uint8_t>& input, int patchSize, int searchWindowSize) :
        input(input), patchSize(patchSize), searchWindowSize(searchWindowSize),
        x("x"), y("y"), a("a"), b("b"), i("i"), j("j"),
        clamped("clamped"), weightedPixelDist("weightedPixelDist"),
        neighborhoodDifference("neighborhoodDifference"),
        neighborhoodWeight("neighborhoodWeight"),
        weightsSum("weightsSum"), newPixelValues("newPixelValues"),
        result("result") {
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
    weightedPixelDist(x, y, a, b, i, j) = gaussian(i + half_inner_neighborhood, j + half_inner_neighborhood) *
                                          pow(absd(clamped(x, y), clamped(a, b)), 2.0f);

    // The difference between two patches
    neighborhoodDifference(x, y, a, b) = sum(
            weightedPixelDist(x + r_inner.x, y + r_inner.y,
                              a + r_inner.x, b + r_inner.y,
                              r_inner.x, r_inner.y) // Pass the shift, which is required by the gaussian.
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
    Var x, y;

    Func gauss;
    gauss(x, y) = exp(
            -((x - (width - 1) / 2) * (x - (width - 1) / 2) + (y - (height - 1) / 2) * (y - (height - 1) / 2)) /
            (2.0f * sigma * sigma));

    RDom r(0, width, 0, height);
    Expr gaussSum = sum(gauss(r.x, r.y));

    Func normalized_gauss;
    normalized_gauss(x, y) = gauss(x, y) / gaussSum;

    return normalized_gauss;
}


void NonlocalMeansFilter::scheduleForCPU() {
    // The Gaussian can be precomputed entirely.
    // Otherwise, it will be recomputed for every patch
    // (with a quadratic complexity).
    gaussian.compute_root();

    result.compute_root().parallel(y);
}

bool NonlocalMeansFilter::scheduleForGPU() {
    return false;
}

